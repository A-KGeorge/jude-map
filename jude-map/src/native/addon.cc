#include <napi.h>
#include <uv.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// SIMD feature detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define SIMD_X86
#if defined(__AVX2__)
#define SIMD_AVX2
#define SIMD_AVX
#define SIMD_SSE3
#include <immintrin.h>
#elif defined(__AVX__)
#define SIMD_AVX
#define SIMD_SSE3
#include <immintrin.h>
#elif defined(__SSE3__)
#define SIMD_SSE3
#include <pmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define SIMD_SSE2
#include <emmintrin.h>
#include <xmmintrin.h>
#elif defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#define SIMD_SSE
#include <xmmintrin.h>
#endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
#define SIMD_NEON
#include <arm_neon.h>
#endif

#include "platform_mmap.h"
#include "segment.h"

/**
 * Optimized string-to-enum conversion.
 * Instead of multiple strcmp calls, we treat the first 8 bytes of the string
 * as a uint64_t (little-endian) to enable a high-speed compiler jump table.
 */
static DType dtype_from_string(const char *s) noexcept
{
    if (!s)
        return DType::UNKNOWN;

    uint64_t magic = 0;
    size_t len = std::strlen(s);
    // Copy up to 8 bytes into our magic number
    std::memcpy(&magic, s, len < 8 ? len : 8);

    switch (magic)
    {
    case 0x323374616f6c66:
        return DType::FLOAT32; // "float32"
    case 0x343674616f6c66:
        return DType::FLOAT64; // "float64"
    case 0x3233746e69:
        return DType::INT32; // "int32"
    case 0x3436746e69:
        return DType::INT64; // "int64"
    case 0x38746e6975:
        return DType::UINT8; // "uint8"
    case 0x38746e69:
        return DType::INT8; // "int8"
    case 0x3631746e6975:
        return DType::UINT16; // "uint16"
    case 0x3631746e69:
        return DType::INT16; // "int16"
    case 0x6c6f6f62:
        return DType::BOOL; // "bool"
    default:
        return DType::UNKNOWN;
    }
}

class SharedTensor : public Napi::ObjectWrap<SharedTensor>
{
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports)
    {
        Napi::Function func = DefineClass(env, "SharedTensor", {
                                                                   InstanceMethod<&SharedTensor::Write>("write"),
                                                                   InstanceMethod<&SharedTensor::Fill>("fill"),
                                                                   InstanceMethod<&SharedTensor::FillAsync>("fillAsync"),
                                                                   InstanceMethod<&SharedTensor::Read>("read"),
                                                                   InstanceMethod<&SharedTensor::ReadCopy>("readCopy"),
                                                                   InstanceMethod<&SharedTensor::ReadWait>("readWait"),
                                                                   InstanceMethod<&SharedTensor::ReadCopyWait>("readCopyWait"),
                                                                   InstanceMethod<&SharedTensor::Destroy>("destroy"),
                                                                   InstanceMethod<&SharedTensor::DestroyAsync>("destroyAsync"),
                                                                   InstanceMethod<&SharedTensor::DestroyAsyncWait>("destroyAsyncWait"),
                                                                   InstanceMethod<&SharedTensor::Pin>("pin"),
                                                                   InstanceMethod<&SharedTensor::Unpin>("unpin"),
                                                                   InstanceAccessor<&SharedTensor::ByteCapacity>("byteCapacity"),
                                                                   InstanceAccessor<&SharedTensor::IsPinned>("isPinned"),
                                                               });

        auto *ctor = new Napi::FunctionReference(Napi::Persistent(func));
        env.SetInstanceData<Napi::FunctionReference>(ctor);
        exports.Set("SharedTensor", func);
        return exports;
    }

    SharedTensor(const Napi::CallbackInfo &info) : Napi::ObjectWrap<SharedTensor>(info)
    {
        Napi::Env env = info.Env();
        env_ = env;

        if (info.Length() < 1 || !info[0].IsNumber())
        {
            Napi::TypeError::New(env, "SharedTensor(maxBytes: number)").ThrowAsJavaScriptException();
            return;
        }

        size_t max_bytes = static_cast<size_t>(info[0].As<Napi::Number>().DoubleValue());
        max_bytes_ = max_bytes; // store before page-rounding
        mapped_size_ = DATA_OFFSET + max_bytes;

        // Round up to page boundary
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        size_t page_size = static_cast<size_t>(si.dwPageSize);
#else
        size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        if (page_size == 0)
            page_size = 4096;
#endif
        mapped_size_ = ((mapped_size_ + page_size - 1) / page_size) * page_size;

        mapped_ = platform_mmap(mapped_size_);
        if (mapped_ == nullptr)
        {
            Napi::Error::New(env, "mmap failed").ThrowAsJavaScriptException();
            return;
        }

        // Placement-new the header to initialize atomics/seqlock
        new (mapped_) SegmentHeader();

        uv_loop_t *loop = nullptr;
        if (napi_get_uv_event_loop(env, &loop) != napi_ok || loop == nullptr)
        {
            unmap(false);
            Napi::Error::New(env, "Failed to get uv event loop").ThrowAsJavaScriptException();
            return;
        }

        read_async_ = new uv_async_t{};
        if (uv_async_init(loop, read_async_, &SharedTensor::OnReadAsyncWake) != 0)
        {
            delete read_async_;
            read_async_ = nullptr;
            unmap(false);
            Napi::Error::New(env, "Failed to initialize uv_async handle").ThrowAsJavaScriptException();
            return;
        }

        read_async_->data = this;

        uv_unref(reinterpret_cast<uv_handle_t *>(read_async_));
        // uv_unref marks this handle as weak — it won't prevent the event loop
        // from exiting when all other work is done. Without this, any surviving
        // SharedTensor instance (e.g. GC'd after loop teardown) causes the
        // uv__finish_close assertion on macOS and Linux.

        read_async_initialized_ = true;

        napi_add_env_cleanup_hook(env, &SharedTensor::OnEnvCleanup, this);
        cleanup_hook_registered_ = true;
    }

    ~SharedTensor()
    {
        if (cleanup_hook_registered_ && !env_closing_)
        {
            napi_remove_env_cleanup_hook(env_, &SharedTensor::OnEnvCleanup, this);
            cleanup_hook_registered_ = false;
        }
        unmap(false);
    }

private:
    enum class ReadStatus
    {
        Ok,
        Empty,
        RetryNeeded,
        Destroyed,
    };

    struct PendingRead
    {
        explicit PendingRead(Napi::Env env, bool copy_result)
            : deferred(Napi::Promise::Deferred::New(env)), copy(copy_result)
        {
        }

        Napi::Promise::Deferred deferred;
        bool copy;
    };

    struct FillWorkCtx;

    static constexpr uint32_t READ_SPIN_LIMIT = 16;

    Napi::Env env_ = nullptr;
    void *mapped_ = nullptr;
    size_t mapped_size_ = 0;
    bool pinned_ = false;  // true after successsful platform_mlock
    size_t max_bytes_ = 0; // user-requested capacity (pre-page-rounding)
    uv_async_t *read_async_ = nullptr;
    bool read_async_initialized_ = false;
    bool read_async_closing_ = false;
    bool env_closing_ = false;
    bool cleanup_hook_registered_ = false;
    std::mutex pending_reads_mu_;
    std::deque<std::shared_ptr<PendingRead>> pending_reads_;
    std::mutex fill_queue_mu_;
    std::deque<FillWorkCtx *> pending_fills_;
    bool fill_inflight_ = false;

    static void OnAsyncClose(uv_handle_t *handle)
    {
        delete reinterpret_cast<uv_async_t *>(handle);
    }

    static void OnEnvCleanup(void *data)
    {
        auto *self = static_cast<SharedTensor *>(data);
        if (!self)
            return;

        self->env_closing_ = true;
        self->cleanup_hook_registered_ = false;
        self->clear_pending_reads();
        self->close_async_handle();
    }

    void close_async_handle()
    {
        if (!read_async_ || read_async_closing_)
            return;

        if (uv_is_closing(reinterpret_cast<uv_handle_t *>(read_async_)))
        {
            read_async_closing_ = true;
            return;
        }

        read_async_->data = nullptr;
        read_async_closing_ = true;
        auto *h = read_async_;
        read_async_ = nullptr;
        uv_close(reinterpret_cast<uv_handle_t *>(h), &SharedTensor::OnAsyncClose);
    }

    void sync_async_ref_with_pending(bool has_pending)
    {
        if (!read_async_ || read_async_closing_)
            return;

        auto *h = reinterpret_cast<uv_handle_t *>(read_async_);
        if (has_pending)
        {
            if (!uv_has_ref(h))
                uv_ref(h);
        }
        else
        {
            if (uv_has_ref(h))
                uv_unref(h);
        }
    }

    void reject_pending_reads(const char *message)
    {
        Napi::HandleScope scope(env_);

        std::deque<std::shared_ptr<PendingRead>> pending;
        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            pending.swap(pending_reads_);
            sync_async_ref_with_pending(false);
        }

        for (const auto &item : pending)
        {
            item->deferred.Reject(Napi::Error::New(env_, message).Value());
        }
    }

    void clear_pending_reads()
    {
        std::lock_guard<std::mutex> lock(pending_reads_mu_);
        pending_reads_.clear();
        sync_async_ref_with_pending(false);
    }

    void enqueue_or_start_fill(FillWorkCtx *ctx)
    {
        FillWorkCtx *start_now = nullptr;
        {
            std::lock_guard<std::mutex> lock(fill_queue_mu_);
            if (fill_inflight_)
            {
                pending_fills_.push_back(ctx);
                return;
            }

            fill_inflight_ = true;
            start_now = ctx;
        }

        start_fill_work(start_now);
    }

    void on_fill_finished()
    {
        FillWorkCtx *next = nullptr;
        {
            std::lock_guard<std::mutex> lock(fill_queue_mu_);
            if (pending_fills_.empty())
            {
                fill_inflight_ = false;
                return;
            }

            next = pending_fills_.front();
            pending_fills_.pop_front();
        }

        start_fill_work(next);
    }

    static void OnReadAsyncWake(uv_async_t *handle)
    {
        auto *self = static_cast<SharedTensor *>(handle->data);
        if (self)
            self->flush_pending_reads();
    }

    void signal_pending_reads()
    {
        if (!read_async_ || read_async_closing_)
            return;

        bool has_pending = false;
        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            has_pending = !pending_reads_.empty();
        }

        if (has_pending)
            uv_async_send(read_async_);
    }

    Napi::Value make_result(Napi::Env env, const TensorMeta &meta, uint64_t seq, bool copy)
    {
        Napi::Object result = Napi::Object::New(env);
        Napi::Array shape_arr = Napi::Array::New(env, meta.ndim);
        for (uint32_t i = 0; i < meta.ndim; ++i)
        {
            shape_arr.Set(i, Napi::Number::New(env, static_cast<double>(meta.shape[i])));
        }

        result.Set("shape", shape_arr);
        result.Set("dtype", static_cast<uint32_t>(meta.dtype));
        result.Set("version", Napi::Number::New(env, static_cast<double>(seq)));

        uint8_t *data = segment_data_ptr(mapped_);
        if (copy)
        {
            result.Set("buffer", Napi::Buffer<uint8_t>::Copy(env, data, meta.byte_length));
        }
        else
        {
            result.Set(
                "buffer",
                Napi::ArrayBuffer::New(env, data, meta.byte_length,
                                       [](Napi::Env, void *)
                                       {
                                           // Lifetime is owned by SharedTensor's mmap/unmap.
                                       }));
        }

        return result;
    }

    Napi::Value try_read(Napi::Env env, bool copy, uint32_t spin_limit, ReadStatus &status)
    {
        if (!mapped_)
        {
            status = ReadStatus::Destroyed;
            return env.Null();
        }

        auto *hdr = reinterpret_cast<SegmentHeader *>(mapped_);
        TensorMeta meta;
        uint64_t stable_seq = 0;
        bool got_snapshot = false;

        for (uint32_t attempt = 0; attempt < spin_limit; ++attempt)
        {
            const uint64_t seq0 = hdr->seqlock.sequence.load(std::memory_order_acquire);
            if (seq0 & 1u)
                continue;

            meta = hdr->meta;
            std::atomic_thread_fence(std::memory_order_acquire);

            const uint64_t seq1 = hdr->seqlock.sequence.load(std::memory_order_relaxed);
            if (seq0 != seq1)
                continue;

            stable_seq = seq0;
            got_snapshot = true;
            break;
        }

        if (!got_snapshot)
        {
            status = ReadStatus::RetryNeeded;
            return env.Null();
        }

        if (meta.byte_length == 0)
        {
            status = ReadStatus::Empty;
            return env.Null();
        }

        status = ReadStatus::Ok;
        return make_result(env, meta, stable_seq, copy);
    }

    void flush_pending_reads()
    {
        if (env_closing_)
        {
            clear_pending_reads();
            return;
        }

        Napi::HandleScope scope(env_);

        std::deque<std::shared_ptr<PendingRead>> pending;
        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            pending.swap(pending_reads_);
            sync_async_ref_with_pending(false);
        }

        if (pending.empty())
            return;

        std::deque<std::shared_ptr<PendingRead>> still_waiting;
        bool needs_retry_kick = false;
        for (const auto &item : pending)
        {
            if (!mapped_)
            {
                item->deferred.Reject(Napi::Error::New(env_, "Destroyed").Value());
                continue;
            }

            ReadStatus status = ReadStatus::RetryNeeded;
            Napi::Value value = try_read(env_, item->copy, READ_SPIN_LIMIT, status);
            if (status == ReadStatus::Ok)
            {
                item->deferred.Resolve(value);
            }
            else
            {
                still_waiting.push_back(item);
                if (status == ReadStatus::RetryNeeded)
                    needs_retry_kick = true;
            }
        }

        if (!still_waiting.empty())
        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            for (auto &item : still_waiting)
                pending_reads_.push_back(std::move(item));
            sync_async_ref_with_pending(true);
        }

        // If we woke during an in-progress commit, retry once more without
        // waiting for another writer signal to avoid stranded pending promises.
        if (needs_retry_kick && read_async_)
            uv_async_send(read_async_);
    }

    void teardown_mapping(bool reject_waiters, void *&addr, size_t &size)
    {
        if (reject_waiters && !env_closing_)
            reject_pending_reads("Destroyed");
        else
            clear_pending_reads();

        close_async_handle();

        addr = nullptr;
        size = 0;
        if (!mapped_)
            return;

        addr = mapped_;
        size = mapped_size_;

        if (pinned_)
        {
            platform_munlock(addr, size);
            pinned_ = false;
        }

        reinterpret_cast<SegmentHeader *>(mapped_)->~SegmentHeader();
        mapped_ = nullptr;
    }

    void unmap(bool reject_waiters)
    {
        void *addr = nullptr;
        size_t size = 0;
        teardown_mapping(reject_waiters, addr, size);
        if (addr)
        {
#ifdef _WIN32
            // On Windows, a separate release-hint pass can be slower than a
            // direct unmap in the synchronous path.
            platform_munmap(addr, size);
#else
            platform_release_hint(addr, size);
            platform_munmap(addr, size);
#endif
        }
    }

    // -----------------------------------------------------------------------
    // unmap_teardown / unmap_release
    //
    // Split unmap() into two phases for destroyAsync():
    //
    //   teardown  — synchronous, event loop thread:
    //               reject pending reads, close uv_async handle, null mapped_,
    //               call SegmentHeader destructor, call platform_release_hint.
    //               After this returns, no further reads/writes can succeed.
    //
    //   release   — slow, offloaded to uv_queue_work thread:
    //               platform_munmap — walks page tables, flushes TLBs.
    //               Even with the release hint, this takes O(mapped_size).
    //
    // The split ensures the event loop is free during the slow phase.
    // -----------------------------------------------------------------------
    struct UnmapCtx
    {
        uv_work_t req; // must be first
        void *addr;
        size_t size;
        bool resolve_on_after;
        Napi::Promise::Deferred deferred;
        Napi::ObjectReference self_ref;

        UnmapCtx(Napi::Env env, void *a, size_t s, bool resolve_after)
            : req{}, addr(a), size(s), resolve_on_after(resolve_after),
              deferred(Napi::Promise::Deferred::New(env))
        {
        }
    };

    static void OnUnmapWork(uv_work_t *req)
    {
        // Runs on libuv thread pool — NO V8 access.
        auto *ctx = reinterpret_cast<UnmapCtx *>(req);
#ifndef _WIN32
        platform_release_hint(ctx->addr, ctx->size);
#endif
        platform_munmap(ctx->addr, ctx->size);
    }

    static void OnUnmapAfter(uv_work_t *req, int /*status*/)
    {
        // Runs back on event loop thread.
        auto *ctx = reinterpret_cast<UnmapCtx *>(req);
        if (ctx->resolve_on_after)
        {
            Napi::Env env = ctx->deferred.Env();
            Napi::HandleScope scope(env);
            ctx->deferred.Resolve(env.Undefined());
            ctx->self_ref.Unref();
        }
        delete ctx;
    }

    void Destroy(const Napi::CallbackInfo &info) { unmap(true); }

    // -----------------------------------------------------------------------
    // destroyAsync() → Promise<void>
    //
    // Non-blocking variant of destroy() for the main event loop thread.
    //
    // Phase 1 (event loop, synchronous):
    //   - Reject all pending readWait() promises
    //   - Close the uv_async handle
    //   - Call SegmentHeader destructor
    //   - Call platform_release_hint (fast — just advises the OS)
    //   - Null mapped_ so all subsequent operations fail immediately
    //
    // Phase 2 (libuv thread pool, async):
    //   - platform_munmap — the slow page-table walk (background)
    //
    // destroyAsync() resolves immediately after logical teardown. Physical
    // virtual-memory release continues in background on libuv worker threads.
    // This minimizes event-loop-visible cleanup latency for large tensors.
    // -----------------------------------------------------------------------
    Napi::Value DestroyAsync(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();

        if (!mapped_)
        {
            // Already destroyed — return a resolved Promise.
            auto deferred = Napi::Promise::Deferred::New(env);
            deferred.Resolve(env.Undefined());
            return deferred.Promise();
        }

        void *addr = nullptr;
        size_t size = 0;
        teardown_mapping(true, addr, size);
        if (!addr)
        {
            auto deferred = Napi::Promise::Deferred::New(env);
            deferred.Resolve(env.Undefined());
            return deferred.Promise();
        }

        // -- Phase 2: async munmap on thread pool --
        auto *ctx = new UnmapCtx(env, addr, size, false);

        uv_loop_t *loop = nullptr;
        napi_get_uv_event_loop(env, &loop);

        if (!loop || uv_queue_work(loop, &ctx->req,
                                   OnUnmapWork, OnUnmapAfter) != 0)
        {
            // Fallback: do it synchronously if queue fails.
            platform_munmap(addr, size);
            delete ctx;
        }
        else
        {
            // Work accepted; callback owns ctx lifetime.
        }

        auto deferred = Napi::Promise::Deferred::New(env);
        deferred.Resolve(env.Undefined());
        return deferred.Promise();
    }

    Napi::Value DestroyAsyncWait(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();

        if (!mapped_)
        {
            auto deferred = Napi::Promise::Deferred::New(env);
            deferred.Resolve(env.Undefined());
            return deferred.Promise();
        }

        void *addr = nullptr;
        size_t size = 0;
        teardown_mapping(true, addr, size);
        if (!addr)
        {
            auto deferred = Napi::Promise::Deferred::New(env);
            deferred.Resolve(env.Undefined());
            return deferred.Promise();
        }

        auto *ctx = new UnmapCtx(env, addr, size, true);
        ctx->self_ref = Napi::ObjectReference::New(info.This().As<Napi::Object>(), 1);

        uv_loop_t *loop = nullptr;
        napi_get_uv_event_loop(env, &loop);

        if (!loop || uv_queue_work(loop, &ctx->req,
                                   OnUnmapWork, OnUnmapAfter) != 0)
        {
            platform_munmap(addr, size);
            ctx->self_ref.Unref();
            ctx->deferred.Resolve(env.Undefined());
            auto promise = ctx->deferred.Promise();
            delete ctx;
            return promise;
        }

        return ctx->deferred.Promise();
    }

    // -----------------------------------------------------------------------
    // pin() → boolean
    //
    // Page-locks the entire mapped region so CUDA DMA can read it directly
    // without a staging copy (cudaMemcpy H2D zero-copy path).
    //
    // Must be called before handing the data pointer to TF_NewTensor.
    // Requires elevated privileges on some systems:
    //   Linux: CAP_IPC_LOCK or sufficient RLIMIT_MEMLOCK
    //   Windows: SeLockMemoryPrivilege or within working-set quota
    // -----------------------------------------------------------------------
    Napi::Value Pin(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        if (!mapped_)
        {
            Napi::Error::New(env, "Destroyed").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        if (pinned_)
            return Napi::Boolean::New(env, true);

        pinned_ = platform_mlock(mapped_, mapped_size_);
        return Napi::Boolean::New(env, pinned_);
    }

    // -----------------------------------------------------------------------
    // unpin() — releases the page lock, allows OS to swap pages again.
    // -----------------------------------------------------------------------
    Napi::Value Unpin(const Napi::CallbackInfo &info)
    {
        if (mapped_ && pinned_)
        {
            platform_munlock(mapped_, mapped_size_);
            pinned_ = false;
        }
        return info.Env().Undefined();
    }

    // -----------------------------------------------------------------------
    // get isPinned → boolean
    // -----------------------------------------------------------------------
    Napi::Value IsPinned(const Napi::CallbackInfo &info)
    {
        return Napi::Boolean::New(info.Env(), pinned_);
    }

    // Returns 0 for unknown dtype — caller checks before write_begin.
    static size_t itemsize_noexcept(DType dt) noexcept
    {
        switch (dt)
        {
        case DType::FLOAT32:
            return 4;
        case DType::FLOAT64:
            return 8;
        case DType::INT32:
            return 4;
        case DType::INT64:
            return 8;
        case DType::UINT8:
            return 1;
        case DType::INT8:
            return 1;
        case DType::UINT16:
            return 2;
        case DType::INT16:
            return 2;
        case DType::BOOL:
            return 1;
        default:
            return 0;
        }
    }

    // -----------------------------------------------------------------------
    // typed_fill<T>(dest, n_elems, raw_value)
    //
    // Fills n_elems elements of type T at dest with a value cast from the
    // double that JS passed in. Uses a simple loop — the compiler will
    // auto-vectorise this with AVX2/NEON given the -O3 + arch flags in
    // binding.gyp, so no manual SIMD needed.
    // -----------------------------------------------------------------------
    template <typename T>
    static void typed_fill(void *dest, size_t n_elems, double raw_value) noexcept
    {
        T val = static_cast<T>(raw_value);
        T *ptr = reinterpret_cast<T *>(dest);
        for (size_t i = 0; i < n_elems; ++i)
            ptr[i] = val;
    }

    static void fill_u8(void *dest, size_t n_elems, uint8_t value) noexcept
    {
        std::memset(dest, static_cast<int>(value), n_elems);
    }

    static void fill_i8(void *dest, size_t n_elems, int8_t value) noexcept
    {
        std::memset(dest, static_cast<unsigned char>(value), n_elems);
    }

    static void fill_u16(void *dest, size_t n_elems, uint16_t value) noexcept
    {
#if defined(SIMD_AVX2)
        auto *ptr = reinterpret_cast<uint16_t *>(dest);
        size_t i = 0;
        const __m256i v = _mm256_set1_epi16(static_cast<short>(value));
        for (; i + 16 <= n_elems; i += 16)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + i), v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_SSE2)
        auto *ptr = reinterpret_cast<uint16_t *>(dest);
        size_t i = 0;
        const __m128i v = _mm_set1_epi16(static_cast<short>(value));
        for (; i + 8 <= n_elems; i += 8)
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + i), v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_NEON)
        auto *ptr = reinterpret_cast<uint16_t *>(dest);
        size_t i = 0;
        const uint16x8_t v = vdupq_n_u16(value);
        for (; i + 8 <= n_elems; i += 8)
            vst1q_u16(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#else
        typed_fill<uint16_t>(dest, n_elems, static_cast<double>(value));
#endif
    }

    static void fill_i16(void *dest, size_t n_elems, int16_t value) noexcept
    {
#if defined(SIMD_AVX2)
        auto *ptr = reinterpret_cast<int16_t *>(dest);
        size_t i = 0;
        const __m256i v = _mm256_set1_epi16(value);
        for (; i + 16 <= n_elems; i += 16)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + i), v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_SSE2)
        auto *ptr = reinterpret_cast<int16_t *>(dest);
        size_t i = 0;
        const __m128i v = _mm_set1_epi16(value);
        for (; i + 8 <= n_elems; i += 8)
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + i), v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_NEON)
        auto *ptr = reinterpret_cast<int16_t *>(dest);
        size_t i = 0;
        const int16x8_t v = vdupq_n_s16(value);
        for (; i + 8 <= n_elems; i += 8)
            vst1q_s16(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#else
        typed_fill<int16_t>(dest, n_elems, static_cast<double>(value));
#endif
    }

    static void fill_f32(void *dest, size_t n_elems, float value) noexcept
    {
#if defined(SIMD_AVX)
        auto *ptr = reinterpret_cast<float *>(dest);
        size_t i = 0;
        const __m256 v = _mm256_set1_ps(value);
        for (; i + 8 <= n_elems; i += 8)
            _mm256_storeu_ps(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_SSE2)
        auto *ptr = reinterpret_cast<float *>(dest);
        size_t i = 0;
        const __m128 v = _mm_set1_ps(value);
        for (; i + 4 <= n_elems; i += 4)
            _mm_storeu_ps(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_NEON)
        auto *ptr = reinterpret_cast<float *>(dest);
        size_t i = 0;
        const float32x4_t v = vdupq_n_f32(value);
        for (; i + 4 <= n_elems; i += 4)
            vst1q_f32(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#else
        typed_fill<float>(dest, n_elems, static_cast<double>(value));
#endif
    }

    static void fill_f64(void *dest, size_t n_elems, double value) noexcept
    {
#if defined(SIMD_AVX)
        auto *ptr = reinterpret_cast<double *>(dest);
        size_t i = 0;
        const __m256d v = _mm256_set1_pd(value);
        for (; i + 4 <= n_elems; i += 4)
            _mm256_storeu_pd(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_SSE2)
        auto *ptr = reinterpret_cast<double *>(dest);
        size_t i = 0;
        const __m128d v = _mm_set1_pd(value);
        for (; i + 2 <= n_elems; i += 2)
            _mm_storeu_pd(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_NEON) && defined(__aarch64__)
        auto *ptr = reinterpret_cast<double *>(dest);
        size_t i = 0;
        const float64x2_t v = vdupq_n_f64(value);
        for (; i + 2 <= n_elems; i += 2)
            vst1q_f64(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#else
        typed_fill<double>(dest, n_elems, value);
#endif
    }

    static void fill_i32(void *dest, size_t n_elems, int32_t value) noexcept
    {
#if defined(SIMD_AVX2)
        auto *ptr = reinterpret_cast<int32_t *>(dest);
        size_t i = 0;
        const __m256i v = _mm256_set1_epi32(value);
        for (; i + 8 <= n_elems; i += 8)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + i), v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_SSE2)
        auto *ptr = reinterpret_cast<int32_t *>(dest);
        size_t i = 0;
        const __m128i v = _mm_set1_epi32(value);
        for (; i + 4 <= n_elems; i += 4)
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + i), v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_NEON)
        auto *ptr = reinterpret_cast<int32_t *>(dest);
        size_t i = 0;
        const int32x4_t v = vdupq_n_s32(value);
        for (; i + 4 <= n_elems; i += 4)
            vst1q_s32(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#else
        typed_fill<int32_t>(dest, n_elems, static_cast<double>(value));
#endif
    }

    static void fill_i64(void *dest, size_t n_elems, int64_t value) noexcept
    {
#if defined(SIMD_AVX2)
        auto *ptr = reinterpret_cast<int64_t *>(dest);
        size_t i = 0;
        const __m256i v = _mm256_set1_epi64x(value);
        for (; i + 4 <= n_elems; i += 4)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + i), v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_SSE2)
        auto *ptr = reinterpret_cast<int64_t *>(dest);
        size_t i = 0;
        const __m128i v = _mm_set_epi64x(value, value);
        for (; i + 2 <= n_elems; i += 2)
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + i), v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#elif defined(SIMD_NEON)
        auto *ptr = reinterpret_cast<int64_t *>(dest);
        size_t i = 0;
        const int64x2_t v = vdupq_n_s64(value);
        for (; i + 2 <= n_elems; i += 2)
            vst1q_s64(ptr + i, v);
        for (; i < n_elems; ++i)
            ptr[i] = value;
#else
        typed_fill<int64_t>(dest, n_elems, static_cast<double>(value));
#endif
    }

    static void parallel_fill(void *dest, size_t n_elems, size_t itemsize,
                              DType dtype, double fill_value) noexcept
    {
        if (itemsize == 0 || n_elems == 0)
            return;

        constexpr size_t PARALLEL_FILL_MIN_BYTES = 4ULL * 1024ULL * 1024ULL;
        constexpr unsigned MAX_FILL_THREADS = 8;

        const size_t total_bytes = n_elems * itemsize;
        if (total_bytes < PARALLEL_FILL_MIN_BYTES)
        {
            dispatch_fill(dest, n_elems, dtype, fill_value);
            return;
        }

        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0)
            hw = 1;

        unsigned threads = std::min<unsigned>(MAX_FILL_THREADS, hw);
        if (threads <= 1 || n_elems < threads)
        {
            dispatch_fill(dest, n_elems, dtype, fill_value);
            return;
        }

        std::vector<std::thread> workers;
        workers.reserve(threads - 1);

        uint8_t *base = reinterpret_cast<uint8_t *>(dest);
        const size_t chunk = n_elems / threads;
        size_t start = 0;

        for (unsigned t = 0; t < threads - 1; ++t)
        {
            const size_t end = start + chunk;
            workers.emplace_back([=]()
                                 {
                uint8_t* ptr = base + (start * itemsize);
                dispatch_fill(ptr, end - start, dtype, fill_value); });
            start = end;
        }

        dispatch_fill(base + (start * itemsize), n_elems - start, dtype, fill_value);

        for (auto &w : workers)
            w.join();
    }

    // -----------------------------------------------------------------------
    // fill(shape: number[], dtype: DType, value: number) → void
    //
    // Writes `value` into every element of the tensor defined by `shape` and
    // `dtype`, entirely inside C++. No V8 buffer is allocated — the data
    // materialises directly in the mmap region, bypassing the V8 heap ceiling.
    //
    // This is the correct path for tensors larger than ~2 GB (e.g. 10 GB LLM
    // activation buffers) where Buffer.allocUnsafe would throw RangeError.
    //
    // Signature mirrors write() for the shape/dtype args so callers can swap
    // between fill() and write() without restructuring their pipeline setup.
    // -----------------------------------------------------------------------
    Napi::Value Fill(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        if (!mapped_)
        {
            Napi::Error::New(env, "Destroyed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        if (info.Length() < 3)
        {
            Napi::TypeError::New(env, "fill(shape, dtype, value)").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // -- shape --
        if (!info[0].IsArray())
        {
            Napi::TypeError::New(env, "shape must be an array").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        Napi::Array shape_arr = info[0].As<Napi::Array>();
        uint32_t ndim = shape_arr.Length();
        if (ndim == 0 || ndim > MAX_DIMS)
        {
            Napi::RangeError::New(env, "shape rank must be 1..MAX_DIMS").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // -- dtype --
        DType dtype;
        if (info[1].IsNumber())
            dtype = static_cast<DType>(info[1].As<Napi::Number>().Uint32Value());
        else
        {
            std::string s = info[1].As<Napi::String>().Utf8Value();
            dtype = dtype_from_string(s.c_str());
        }
        if (dtype == DType::UNKNOWN)
        {
            Napi::TypeError::New(env, "Unsupported dtype").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // -- value --
        if (!info[2].IsNumber())
        {
            Napi::TypeError::New(env, "value must be a number").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        double fill_value = info[2].As<Napi::Number>().DoubleValue();

        // -- element count from shape --
        uint64_t shape[MAX_DIMS] = {};
        size_t n_elems = 1;
        for (uint32_t i = 0; i < ndim; ++i)
        {
            uint64_t dim = static_cast<uint64_t>(
                shape_arr.Get(i).As<Napi::Number>().DoubleValue());
            shape[i] = dim;
            n_elems *= static_cast<size_t>(dim);
        }

        size_t itemsize = itemsize_noexcept(dtype);
        if (itemsize == 0)
        {
            Napi::TypeError::New(env, "Cannot determine itemsize for dtype").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        size_t byte_length = n_elems * itemsize;

        if (byte_length > max_bytes_)
        {
            Napi::RangeError::New(env, "tensor byte size exceeds segment capacity").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // -- fill directly in mmap memory --
        void *dest = segment_data_ptr(mapped_);
        auto *hdr = reinterpret_cast<SegmentHeader *>(mapped_);

        hdr->seqlock.write_begin();
        parallel_fill(dest, n_elems, itemsize, dtype, fill_value);
        hdr->meta.ndim = ndim;
        hdr->meta.dtype = dtype;
        hdr->meta.byte_length = byte_length;
        std::memcpy(hdr->meta.shape, shape, sizeof(shape));
        hdr->seqlock.write_end();
        signal_pending_reads();
        return env.Undefined();
    }

    // -----------------------------------------------------------------------
    // dispatch_fill — typed fill dispatcher for one contiguous chunk.
    //
    // Called by both Fill (sync) and the uv_queue_work thread in FillAsync.
    // No V8 access — safe to call from any thread.
    // -----------------------------------------------------------------------
    static void dispatch_fill(void *dest, size_t n_elems,
                              DType dtype, double fill_value) noexcept
    {
        switch (dtype)
        {
        case DType::FLOAT32:
            fill_f32(dest, n_elems, static_cast<float>(fill_value));
            break;
        case DType::FLOAT64:
            fill_f64(dest, n_elems, fill_value);
            break;
        case DType::INT32:
            fill_i32(dest, n_elems, static_cast<int32_t>(fill_value));
            break;
        case DType::INT64:
            fill_i64(dest, n_elems, static_cast<int64_t>(fill_value));
            break;
        case DType::UINT8:
            fill_u8(dest, n_elems, static_cast<uint8_t>(fill_value));
            break;
        case DType::INT8:
            fill_i8(dest, n_elems, static_cast<int8_t>(fill_value));
            break;
        case DType::UINT16:
            fill_u16(dest, n_elems, static_cast<uint16_t>(fill_value));
            break;
        case DType::INT16:
            fill_i16(dest, n_elems, static_cast<int16_t>(fill_value));
            break;
        case DType::BOOL:
            fill_u8(dest, n_elems, fill_value != 0.0 ? 1 : 0);
            break;
        default:
            break;
        }
    }

    // -----------------------------------------------------------------------
    // fillAsync(shape, dtype, value) → Promise<void>
    //
    // Non-blocking variant of fill() for use on the main event loop thread.
    //
    // Pushes the C++ fill work onto libuv's thread pool via uv_queue_work.
    // The event loop is free to process I/O, timers, and readWait() callbacks
    // while the fill runs. The returned Promise resolves when the seqlock
    // commit is complete and all parked readWait() callers have been woken.
    //
    // Intended split:
    //   Main thread  → fillAsync()   non-blocking, Promise-based
    //   Worker thread → fill()       synchronous, isolated event loop
    //
    // Design notes:
    //   - write_begin() is called on the event loop thread before queuing,
    //     so readers see the seqlock as in-progress immediately (odd counter).
    //   - The thread pool thread does the actual fill — no V8 access there.
    //   - write_end() + signal_pending_reads() happen in the after-callback,
    //     which runs back on the event loop thread.
    //   - The work ctx stores only plain C++ data and the output pointers.
    //   - No V8 handles are touched from the worker thread.
    // -----------------------------------------------------------------------

    // Work context — allocated on heap, owned by uv_queue_work lifecycle.
    struct FillWorkCtx
    {
        uv_work_t req; // must be first — libuv casts req* → FillWorkCtx*

        // Fill parameters (copied from JS args — no V8 refs after queue)
        void *dest;
        size_t n_elems;
        size_t itemsize;
        DType dtype;
        double fill_value;

        // Segment metadata to commit after fill
        SegmentHeader *hdr;
        uint32_t ndim;
        uint64_t shape[MAX_DIMS];
        size_t byte_length;

        // Promise to resolve on completion
        Napi::Promise::Deferred deferred;

        // Keeps JS wrapper/object alive until OnFillAfter finishes.
        Napi::ObjectReference self_ref;

        // Back-pointer for signal_pending_reads() in after-callback
        SharedTensor *owner;

        explicit FillWorkCtx(Napi::Env env)
            : req{}, dest(nullptr), n_elems(0), itemsize(0), dtype(DType::UNKNOWN), fill_value(0.0),
              hdr(nullptr), ndim(0), shape{}, byte_length(0),
              deferred(Napi::Promise::Deferred::New(env)), owner(nullptr)
        {
        }
    };

    void start_fill_work(FillWorkCtx *ctx)
    {
        Napi::Env env = ctx->deferred.Env();

        if (!mapped_)
        {
            ctx->deferred.Reject(Napi::Error::New(env, "Destroyed").Value());
            ctx->self_ref.Unref();
            delete ctx;
            on_fill_finished();
            return;
        }

        ctx->hdr->seqlock.write_begin();

        uv_loop_t *loop = nullptr;
        if (napi_get_uv_event_loop(env, &loop) != napi_ok || loop == nullptr)
        {
            ctx->hdr->seqlock.write_end();
            ctx->deferred.Reject(Napi::Error::New(env, "Failed to get uv event loop").Value());
            ctx->self_ref.Unref();
            delete ctx;
            on_fill_finished();
            return;
        }

        int rc = uv_queue_work(loop, &ctx->req, &SharedTensor::OnFillWork, &SharedTensor::OnFillAfter);
        if (rc != 0)
        {
            ctx->hdr->seqlock.write_end();
            ctx->deferred.Reject(Napi::Error::New(env, "Failed to queue fillAsync work").Value());
            ctx->self_ref.Unref();
            delete ctx;
            on_fill_finished();
            return;
        }
    }

    // Runs on libuv thread pool thread — NO V8 access allowed.
    static void OnFillWork(uv_work_t *req)
    {
        auto *ctx = reinterpret_cast<FillWorkCtx *>(req);
        parallel_fill(ctx->dest, ctx->n_elems, ctx->itemsize, ctx->dtype, ctx->fill_value);
    }

    // Runs back on event loop thread — V8 access safe.
    static void OnFillAfter(uv_work_t *req, int /*status*/)
    {
        auto *ctx = reinterpret_cast<FillWorkCtx *>(req);

        Napi::Env env = ctx->deferred.Env();
        Napi::HandleScope scope(env);

        // Commit the seqlock — makes the filled data visible to readers.
        ctx->hdr->meta.ndim = ctx->ndim;
        ctx->hdr->meta.dtype = ctx->dtype;
        ctx->hdr->meta.byte_length = ctx->byte_length;
        std::memcpy(ctx->hdr->meta.shape, ctx->shape, sizeof(ctx->shape));
        ctx->hdr->seqlock.write_end();

        // Wake any parked readWait() / readCopyWait() callers.
        if (ctx->owner)
            ctx->owner->signal_pending_reads();

        ctx->deferred.Resolve(env.Undefined());
        ctx->self_ref.Unref();

        SharedTensor *owner = ctx->owner;
        delete ctx;

        if (owner)
            owner->on_fill_finished();
    }

    Napi::Value FillAsync(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        if (!mapped_)
        {
            Napi::Error::New(env, "Destroyed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        if (info.Length() < 3)
        {
            Napi::TypeError::New(env, "fillAsync(shape, dtype, value)").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // -- parse args (identical to Fill) --
        if (!info[0].IsArray())
        {
            Napi::TypeError::New(env, "shape must be an array").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        Napi::Array shape_arr = info[0].As<Napi::Array>();
        uint32_t ndim = shape_arr.Length();
        if (ndim == 0 || ndim > MAX_DIMS)
        {
            Napi::RangeError::New(env, "shape rank must be 1..MAX_DIMS").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        DType dtype;
        if (info[1].IsNumber())
            dtype = static_cast<DType>(info[1].As<Napi::Number>().Uint32Value());
        else
        {
            std::string s = info[1].As<Napi::String>().Utf8Value();
            dtype = dtype_from_string(s.c_str());
        }
        if (dtype == DType::UNKNOWN)
        {
            Napi::TypeError::New(env, "Unsupported dtype").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        if (!info[2].IsNumber())
        {
            Napi::TypeError::New(env, "value must be a number").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        double fill_value = info[2].As<Napi::Number>().DoubleValue();

        uint64_t shape[MAX_DIMS] = {};
        size_t n_elems = 1;
        for (uint32_t i = 0; i < ndim; ++i)
        {
            uint64_t dim = static_cast<uint64_t>(
                shape_arr.Get(i).As<Napi::Number>().DoubleValue());
            shape[i] = dim;
            n_elems *= static_cast<size_t>(dim);
        }

        size_t itemsize = itemsize_noexcept(dtype);
        if (itemsize == 0)
        {
            Napi::TypeError::New(env, "Cannot determine itemsize for dtype").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        size_t byte_length = n_elems * itemsize;
        if (byte_length > max_bytes_)
        {
            Napi::RangeError::New(env, "tensor byte size exceeds segment capacity").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // -- build work context --
        auto *ctx = new FillWorkCtx(env);
        ctx->dest = segment_data_ptr(mapped_);
        ctx->n_elems = n_elems;
        ctx->itemsize = itemsize;
        ctx->dtype = dtype;
        ctx->fill_value = fill_value;
        ctx->hdr = reinterpret_cast<SegmentHeader *>(mapped_);
        ctx->ndim = ndim;
        ctx->byte_length = byte_length;
        ctx->self_ref = Napi::ObjectReference::New(info.This().As<Napi::Object>(), 1);
        ctx->owner = this;
        std::memcpy(ctx->shape, shape, sizeof(shape));

        enqueue_or_start_fill(ctx);

        return ctx->deferred.Promise();
    }

    Napi::Value Write(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        if (!mapped_)
        {
            Napi::Error::New(env, "Destroyed").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // 1. Resolve DType (Fast path)
        DType dtype;
        if (info[1].IsNumber())
        {
            // If the TS wrapper passes an integer ID
            dtype = static_cast<DType>(info[1].As<Napi::Number>().Uint32Value());
        }
        else
        {
            // Fallback for string input
            std::string s = info[1].As<Napi::String>().Utf8Value();
            dtype = dtype_from_string(s.c_str());
        }

        if (dtype == DType::UNKNOWN)
        {
            Napi::TypeError::New(env, "Unsupported dtype").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // 2. Resolve Buffers/Shape
        Napi::Array shape_arr = info[0].As<Napi::Array>();
        uint32_t ndim = shape_arr.Length();
        if (ndim > MAX_DIMS)
        {
            Napi::RangeError::New(env, "shape rank exceeds MAX_DIMS").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        uint8_t *src_ptr = nullptr;
        size_t src_size = 0;
        if (info[2].IsArrayBuffer())
        {
            auto ab = info[2].As<Napi::ArrayBuffer>();
            src_ptr = reinterpret_cast<uint8_t *>(ab.Data());
            src_size = ab.ByteLength();
        }
        else
        {
            auto ta = info[2].As<Napi::TypedArray>();
            src_ptr = reinterpret_cast<uint8_t *>(ta.ArrayBuffer().Data()) + ta.ByteOffset();
            src_size = ta.ByteLength();
        }

        const size_t capacity = max_bytes_;
        if (src_size > capacity)
        {
            Napi::RangeError::New(env, "tensor byte size exceeds segment capacity").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // 3. Metadata commit with Seqlock
        auto *hdr = reinterpret_cast<SegmentHeader *>(mapped_);
        hdr->seqlock.write_begin(); // Corrected member name

        hdr->meta.ndim = ndim;
        hdr->meta.dtype = dtype;
        hdr->meta.byte_length = src_size;
        for (uint32_t i = 0; i < ndim; ++i)
        {
            hdr->meta.shape[i] = shape_arr.Get(i).As<Napi::Number>().Uint32Value();
        }
        std::memcpy(segment_data_ptr(mapped_), src_ptr, src_size); // Fixed typo

        hdr->seqlock.write_end();
        signal_pending_reads();
        return env.Undefined();
    }

    Napi::Value read_internal(const Napi::CallbackInfo &info, bool copy)
    {
        Napi::Env env = info.Env();
        ReadStatus status = ReadStatus::RetryNeeded;
        return try_read(env, copy, READ_SPIN_LIMIT, status);
    }

    Napi::Value read_wait_internal(const Napi::CallbackInfo &info, bool copy)
    {
        Napi::Env env = info.Env();

        auto pending = std::make_shared<PendingRead>(env, copy);
        ReadStatus status = ReadStatus::RetryNeeded;
        Napi::Value value = try_read(env, copy, READ_SPIN_LIMIT, status);
        if (status == ReadStatus::Ok)
        {
            pending->deferred.Resolve(value);
            return pending->deferred.Promise();
        }
        if (status == ReadStatus::Destroyed)
        {
            pending->deferred.Reject(Napi::Error::New(env, "Destroyed").Value());
            return pending->deferred.Promise();
        }

        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            pending_reads_.push_back(pending);
            sync_async_ref_with_pending(true);
        }

        return pending->deferred.Promise();
    }

    Napi::Value Read(const Napi::CallbackInfo &info) { return read_internal(info, false); }
    Napi::Value ReadCopy(const Napi::CallbackInfo &info) { return read_internal(info, true); }
    Napi::Value ReadWait(const Napi::CallbackInfo &info) { return read_wait_internal(info, false); }
    Napi::Value ReadCopyWait(const Napi::CallbackInfo &info) { return read_wait_internal(info, true); }

    Napi::Value ByteCapacity(const Napi::CallbackInfo &info)
    {
        if (!mapped_)
            return Napi::Number::New(info.Env(), 0);
        return Napi::Number::New(info.Env(), static_cast<double>(max_bytes_));
    }
};

Napi::Object InitAll(Napi::Env env, Napi::Object exports)
{
    return SharedTensor::Init(env, exports);
}

NODE_API_MODULE(jude_map, InitAll)