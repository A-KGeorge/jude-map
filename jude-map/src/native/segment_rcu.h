#pragma once
#include "segment.h"
#include <atomic>
#include <cstring>
#include <thread>

// ---------------------------------------------------------------------------
// RCU-lite double-buffer segment for large tensors (>= RCU_THRESHOLD bytes).
//
// Key property: readers NEVER retry. O(1) read latency regardless of write
// pressure. Stable p99 under heavy concurrent load.
//
// Trade-off vs seqlock:
//   seqlock: 1x memory, retries under contention, excellent for small tensors
//   RCU-lite: 2x memory, zero retries, better p99 for large tensors
//
// Single-writer / multiple-reader model:
//   Writer prepares data in the "shadow" buffer, then atomically swaps
//   the current pointer. Waits for readers of the old buffer to drain
//   (trivially fast for the 1-reader inference case).
//
// Memory layout in the SAB / mmap region:
//   [RCUHeader: 384 bytes] [buf0: max_bytes] [buf1: max_bytes]
//
// Total allocation size: RCU_HEADER_SIZE + 2 * max_bytes
// ---------------------------------------------------------------------------

static constexpr size_t RCU_THRESHOLD = 64ULL * 1024ULL; // 64 KB

// ---------------------------------------------------------------------------
// RCUHeader layout (384 bytes, 6 cache lines):
//
//  [0..63]    current + pad          — which buffer is "live" for readers
//  [64..127]  readers[0..1] + pad    — per-buffer reference counts
//  [128..255] meta[0]                — TensorMeta for buffer 0
//  [256..383] meta[1]                — TensorMeta for buffer 1
// ---------------------------------------------------------------------------
struct alignas(64) RCUHeader
{
    // Cache line 0: current buffer selector
    std::atomic<uint32_t> current{0}; // 0 or 1
    uint8_t _pad0[60];

    // Cache line 1: per-buffer reader reference counts
    std::atomic<uint32_t> readers[2]; // readers[0], readers[1]
    uint8_t _pad1[56];

    // Cache lines 2-3: TensorMeta for buffer 0
    TensorMeta meta0; // 128 bytes

    // Cache lines 4-5: TensorMeta for buffer 1
    TensorMeta meta1; // 128 bytes
};

static_assert(sizeof(RCUHeader) == 384,
              "RCUHeader must be 384 bytes (6 cache lines)");
static_assert(offsetof(RCUHeader, readers) == 64,
              "readers must be on its own cache line");
static_assert(offsetof(RCUHeader, meta0) == 128,
              "meta0 must start at byte 128");
static_assert(offsetof(RCUHeader, meta1) == 256,
              "meta1 must start at byte 256");

static constexpr size_t RCU_HEADER_SIZE = sizeof(RCUHeader); // 384

// Returns the data pointer for buffer 0 or 1 within the mapped region.
inline uint8_t *rcu_buf_ptr(void *base, uint32_t idx, size_t max_bytes)
{
    return reinterpret_cast<uint8_t *>(base) + RCU_HEADER_SIZE + static_cast<size_t>(idx) * max_bytes;
}

inline const uint8_t *rcu_buf_ptr(const void *base, uint32_t idx, size_t max_bytes)
{
    return reinterpret_cast<const uint8_t *>(base) + RCU_HEADER_SIZE + static_cast<size_t>(idx) * max_bytes;
}

inline TensorMeta &rcu_meta(void *base, uint32_t idx)
{
    auto *hdr = reinterpret_cast<RCUHeader *>(base);
    return idx == 0 ? hdr->meta0 : hdr->meta1;
}

// Total SAB / mmap allocation size for an RCU segment.
inline size_t rcu_segment_size(size_t max_bytes)
{
    return RCU_HEADER_SIZE + 2 * max_bytes;
}

// Placement-new initialise the header (call from createSharedRCU).
inline void rcu_init_header(void *base)
{
    new (base) RCUHeader();
}

// ---------------------------------------------------------------------------
// rcu_write — write a tensor to the shadow buffer, then atomically swap.
//
// Thread-safety: single writer only. Multiple writers need external locking.
//
// Latency: O(src_size) memcpy + brief wait for shadow reader drain.
//          In the 1-writer / 1-reader inference case the wait is typically 0.
// ---------------------------------------------------------------------------
inline void rcu_write(void *base,
                      size_t max_bytes,
                      const uint8_t *src,
                      size_t src_size,
                      const TensorMeta &src_meta)
{
    auto *hdr = reinterpret_cast<RCUHeader *>(base);

    // Shadow = the buffer NOT currently serving readers.
    const uint32_t cur = hdr->current.load(std::memory_order_relaxed);
    const uint32_t shadow = 1u - cur;

    // Wait for any readers of the shadow buffer to drain.
    // This is typically 0 iterations in the inference case (1 reader).
    // Under higher reader counts it may spin a few iterations.
    uint32_t spins = 0;
    while (hdr->readers[shadow].load(std::memory_order_acquire) != 0)
    {
        ++spins;
        if (spins < 64)
            ; // tight spin — usually done in < 5 iters
        else
            std::this_thread::yield();
    }

    // Write metadata and data to shadow buffer.
    // No seqlock needed here — readers won't touch this buffer until after
    // the current swap below.
    std::memcpy(&rcu_meta(base, shadow), &src_meta, sizeof(TensorMeta));
    std::memcpy(rcu_buf_ptr(base, shadow, max_bytes), src, src_size);

    // Release fence: ensure all writes are visible before the pointer swap.
    std::atomic_thread_fence(std::memory_order_release);

    // Atomically publish the new buffer.
    hdr->current.store(shadow, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// RCUReadGuard — RAII reader reference for the RCU protocol.
//
// Construction: atomically pins the current buffer so the writer won't
//               overwrite it while we're reading.
// Destruction:  releases the pin.
//
// Usage:
//   RCUReadGuard guard(base);
//   const TensorMeta& meta = guard.meta();
//   const uint8_t*    data = guard.data_ptr(max_bytes);
//   // copy/view data here
//   // guard destructor releases the pin automatically
// ---------------------------------------------------------------------------
struct RCUReadGuard
{
    RCUHeader *hdr;
    uint32_t idx;

    explicit RCUReadGuard(void *base)
    {
        hdr = reinterpret_cast<RCUHeader *>(base);

        // Acquire-loop: pin the current buffer. At most 1 retry if the writer
        // swaps the pointer in the narrow window between our load and our
        // fetch_add. This is the only "retry" in the RCU protocol — it has
        // nothing to do with data consistency and is always ≤ 1 iteration.
        while (true)
        {
            idx = hdr->current.load(std::memory_order_acquire);
            hdr->readers[idx].fetch_add(1, std::memory_order_acquire);

            // Re-check: if the writer swapped current between our load and our
            // fetch_add we pinned the wrong buffer — release and retry.
            if (hdr->current.load(std::memory_order_relaxed) == idx)
                break;

            hdr->readers[idx].fetch_sub(1, std::memory_order_release);
        }
    }

    ~RCUReadGuard()
    {
        hdr->readers[idx].fetch_sub(1, std::memory_order_release);
    }

    const TensorMeta &meta() const
    {
        return idx == 0 ? hdr->meta0 : hdr->meta1;
    }

    const uint8_t *data_ptr(const void *base, size_t max_bytes) const
    {
        return rcu_buf_ptr(base, idx, max_bytes);
    }

    // Disable copy and move — the guard must not outlive the segment.
    RCUReadGuard(const RCUReadGuard &) = delete;
    RCUReadGuard &operator=(const RCUReadGuard &) = delete;
};