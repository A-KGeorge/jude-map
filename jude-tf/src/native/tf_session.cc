#include <napi.h>
#include <uv.h>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "platform_tf.h"
#include "proto_parser.h"

// ---------------------------------------------------------------------------
// RAII wrappers for TF C objects
// ---------------------------------------------------------------------------
namespace
{
    struct StatusGuard
    {
        TF_Status *s;
        StatusGuard() : s(TF_NewStatus()) {}
        ~StatusGuard()
        {
            if (s)
                TF_DeleteStatus(s);
        }
        bool ok() const { return TF_GetCode(s) == TF_OK; }
        std::string message() const { return TF_Message(s); }
    };

    void noop_deallocator(void *, size_t, void *) {}
} // namespace

// ---------------------------------------------------------------------------
// TFSession — N-API ObjectWrap
// ---------------------------------------------------------------------------
class TFSession : public Napi::ObjectWrap<TFSession>
{
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports)
    {
        Napi::Function func = DefineClass(env, "TFSession", {
                                                                StaticMethod<&TFSession::LoadSavedModel>("loadSavedModel"),
                                                                StaticMethod<&TFSession::LoadFrozenGraph>("loadFrozenGraph"),
                                                                InstanceMethod<&TFSession::Run>("run"),
                                                                InstanceMethod<&TFSession::RunAsync>("runAsync"),
                                                                InstanceMethod<&TFSession::Destroy>("destroy"),
                                                                InstanceAccessor<&TFSession::Signatures>("signatures"),
                                                                InstanceAccessor<&TFSession::InferredInputs>("inputs"),
                                                                InstanceAccessor<&TFSession::InferredOutputs>("outputs"),
                                                            });
        auto *ctor = new Napi::FunctionReference(Napi::Persistent(func));
        env.SetInstanceData<Napi::FunctionReference>(ctor);
        exports.Set("TFSession", func);
        return exports;
    }

    TFSession(const Napi::CallbackInfo &info)
        : Napi::ObjectWrap<TFSession>(info) {}

    ~TFSession() { cleanup(); }

private:
    TF_Graph *graph_ = nullptr;
    TF_Session *session_ = nullptr;
    jude_tf::SignatureMap signatures_;
    std::vector<std::string> inferred_inputs_;
    std::vector<std::string> inferred_outputs_;
    bool is_frozen_ = false;

    void cleanup()
    {
        if (session_)
        {
            StatusGuard s;
            TF_CloseSession(session_, s.s);
            TF_DeleteSession(session_, s.s);
            session_ = nullptr;
        }
        if (graph_)
        {
            TF_DeleteGraph(graph_);
            graph_ = nullptr;
        }
    }

    // -----------------------------------------------------------------------
    // build_run_args — shared input/output resolution (event loop thread only)
    //
    // Populates tf_inputs, tf_input_tensors, tf_outputs, output_keys.
    // segment_refs accumulates ObjectReferences to any SharedTensorSegment
    // objects so their mmap stays valid across the async gap in runAsync.
    // Returns false and sets error_out on any validation failure.
    // -----------------------------------------------------------------------
    bool build_run_args(
        Napi::Env env,
        Napi::Object input_obj,
        const Napi::Value &output_keys_val,
        std::vector<TF_Output> &tf_inputs,
        std::vector<TF_Tensor *> &tf_input_tensors,
        std::vector<TF_Output> &tf_outputs,
        std::vector<std::string> &output_keys,
        std::vector<Napi::ObjectReference> &segment_refs,
        std::string &error_out)
    {
        const jude_tf::SignatureDef *sig = nullptr;
        if (!signatures_.empty())
            sig = jude_tf::pick_signature(signatures_);

        // ── inputs ──────────────────────────────────────────────────────────
        Napi::Array input_key_arr = input_obj.GetPropertyNames();
        for (uint32_t i = 0; i < input_key_arr.Length(); ++i)
        {
            std::string key = input_key_arr.Get(i).As<Napi::String>().Utf8Value();

            std::string op_name;
            int op_index = 0;
            if (sig)
            {
                auto it = sig->inputs.find(key);
                if (it == sig->inputs.end())
                {
                    error_out = "Unknown input key: " + key;
                    return false;
                }
                op_name = it->second.name;
                auto col = op_name.rfind(':');
                if (col != std::string::npos)
                {
                    op_index = std::stoi(op_name.substr(col + 1));
                    op_name = op_name.substr(0, col);
                }
            }
            else
            {
                op_name = key;
            }

            TF_Operation *op = TF_GraphOperationByName(graph_, op_name.c_str());
            if (!op)
            {
                error_out = "Graph op not found: " + op_name;
                return false;
            }
            tf_inputs.push_back({op, op_index});

            Napi::Value val = input_obj.Get(key);
            TF_Tensor *tensor = make_tensor(env, val, sig, key, segment_refs);
            if (!tensor)
            {
                error_out = "Failed to create tensor for input: " + key;
                return false;
            }
            tf_input_tensors.push_back(tensor);
        }

        // ── outputs ─────────────────────────────────────────────────────────
        bool user_specified = output_keys_val.IsArray();

        if (sig)
        {
            if (user_specified)
            {
                auto arr = output_keys_val.As<Napi::Array>();
                for (uint32_t i = 0; i < arr.Length(); ++i)
                    output_keys.push_back(arr.Get(i).As<Napi::String>().Utf8Value());
            }
            else
            {
                for (auto &[k, _] : sig->outputs)
                    output_keys.push_back(k);
            }

            for (auto &key : output_keys)
            {
                auto it = sig->outputs.find(key);
                if (it == sig->outputs.end())
                {
                    error_out = "Unknown output key: " + key;
                    return false;
                }
                std::string op_name = it->second.name;
                int op_idx = 0;
                auto col = op_name.rfind(':');
                if (col != std::string::npos)
                {
                    op_idx = std::stoi(op_name.substr(col + 1));
                    op_name = op_name.substr(0, col);
                }
                TF_Operation *op = TF_GraphOperationByName(graph_, op_name.c_str());
                if (!op)
                {
                    error_out = "Output op not found: " + op_name;
                    return false;
                }
                tf_outputs.push_back({op, op_idx});
            }
        }
        else
        {
            if (user_specified)
            {
                auto arr = output_keys_val.As<Napi::Array>();
                for (uint32_t i = 0; i < arr.Length(); ++i)
                    output_keys.push_back(arr.Get(i).As<Napi::String>().Utf8Value());
            }
            else
            {
                output_keys = inferred_outputs_;
            }

            if (output_keys.empty())
            {
                error_out = "No output ops found. Pass output op names as the "
                            "second argument to run() / runAsync().";
                return false;
            }

            for (auto &key : output_keys)
            {
                std::string op_name = key;
                int op_idx = 0;
                auto col = op_name.rfind(':');
                if (col != std::string::npos)
                {
                    op_idx = std::stoi(op_name.substr(col + 1));
                    op_name = op_name.substr(0, col);
                }
                TF_Operation *op = TF_GraphOperationByName(graph_, op_name.c_str());
                if (!op)
                {
                    error_out = "Output op not found: " + op_name;
                    return false;
                }
                tf_outputs.push_back({op, op_idx});
            }
        }

        return true;
    }

    // -----------------------------------------------------------------------
    // loadSavedModel
    // -----------------------------------------------------------------------
    static Napi::Value LoadSavedModel(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsString())
        {
            Napi::TypeError::New(env, "loadSavedModel(dir: string, tags?: string[])")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        std::string dir = info[0].As<Napi::String>().Utf8Value();
        std::vector<std::string> tags = {"serve"};
        if (info.Length() >= 2 && info[1].IsArray())
        {
            auto arr = info[1].As<Napi::Array>();
            tags.clear();
            for (uint32_t i = 0; i < arr.Length(); ++i)
                tags.push_back(arr.Get(i).As<Napi::String>().Utf8Value());
        }

        auto deferred = Napi::Promise::Deferred::New(env);
        auto *ctor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = ctor->New({});
        TFSession *self = Unwrap(obj);

        self->graph_ = TF_NewGraph();
        TF_SessionOptions *opts = TF_NewSessionOptions();
        std::vector<const char *> tag_ptrs;
        for (auto &t : tags)
            tag_ptrs.push_back(t.c_str());

        StatusGuard status;
        self->session_ = TF_LoadSessionFromSavedModel(
            opts, nullptr,
            dir.c_str(),
            tag_ptrs.data(), static_cast<int>(tag_ptrs.size()),
            self->graph_, nullptr, status.s);
        TF_DeleteSessionOptions(opts);

        if (!status.ok())
        {
            self->cleanup();
            deferred.Reject(Napi::Error::New(env,
                                             "TF_LoadSessionFromSavedModel failed: " + status.message())
                                .Value());
            return deferred.Promise();
        }

        std::string pb_path = dir + "/saved_model.pb";
        std::ifstream pb_file(pb_path, std::ios::binary | std::ios::ate);
        if (pb_file.is_open())
        {
            size_t sz = static_cast<size_t>(pb_file.tellg());
            pb_file.seekg(0);
            std::vector<uint8_t> buf(sz);
            pb_file.read(reinterpret_cast<char *>(buf.data()), sz);
            jude_tf::parse_saved_model(buf.data(), sz, self->signatures_);
        }

        deferred.Resolve(obj);
        return deferred.Promise();
    }

    // -----------------------------------------------------------------------
    // loadFrozenGraph
    // -----------------------------------------------------------------------
    static Napi::Value LoadFrozenGraph(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsString())
        {
            Napi::TypeError::New(env, "loadFrozenGraph(path: string)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        std::string path = info[0].As<Napi::String>().Utf8Value();
        auto deferred = Napi::Promise::Deferred::New(env);
        auto *ctor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = ctor->New({});
        TFSession *self = Unwrap(obj);

        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.is_open())
        {
            deferred.Reject(Napi::Error::New(env,
                                             "Cannot open frozen graph: " + path)
                                .Value());
            return deferred.Promise();
        }
        size_t sz = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<char> raw(sz);
        f.read(raw.data(), sz);

        TF_Buffer *graph_buf = TF_NewBufferFromString(raw.data(), sz);
        self->graph_ = TF_NewGraph();

        TF_ImportGraphDefOptions *import_opts = TF_NewImportGraphDefOptions();
        TF_ImportGraphDefOptionsSetPrefix(import_opts, "");

        StatusGuard status;
        TF_GraphImportGraphDef(self->graph_, graph_buf, import_opts, status.s);
        TF_DeleteImportGraphDefOptions(import_opts);
        TF_DeleteBuffer(graph_buf);

        if (!status.ok())
        {
            self->cleanup();
            deferred.Reject(Napi::Error::New(env,
                                             "TF_GraphImportGraphDef failed: " + status.message())
                                .Value());
            return deferred.Promise();
        }

        TF_SessionOptions *sess_opts = TF_NewSessionOptions();
        StatusGuard sess_status;
        self->session_ = TF_NewSession(self->graph_, sess_opts, sess_status.s);
        TF_DeleteSessionOptions(sess_opts);

        if (!sess_status.ok())
        {
            self->cleanup();
            deferred.Reject(Napi::Error::New(env,
                                             "TF_NewSession failed: " + sess_status.message())
                                .Value());
            return deferred.Promise();
        }

        self->is_frozen_ = true;
        self->infer_frozen_io();

        deferred.Resolve(obj);
        return deferred.Promise();
    }

    // -----------------------------------------------------------------------
    // infer_frozen_io
    //
    // Two-pass graph walk:
    //   Pass 1 — collect all Placeholder op names (inputs), and build a set
    //            of every op name that appears as a source in some other op's
    //            input list (i.e. ops that are "consumed").
    //   Pass 2 — ops that are NOT consumed, NOT Placeholder, NOT Const, and
    //            NOT NoOp are graph sinks → inferred outputs.
    //
    // This handles the common frozen graph layout produced by
    // convert_variables_to_constants_v2 where the final op is an Identity
    // or StatefulPartitionedCall that has no consumers.
    // -----------------------------------------------------------------------
    void infer_frozen_io()
    {
        // Pass 1: gather all ops and mark consumed ops.
        std::vector<TF_Operation *> all_ops;
        std::unordered_set<std::string> consumed;

        // Consumer ops that should not disqualify a producer from being treated
        // as a user-visible graph output.
        static const std::unordered_set<std::string> non_semantic_consumers = {
            "NoOp",
            "Assert",
            "_Retval",
            "IdentityN",
        };
        static const std::unordered_set<std::string> skip_types = {
            "Placeholder",
            "Const",
            "NoOp",
            "Assert",
            "_Arg",
            "_Retval",
            "VarHandleOp",
            "ReadVariableOp",
            "AssignVariableOp",
        };

        size_t pos = 0;
        TF_Operation *op;
        while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr)
        {
            all_ops.push_back(op);
            const char *type = TF_OperationOpType(op);

            if (strcmp(type, "Placeholder") == 0)
                inferred_inputs_.emplace_back(TF_OperationName(op));

            // Ignore infra consumers so terminal compute ops (often Identity)
            // remain discoverable as outputs for frozen graphs.
            if (non_semantic_consumers.count(type))
                continue;

            int n = TF_OperationNumInputs(op);
            for (int i = 0; i < n; ++i)
            {
                TF_Output src = TF_OperationInput({op, i});
                if (src.oper)
                    consumed.emplace(TF_OperationName(src.oper));
            }
        }

        for (auto *o : all_ops)
        {
            const char *type = TF_OperationOpType(o);
            const char *name = TF_OperationName(o);
            if (skip_types.count(type))
                continue;
            if (consumed.count(name))
                continue;

            // Ops whose name starts with ^ are control edges — skip.
            if (name[0] == '^')
                continue;
            inferred_outputs_.emplace_back(name);
        }
    }

    // -----------------------------------------------------------------------
    // run — synchronous (blocks event loop during TF_SessionRun)
    // -----------------------------------------------------------------------
    Napi::Value Run(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        auto deferred = Napi::Promise::Deferred::New(env);

        if (!session_)
        {
            deferred.Reject(Napi::Error::New(env, "Session destroyed").Value());
            return deferred.Promise();
        }
        if (info.Length() < 1 || !info[0].IsObject())
        {
            deferred.Reject(Napi::TypeError::New(env,
                                                 "run(inputs: Record<string, ...>)")
                                .Value());
            return deferred.Promise();
        }

        std::vector<TF_Output> tf_inputs, tf_outputs;
        std::vector<TF_Tensor *> tf_input_tensors;
        std::vector<std::string> output_keys;
        std::vector<Napi::ObjectReference> segment_refs;
        std::string error;

        Napi::Value out_val = (info.Length() >= 2) ? info[1] : env.Undefined();
        if (!build_run_args(env, info[0].As<Napi::Object>(), out_val,
                            tf_inputs, tf_input_tensors, tf_outputs, output_keys,
                            segment_refs, error))
        {
            deferred.Reject(Napi::Error::New(env, error).Value());
            return deferred.Promise();
        }

        std::vector<TF_Tensor *> output_tensors(tf_outputs.size(), nullptr);
        StatusGuard status;
        TF_SessionRun(
            session_, nullptr,
            tf_inputs.data(), tf_input_tensors.data(),
            static_cast<int>(tf_inputs.size()),
            tf_outputs.data(), output_tensors.data(),
            static_cast<int>(tf_outputs.size()),
            nullptr, 0, nullptr, status.s);

        for (auto *t : tf_input_tensors)
            TF_DeleteTensor(t);

        if (!status.ok())
        {
            for (auto *t : output_tensors)
                if (t)
                    TF_DeleteTensor(t);
            deferred.Reject(Napi::Error::New(env,
                                             "TF_SessionRun failed: " + status.message())
                                .Value());
            return deferred.Promise();
        }

        Napi::Object result = Napi::Object::New(env);
        for (size_t i = 0; i < output_tensors.size(); ++i)
        {
            if (!output_tensors[i])
                continue;
            result.Set(output_keys[i], tensor_to_js(env, output_tensors[i]));
            TF_DeleteTensor(output_tensors[i]);
        }

        deferred.Resolve(result);
        return deferred.Promise();
    }

    // -----------------------------------------------------------------------
    // runAsync — TF_SessionRun on libuv thread pool
    //
    // Phase 1 (event loop thread):
    //   build_run_args — all V8 access. Resolves op names, allocates input
    //   TF_Tensors, extracts zero-copy pointers, anchors segment lifetimes.
    //   After this, ctx contains only raw C pointers and TF objects — safe
    //   to hand to the thread pool.
    //
    // Phase 2 (libuv thread pool — OnRunWork):
    //   TF_SessionRun — pure C, no V8. May run concurrently with JS.
    //
    // Phase 3 (event loop thread — OnRunAfter):
    //   tensor_to_js, resolve/reject Promise. Release all resources.
    // -----------------------------------------------------------------------
    struct RunWorkCtx
    {
        uv_work_t req; // must be first — libuv casts req* → RunWorkCtx*

        TF_Session *session; // raw pointer, kept alive by self_ref

        // Pre-built args (event loop thread → thread pool)
        std::vector<TF_Output> tf_inputs;
        std::vector<TF_Tensor *> tf_input_tensors; // owned, deleted in OnRunAfter
        std::vector<TF_Output> tf_outputs;
        std::vector<std::string> output_keys;

        // Filled by OnRunWork
        std::vector<TF_Tensor *> output_tensors;

        // Zero-copy lifetime anchors.
        // Each ref keeps a SharedTensorSegment JS object (and its mmap) alive
        // across the async gap. Released in OnRunAfter after TF_SessionRun
        // returns — at that point TF has finished reading the buffer.
        std::vector<Napi::ObjectReference> segment_refs;

        bool ok = true;
        std::string error_message;

        Napi::Promise::Deferred deferred;
        Napi::ObjectReference self_ref; // keeps TFSession JS object alive
        TFSession *owner;               // for tensor_to_js in OnRunAfter

        explicit RunWorkCtx(Napi::Env env)
            : req{}, session(nullptr),
              deferred(Napi::Promise::Deferred::New(env)),
              owner(nullptr)
        {
        }
    };

    static void OnRunWork(uv_work_t *req)
    {
        // Thread pool — NO V8 access.
        auto *ctx = reinterpret_cast<RunWorkCtx *>(req);
        ctx->output_tensors.assign(ctx->tf_outputs.size(), nullptr);

        StatusGuard status;
        TF_SessionRun(
            ctx->session, nullptr,
            ctx->tf_inputs.data(), ctx->tf_input_tensors.data(),
            static_cast<int>(ctx->tf_inputs.size()),
            ctx->tf_outputs.data(), ctx->output_tensors.data(),
            static_cast<int>(ctx->tf_outputs.size()),
            nullptr, 0, nullptr, status.s);

        if (!status.ok())
        {
            ctx->ok = false;
            ctx->error_message = status.message();
        }
    }

    static void OnRunAfter(uv_work_t *req, int /*status*/)
    {
        // Event loop thread — V8 access safe.
        auto *ctx = reinterpret_cast<RunWorkCtx *>(req);
        Napi::Env env = ctx->deferred.Env();
        Napi::HandleScope scope(env);

        // Free input tensors regardless of outcome.
        for (auto *t : ctx->tf_input_tensors)
            TF_DeleteTensor(t);

        // segment_refs destructors run here — mmap stays valid until this point,
        // which is after TF_SessionRun has returned in OnRunWork.
        // self_ref destructor keeps TFSession alive through Unref below.

        if (!ctx->ok)
        {
            for (auto *t : ctx->output_tensors)
                if (t)
                    TF_DeleteTensor(t);
            ctx->deferred.Reject(Napi::Error::New(env,
                                                  "TF_SessionRun failed: " + ctx->error_message)
                                     .Value());
        }
        else
        {
            Napi::Object result = Napi::Object::New(env);
            for (size_t i = 0; i < ctx->output_tensors.size(); ++i)
            {
                TF_Tensor *t = ctx->output_tensors[i];
                if (!t)
                    continue;
                result.Set(ctx->output_keys[i], ctx->owner->tensor_to_js(env, t));
                TF_DeleteTensor(t);
            }
            ctx->deferred.Resolve(result);
        }

        ctx->self_ref.Unref();
        delete ctx;
    }

    Napi::Value RunAsync(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();

        if (!session_)
        {
            auto d = Napi::Promise::Deferred::New(env);
            d.Reject(Napi::Error::New(env, "Session destroyed").Value());
            return d.Promise();
        }
        if (info.Length() < 1 || !info[0].IsObject())
        {
            auto d = Napi::Promise::Deferred::New(env);
            d.Reject(Napi::TypeError::New(env,
                                          "runAsync(inputs: Record<string, ...>)")
                         .Value());
            return d.Promise();
        }

        auto *ctx = new RunWorkCtx(env);
        ctx->session = session_;
        ctx->owner = this;
        ctx->self_ref = Napi::ObjectReference::New(
            info.This().As<Napi::Object>(), 1);

        // Phase 1: all V8 work on event loop thread.
        Napi::Value out_val = (info.Length() >= 2) ? info[1] : env.Undefined();
        std::string error;
        if (!build_run_args(
                env, info[0].As<Napi::Object>(), out_val,
                ctx->tf_inputs, ctx->tf_input_tensors,
                ctx->tf_outputs, ctx->output_keys,
                ctx->segment_refs, error))
        {
            auto promise = ctx->deferred.Promise();
            ctx->deferred.Reject(Napi::Error::New(env, error).Value());
            ctx->self_ref.Unref();
            delete ctx;
            return promise;
        }

        uv_loop_t *loop = nullptr;
        if (napi_get_uv_event_loop(env, &loop) != napi_ok || !loop ||
            uv_queue_work(loop, &ctx->req, OnRunWork, OnRunAfter) != 0)
        {
            // Fallback: synchronous if queue fails.
            for (auto *t : ctx->tf_input_tensors)
                TF_DeleteTensor(t);
            auto promise = ctx->deferred.Promise();
            ctx->deferred.Reject(Napi::Error::New(env,
                                                  "Failed to queue runAsync work")
                                     .Value());
            ctx->self_ref.Unref();
            delete ctx;
            return promise;
        }

        return ctx->deferred.Promise();
    }

    // -----------------------------------------------------------------------
    // make_tensor
    // -----------------------------------------------------------------------
    TF_Tensor *make_tensor(
        Napi::Env env,
        Napi::Value val,
        const jude_tf::SignatureDef *sig,
        const std::string &key,
        std::vector<Napi::ObjectReference> &segment_refs)
    {
        if (val.IsTypedArray())
        {
            Napi::TypedArray ta = val.As<Napi::TypedArray>();
            TF_DataType dtype = js_typed_array_dtype(ta.TypedArrayType());

            std::vector<int64_t> dims;
            if (sig)
            {
                auto it = sig->inputs.find(key);
                if (it != sig->inputs.end())
                    dims = it->second.shape.dims;
            }
            else
            {
                TF_Operation *op = TF_GraphOperationByName(graph_, key.c_str());
                if (op)
                {
                    TF_Output out{op, 0};
                    StatusGuard sg;
                    int n = TF_GraphGetTensorNumDims(graph_, out, sg.s);
                    if (sg.ok() && n > 0)
                    {
                        dims.resize(static_cast<size_t>(n), -1);
                        StatusGuard sg2;
                        TF_GraphGetTensorShape(graph_, out, dims.data(), n, sg2.s);
                        if (!sg2.ok())
                            dims.clear();
                    }
                }
            }

            if (dims.empty())
            {
                dims.push_back(static_cast<int64_t>(ta.ElementLength()));
            }
            else
            {
                int64_t total = static_cast<int64_t>(ta.ElementLength());
                int unk_count = 0, unk_idx = -1;
                int64_t known = 1;
                for (int i = 0; i < static_cast<int>(dims.size()); ++i)
                {
                    if (dims[i] < 1)
                    {
                        unk_count++;
                        unk_idx = i;
                    }
                    else
                    {
                        known *= dims[i];
                    }
                }
                if (unk_count == 1 && known > 0 && total % known == 0)
                    dims[unk_idx] = total / known;
                else
                    for (auto &d : dims)
                        if (d < 1)
                            d = 1;
            }

            TF_Tensor *t = TF_AllocateTensor(
                dtype, dims.data(), static_cast<int>(dims.size()),
                ta.ByteLength());
            if (!t)
                return nullptr;
            std::memcpy(TF_TensorData(t),
                        reinterpret_cast<const uint8_t *>(ta.ArrayBuffer().Data()) + ta.ByteOffset(),
                        ta.ByteLength());
            return t;
        }

        if (val.IsObject())
        {
            Napi::Object obj = val.As<Napi::Object>();
            if (obj.Has("read") && obj.Get("read").IsFunction())
            {
                Napi::Value res = obj.Get("read").As<Napi::Function>().Call(obj, {});
                if (res.IsNull() || res.IsUndefined())
                    return nullptr;

                Napi::Object r = res.As<Napi::Object>();
                Napi::Array shape_arr = r.Get("shape").As<Napi::Array>();
                int dtype_id = r.Get("dtype").As<Napi::Number>().Int32Value();

                void *data_ptr = nullptr;
                size_t data_len = 0;

                if (r.Has("buffer") && r.Get("buffer").IsArrayBuffer())
                {
                    auto ab = r.Get("buffer").As<Napi::ArrayBuffer>();
                    data_ptr = ab.Data();
                    data_len = ab.ByteLength();
                }
                else if (r.Has("data") && r.Get("data").IsTypedArray())
                {
                    auto ta2 = r.Get("data").As<Napi::TypedArray>();
                    data_ptr = reinterpret_cast<uint8_t *>(
                                   ta2.ArrayBuffer().Data()) +
                               ta2.ByteOffset();
                    data_len = ta2.ByteLength();
                }
                else
                {
                    return nullptr;
                }

                std::vector<int64_t> dims;
                dims.reserve(shape_arr.Length());
                for (uint32_t i = 0; i < shape_arr.Length(); ++i)
                    dims.push_back(static_cast<int64_t>(
                        shape_arr.Get(i).As<Napi::Number>().Int64Value()));

                TF_DataType tf_dtype = static_cast<TF_DataType>(
                    jude_dtype_to_tf(dtype_id));

                // Anchor the segment so the mmap survives the async gap.
                segment_refs.push_back(Napi::ObjectReference::New(obj, 1));

                return TF_NewTensor(
                    tf_dtype, dims.data(), static_cast<int>(dims.size()),
                    data_ptr, data_len, noop_deallocator, nullptr);
            }
        }
        return nullptr;
    }

    // -----------------------------------------------------------------------
    // Output helpers
    // -----------------------------------------------------------------------
    Napi::Object tensor_to_js(Napi::Env env, TF_Tensor *t)
    {
        Napi::Object obj = Napi::Object::New(env);
        TF_DataType dtype = TF_TensorType(t);
        int ndims = TF_NumDims(t);
        size_t nb = TF_TensorByteSize(t);

        Napi::Array shape = Napi::Array::New(env, ndims);
        for (int i = 0; i < ndims; ++i)
            shape.Set(i, Napi::Number::New(env,
                                           static_cast<double>(TF_Dim(t, i))));

        obj.Set("dtype", Napi::Number::New(env, static_cast<double>(dtype)));
        obj.Set("shape", shape);

        Napi::ArrayBuffer buf = Napi::ArrayBuffer::New(env, nb);
        std::memcpy(buf.Data(), TF_TensorData(t), nb);
        obj.Set("data", tf_dtype_to_typed_array(env, dtype, buf, nb));
        return obj;
    }

    Napi::Value tf_dtype_to_typed_array(
        Napi::Env env, TF_DataType dtype, Napi::ArrayBuffer buf, size_t nb)
    {
        switch (dtype)
        {
        case TF_FLOAT:
            return Napi::Float32Array::New(env, nb / 4, buf, 0);
        case TF_DOUBLE:
            return Napi::Float64Array::New(env, nb / 8, buf, 0);
        case TF_INT32:
            return Napi::Int32Array::New(env, nb / 4, buf, 0);
        case TF_UINT8:
            return Napi::Uint8Array::New(env, nb / 1, buf, 0);
        case TF_INT8:
            return Napi::Int8Array::New(env, nb / 1, buf, 0);
        case TF_UINT16:
            return Napi::Uint16Array::New(env, nb / 2, buf, 0);
        case TF_INT16:
            return Napi::Int16Array::New(env, nb / 2, buf, 0);
        default:
            return buf;
        }
    }

    static int jude_dtype_to_tf(int id)
    {
        static const int map[] = {
            TF_FLOAT,
            TF_DOUBLE,
            TF_INT32,
            TF_INT64,
            TF_UINT8,
            TF_INT8,
            TF_UINT16,
            TF_INT16,
            TF_BOOL,
        };
        return (id >= 0 && id < 9) ? map[id] : TF_FLOAT;
    }

    static TF_DataType js_typed_array_dtype(napi_typedarray_type t)
    {
        switch (t)
        {
        case napi_float32_array:
            return TF_FLOAT;
        case napi_float64_array:
            return TF_DOUBLE;
        case napi_int32_array:
            return TF_INT32;
        case napi_uint8_array:
            return TF_UINT8;
        case napi_int8_array:
            return TF_INT8;
        case napi_uint16_array:
            return TF_UINT16;
        case napi_int16_array:
            return TF_INT16;
        default:
            return TF_FLOAT;
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    Napi::Value Signatures(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        Napi::Object out = Napi::Object::New(env);
        for (auto &[sig_key, sig] : signatures_)
        {
            Napi::Object sig_obj = Napi::Object::New(env);
            auto make_map = [&](
                                const std::unordered_map<std::string, jude_tf::TensorInfo> &m)
            {
                Napi::Object mo = Napi::Object::New(env);
                for (auto &[k, ti] : m)
                {
                    Napi::Object io = Napi::Object::New(env);
                    io.Set("name", Napi::String::New(env, ti.name));
                    io.Set("dtype", Napi::Number::New(env, ti.dtype));
                    Napi::Array dims = Napi::Array::New(env, ti.shape.dims.size());
                    for (size_t i = 0; i < ti.shape.dims.size(); ++i)
                        dims.Set(i, Napi::Number::New(env,
                                                      static_cast<double>(ti.shape.dims[i])));
                    io.Set("shape", dims);
                    mo.Set(k, io);
                }
                return mo;
            };
            sig_obj.Set("inputs", make_map(sig.inputs));
            sig_obj.Set("outputs", make_map(sig.outputs));
            sig_obj.Set("methodName", Napi::String::New(env, sig.method_name));
            out.Set(sig_key, sig_obj);
        }
        return out;
    }

    Napi::Value InferredInputs(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        Napi::Array arr = Napi::Array::New(env, inferred_inputs_.size());
        for (size_t i = 0; i < inferred_inputs_.size(); ++i)
            arr.Set(i, Napi::String::New(env, inferred_inputs_[i]));
        return arr;
    }

    Napi::Value InferredOutputs(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        Napi::Array arr = Napi::Array::New(env, inferred_outputs_.size());
        for (size_t i = 0; i < inferred_outputs_.size(); ++i)
            arr.Set(i, Napi::String::New(env, inferred_outputs_[i]));
        return arr;
    }

    void Destroy(const Napi::CallbackInfo &) { cleanup(); }
};

// ---------------------------------------------------------------------------
Napi::Object InitAll(Napi::Env env, Napi::Object exports)
{
    return TFSession::Init(env, exports);
}
NODE_API_MODULE(jude_tf, InitAll)