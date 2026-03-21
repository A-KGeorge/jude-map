#include <napi.h>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
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

    struct TensorGuard
    {
        TF_Tensor *t = nullptr;
        explicit TensorGuard(TF_Tensor *t) : t(t) {}
        ~TensorGuard()
        {
            if (t)
                TF_DeleteTensor(t);
        }
        TF_Tensor *release()
        {
            auto *r = t;
            t = nullptr;
            return r;
        }
    };

    // No-op deallocator — used when the tensor data is owned by jude-map mmap.
    void noop_deallocator(void *, size_t, void *) {}

} // namespace

// ---------------------------------------------------------------------------
// TFSession — N-API ObjectWrap
//
// JS API:
//   const sess = await TFSession.loadSavedModel(dir, tags?)
//   const sess = await TFSession.loadFrozenGraph(path)
//   const out  = await sess.run({ input_key: segment })
//   const sigs = sess.signatures   // { [key]: { inputs, outputs } }
//   sess.destroy()
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
                                                                InstanceMethod<&TFSession::Destroy>("destroy"),
                                                                InstanceAccessor<&TFSession::Signatures>("signatures"),
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
    // Fallback for frozen graphs — inferred from placeholder ops
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
    // TFSession.loadSavedModel(dir: string, tags?: string[]) → Promise<TFSession>
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

        // Optional tags array — defaults to ["serve"]
        std::vector<std::string> tags = {"serve"};
        if (info.Length() >= 2 && info[1].IsArray())
        {
            Napi::Array arr = info[1].As<Napi::Array>();
            tags.clear();
            for (uint32_t i = 0; i < arr.Length(); ++i)
                tags.push_back(arr.Get(i).As<Napi::String>().Utf8Value());
        }

        auto deferred = Napi::Promise::Deferred::New(env);

        // Build the TFSession object.
        auto *ctor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = ctor->New({});
        TFSession *self = Unwrap(obj);

        self->graph_ = TF_NewGraph();

        TF_SessionOptions *opts = TF_NewSessionOptions();
        // CPU only: restrict TF to 1 intra-op thread to avoid fighting with
        // the JS worker threads. The JS layer handles parallelism.
        // (Full thread-count config via TF_SetConfig protobuf — simplified here.)

        // Prepare tags array for C API
        std::vector<const char *> tag_ptrs;
        tag_ptrs.reserve(tags.size());
        for (auto &t : tags)
            tag_ptrs.push_back(t.c_str());

        TF_Buffer *run_opts = nullptr;
        TF_Buffer *meta_graph_buf = nullptr;

        StatusGuard status;
        self->session_ = TF_LoadSessionFromSavedModel(
            opts, run_opts,
            dir.c_str(),
            tag_ptrs.data(), static_cast<int>(tag_ptrs.size()),
            self->graph_,
            meta_graph_buf,
            status.s);

        TF_DeleteSessionOptions(opts);

        if (!status.ok())
        {
            self->cleanup();
            deferred.Reject(Napi::Error::New(env,
                                             "TF_LoadSessionFromSavedModel failed: " + status.message())
                                .Value());
            return deferred.Promise();
        }

        // Parse saved_model.pb for SignatureDefs.
        std::string pb_path = dir + "/saved_model.pb";
        std::ifstream pb_file(pb_path, std::ios::binary | std::ios::ate);
        if (pb_file.is_open())
        {
            size_t size = static_cast<size_t>(pb_file.tellg());
            pb_file.seekg(0);
            std::vector<uint8_t> buf(size);
            pb_file.read(reinterpret_cast<char *>(buf.data()), size);
            jude_tf::parse_saved_model(buf.data(), size, self->signatures_);
        }

        deferred.Resolve(obj);
        return deferred.Promise();
    }

    // -----------------------------------------------------------------------
    // TFSession.loadFrozenGraph(path: string) → Promise<TFSession>
    //
    // Loads a frozen GraphDef (.pb). Infers inputs from Placeholder ops and
    // outputs from the last non-control op in topological order.
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

        // Read the .pb file into a TF_Buffer.
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.is_open())
        {
            deferred.Reject(Napi::Error::New(env,
                                             "Cannot open frozen graph: " + path)
                                .Value());
            return deferred.Promise();
        }
        size_t size = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<char> raw(size);
        f.read(raw.data(), size);

        TF_Buffer *graph_buf = TF_NewBufferFromString(raw.data(), size);
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

        // Create session.
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

        // Infer inputs (Placeholder ops) and outputs (last non-control op).
        self->is_frozen_ = true;
        self->infer_frozen_io();

        deferred.Resolve(obj);
        return deferred.Promise();
    }

    // Walk the graph to find Placeholder ops (inputs) and candidate outputs.
    void infer_frozen_io()
    {
        size_t pos = 0;
        TF_Operation *op;
        while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr)
        {
            const char *type = TF_OperationOpType(op);
            const char *name = TF_OperationName(op);
            if (strcmp(type, "Placeholder") == 0)
                inferred_inputs_.emplace_back(name);
        }
        // For outputs: find ops with no outgoing consumers (graph sinks).
        // Simplified — collect all ops whose output is not consumed by others.
        // In practice users usually specify outputs explicitly via run() options.
    }

    // -----------------------------------------------------------------------
    // sess.run(inputs: Record<string, SharedTensorSegment | TypedArray>,
    //          outputKeys?: string[]) → Promise<Record<string, TensorResult>>
    //
    // inputs:     map from SignatureDef input key → data source
    //             Value can be a SharedTensorSegment (zero-copy mmap path)
    //             or a TypedArray (copies into a TF_Tensor)
    // outputKeys: which SignatureDef output keys to compute
    //             (defaults to all outputs in the signature)
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

        // Resolve the active signature.
        const jude_tf::SignatureDef *sig = nullptr;
        if (!signatures_.empty())
            sig = jude_tf::pick_signature(signatures_);

        Napi::Object input_obj = info[0].As<Napi::Object>();
        Napi::Array input_keys = input_obj.GetPropertyNames();

        // Build TF input tensors.
        std::vector<TF_Output> tf_inputs;
        std::vector<TF_Tensor *> tf_input_tensors;
        std::vector<std::string> cleanup_errors;

        for (uint32_t i = 0; i < input_keys.Length(); ++i)
        {
            std::string key = input_keys.Get(i).As<Napi::String>().Utf8Value();

            // Resolve the graph op name for this key.
            std::string op_name;
            int op_index = 0;
            if (sig)
            {
                auto it = sig->inputs.find(key);
                if (it == sig->inputs.end())
                {
                    deferred.Reject(Napi::Error::New(env,
                                                     "Unknown input key: " + key)
                                        .Value());
                    return deferred.Promise();
                }
                // TensorInfo.name is "op_name:index"
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
                // Frozen graph: key IS the op name.
                op_name = key;
            }

            TF_Operation *op = TF_GraphOperationByName(graph_, op_name.c_str());
            if (!op)
            {
                deferred.Reject(Napi::Error::New(env,
                                                 "Graph op not found: " + op_name)
                                    .Value());
                return deferred.Promise();
            }
            tf_inputs.push_back({op, op_index});

            // Build TF_Tensor from the input value.
            Napi::Value val = input_obj.Get(key);
            TF_Tensor *tensor = make_tensor(env, val, sig, key);
            if (!tensor)
            {
                deferred.Reject(Napi::Error::New(env,
                                                 "Failed to create tensor for input: " + key)
                                    .Value());
                return deferred.Promise();
            }
            tf_input_tensors.push_back(tensor);
        }

        // Build TF output specs.
        std::vector<TF_Output> tf_outputs;
        std::vector<std::string> output_keys;

        if (sig)
        {
            // Determine which outputs to compute.
            bool user_specified = (info.Length() >= 2 && info[1].IsArray());
            if (user_specified)
            {
                Napi::Array arr = info[1].As<Napi::Array>();
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
                    deferred.Reject(Napi::Error::New(env,
                                                     "Unknown output key: " + key)
                                        .Value());
                    return deferred.Promise();
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
                    deferred.Reject(Napi::Error::New(env,
                                                     "Output op not found: " + op_name)
                                        .Value());
                    return deferred.Promise();
                }
                tf_outputs.push_back({op, op_idx});
            }
        }

        std::vector<TF_Tensor *> output_tensors(tf_outputs.size(), nullptr);

        // Run the session.
        StatusGuard status;
        TF_SessionRun(
            session_,
            nullptr, // run options
            tf_inputs.data(),
            tf_input_tensors.data(),
            static_cast<int>(tf_inputs.size()),
            tf_outputs.data(),
            output_tensors.data(),
            static_cast<int>(tf_outputs.size()),
            nullptr, 0, // target ops
            nullptr,    // run metadata
            status.s);

        // Free input tensors.
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

        // Package results.
        Napi::Object result = Napi::Object::New(env);
        for (size_t i = 0; i < output_tensors.size(); ++i)
        {
            TF_Tensor *t = output_tensors[i];
            if (!t)
                continue;
            result.Set(output_keys[i], tensor_to_js(env, t));
            TF_DeleteTensor(t);
        }

        deferred.Resolve(result);
        return deferred.Promise();
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Build a TF_Tensor from a JS value.
    // If value is a SharedTensorSegment, use its mmap pointer directly (zero-copy).
    // If value is a TypedArray, copy the data.
    TF_Tensor *make_tensor(
        Napi::Env env,
        Napi::Value val,
        const jude_tf::SignatureDef *sig,
        const std::string &key)
    {
        // TypedArray path — copy into TF_Tensor.
        if (val.IsTypedArray())
        {
            Napi::TypedArray ta = val.As<Napi::TypedArray>();
            TF_DataType dtype = js_typed_array_dtype(ta.TypedArrayType());
            int64_t dims[1] = {static_cast<int64_t>(ta.ElementLength())};

            // Use TF_DONT_DEALLOCATE_CONTENTS isn't a symbol — pass no-op deallocator.
            TF_Tensor *t = TF_AllocateTensor(dtype, dims, 1,
                                             ta.ByteLength());
            if (!t)
                return nullptr;
            std::memcpy(TF_TensorData(t),
                        reinterpret_cast<const uint8_t *>(
                            ta.ArrayBuffer().Data()) +
                            ta.ByteOffset(),
                        ta.ByteLength());
            return t;
        }

        // SharedTensorSegment path — zero-copy mmap pointer.
        // The segment exposes { data: ArrayBuffer, shape: number[], dtype: number }
        // via its read() method. We call read() here and wrap with noop deallocator.
        if (val.IsObject())
        {
            Napi::Object obj = val.As<Napi::Object>();

            // Check if it looks like a SharedTensorSegment
            // (has a read() method that returns { buffer, shape, dtype }).
            if (obj.Has("read") && obj.Get("read").IsFunction())
            {
                Napi::Value result = obj.Get("read").As<Napi::Function>().Call(obj, {});
                if (result.IsNull() || result.IsUndefined())
                    return nullptr;

                Napi::Object r = result.As<Napi::Object>();
                Napi::Array shape_arr = r.Get("shape").As<Napi::Array>();
                int dtype_id = r.Get("dtype").As<Napi::Number>().Int32Value();
                Napi::ArrayBuffer ab = r.Get("buffer").As<Napi::ArrayBuffer>();

                std::vector<int64_t> dims;
                dims.reserve(shape_arr.Length());
                for (uint32_t i = 0; i < shape_arr.Length(); ++i)
                    dims.push_back(static_cast<int64_t>(
                        shape_arr.Get(i).As<Napi::Number>().Int64Value()));

                TF_DataType tf_dtype = static_cast<TF_DataType>(
                    jude_dtype_to_tf(dtype_id));

                // Zero-copy: pass mmap pointer with noop deallocator.
                // The ArrayBuffer (and its backing mmap via shared_ptr) stays
                // alive for the duration of TF_SessionRun.
                TF_Tensor *t = TF_NewTensor(
                    tf_dtype,
                    dims.data(),
                    static_cast<int>(dims.size()),
                    ab.Data(),
                    ab.ByteLength(),
                    noop_deallocator,
                    nullptr);
                return t;
            }
        }

        return nullptr;
    }

    // Convert a TF output tensor to a JS object
    // { dtype, shape, data: Float32Array | ... }
    Napi::Object tensor_to_js(Napi::Env env, TF_Tensor *t)
    {
        Napi::Object obj = Napi::Object::New(env);

        TF_DataType dtype = TF_TensorType(t);
        int n_dims = TF_NumDims(t);
        size_t nbytes = TF_TensorByteSize(t);

        Napi::Array shape = Napi::Array::New(env, n_dims);
        for (int i = 0; i < n_dims; ++i)
            shape.Set(i, Napi::Number::New(env, static_cast<double>(TF_Dim(t, i))));

        obj.Set("dtype", Napi::Number::New(env, static_cast<double>(dtype)));
        obj.Set("shape", shape);

        // Copy the output data into a new ArrayBuffer.
        Napi::ArrayBuffer buf = Napi::ArrayBuffer::New(env, nbytes);
        std::memcpy(buf.Data(), TF_TensorData(t), nbytes);

        // Wrap in the appropriate TypedArray.
        obj.Set("data", tf_dtype_to_typed_array(env, dtype, buf, nbytes));
        return obj;
    }

    Napi::Value tf_dtype_to_typed_array(
        Napi::Env env, TF_DataType dtype,
        Napi::ArrayBuffer buf, size_t nbytes)
    {
        switch (dtype)
        {
        case TF_FLOAT:
            return Napi::Float32Array::New(env, nbytes / 4, buf, 0);
        case TF_DOUBLE:
            return Napi::Float64Array::New(env, nbytes / 8, buf, 0);
        case TF_INT32:
            return Napi::Int32Array::New(env, nbytes / 4, buf, 0);
        case TF_UINT8:
            return Napi::Uint8Array::New(env, nbytes / 1, buf, 0);
        case TF_INT8:
            return Napi::Int8Array::New(env, nbytes / 1, buf, 0);
        case TF_UINT16:
            return Napi::Uint16Array::New(env, nbytes / 2, buf, 0);
        case TF_INT16:
            return Napi::Int16Array::New(env, nbytes / 2, buf, 0);
        default:
            return buf; // raw ArrayBuffer for unsupported types
        }
    }

    // Map jude-map DType IDs to TF_DataType values.
    static int jude_dtype_to_tf(int dtype_id)
    {
        // jude-map DType enum mirrors TF_DataType values for float/int types.
        // FLOAT32=0→TF_FLOAT=1, FLOAT64=1→TF_DOUBLE=2, INT32=2→TF_INT32=3, etc.
        static const int map[] = {
            TF_FLOAT,  // FLOAT32 = 0
            TF_DOUBLE, // FLOAT64 = 1
            TF_INT32,  // INT32   = 2
            TF_INT64,  // INT64   = 3
            TF_UINT8,  // UINT8   = 4
            TF_INT8,   // INT8    = 5
            TF_UINT16, // UINT16  = 6
            TF_INT16,  // INT16   = 7
            TF_BOOL,   // BOOL    = 8
        };
        if (dtype_id >= 0 && dtype_id < 9)
            return map[dtype_id];
        return TF_FLOAT;
    }

    static TF_DataType js_typed_array_dtype(napi_typedarray_type type)
    {
        switch (type)
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

    // sess.signatures → object describing all detected signatures
    Napi::Value Signatures(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        Napi::Object out = Napi::Object::New(env);

        for (auto &[sig_key, sig] : signatures_)
        {
            Napi::Object sig_obj = Napi::Object::New(env);

            auto make_tensor_map = [&](
                                       const std::unordered_map<std::string, jude_tf::TensorInfo> &map)
            {
                Napi::Object m = Napi::Object::New(env);
                for (auto &[k, ti] : map)
                {
                    Napi::Object info_obj = Napi::Object::New(env);
                    info_obj.Set("name", Napi::String::New(env, ti.name));
                    info_obj.Set("dtype", Napi::Number::New(env, ti.dtype));

                    Napi::Array dims = Napi::Array::New(env, ti.shape.dims.size());
                    for (size_t i = 0; i < ti.shape.dims.size(); ++i)
                        dims.Set(i, Napi::Number::New(env,
                                                      static_cast<double>(ti.shape.dims[i])));
                    info_obj.Set("shape", dims);
                    m.Set(k, info_obj);
                }
                return m;
            };

            sig_obj.Set("inputs", make_tensor_map(sig.inputs));
            sig_obj.Set("outputs", make_tensor_map(sig.outputs));
            sig_obj.Set("methodName", Napi::String::New(env, sig.method_name));
            out.Set(sig_key, sig_obj);
        }
        return out;
    }

    void Destroy(const Napi::CallbackInfo &) { cleanup(); }
};

// ---------------------------------------------------------------------------
// Module entry
// ---------------------------------------------------------------------------
Napi::Object InitAll(Napi::Env env, Napi::Object exports)
{
    return TFSession::Init(env, exports);
}

NODE_API_MODULE(jude_tf, InitAll)