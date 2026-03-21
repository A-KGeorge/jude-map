#pragma once

//----------------------------------------------------------------------------
// proto_parser.h
//
// Minimal binary protobuf decoder targeting exactly the fields needed to
// extract SignatureDef from as TensorFlow SavedModel's saved_model.pb.
//
// No dependencies - implements only the wire-format operations required:
//   varint decoding, length-delimited field traversal, nested message walking.
//
// Proto field numbers referenced:
//
//   SavedModel        { meta_graphs = 2 }
//   MetaGraphDef      { signature_def = 5 }      map<string, SignatureDef>
//   SignatureDef      { inputs = 1, outputs = 2, method_name = 3 }
//   TensorInfo        { name = 1, dtype = 2, tensor_shape = 3 }
//   TensorShapeProto  { dim = 2, unknown_rank = 3 }
//   TensorShapeProto.Dim { size = 1 }
//
// Map fields are encoded as repeated messages:
//   message MapEntry { string key = 1; T value = 2; }
// -----------------------------------------------------------------------------

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace jude_tf
{

    // -------------------------------------------------------------------------------
    // Wire types
    // -------------------------------------------------------------------------------
    enum WireType : uint8_t
    {
        WIRE_VARINT = 0,
        WIRE_64BIT = 1,
        WIRE_LEN_DELIMITED = 2,
        WIRE_32BIT = 5
    };

    struct ProtoReader
    {
        const uint8_t *ptr;
        const uint8_t *end;

        ProtoReader(const uint8_t *data, size_t len)
            : ptr(data), end(data + len) {}

        bool ok() const noexcept { return ptr < end; }
        bool depleated() const noexcept { return ptr >= end; }

        // Decode a varint: returns false on overrun.
        bool read_varint(uint64_t &out) noexcept
        {
            out = 0;
            uint32_t shift = 0;
            while (ptr < end)
            {
                uint8_t b = *ptr++;
                out |= static_cast<uint64_t>(b & 0x7Fu) << shift;
                if (!(b & 0x80u))
                    return true;
                shift += 7;
                if (shift >= 64)
                    return false; // varint too long
            }
            return false;
        }

        // Decode a tag: returns (field_number, wire_type)
        bool read_tag(uint32_t &field, WireType &wtype) noexcept
        {
            uint64_t raw;
            if (!read_varint(raw))
                return false;
            field = static_cast<uint32_t>(raw >> 3);
            wtype = static_cast<WireType>(raw & 0x07u);
            return field > 0;
        }

        // Skip a field with the given wire type.
        bool skip(WireType wtype) noexcept
        {
            switch (wtype)
            {
            case WIRE_VARINT:
                uint64_t dummy;
                return read_varint(dummy);
            case WIRE_64BIT:
                if (ptr + 8 > end)
                    return false;
                ptr += 8;
                return true;
            case WIRE_LEN_DELIMITED:
                uint64_t len;
                if (!read_varint(len))
                    return false;
                if (ptr + len > end)
                    return false;
                ptr += len;
                return true;
            case WIRE_32BIT:
                if (ptr + 4 > end)
                    return false;
                ptr += 4;
                return true;
            default:
                return false;
            }
        }

        // Read a length-delimited bytes/string/submessage span.
        bool read_bytes(const uint8_t *&data, size_t &len) noexcept
        {
            uint64_t raw_len;
            if (!read_varint(raw_len))
                return false;
            if (ptr + raw_len > end)
                return false;
            data = ptr;
            len = static_cast<size_t>(raw_len);
            ptr += raw_len;
            return true;
        }

        // Read a string field.
        bool read_string(std::string &out) noexcept
        {
            const uint8_t *data;
            size_t len;
            if (!read_bytes(data, len))
                return false;
            out.assign(reinterpret_cast<const char *>(data), len);
            return true;
        }

        // Return a sub-reader for a nested message without advancing past it.
        // (Caller must have already read the length prefix via read_bytes.)
        static ProtoReader sub(const uint8_t *data, size_t len) noexcept
        {
            return ProtoReader(data, len);
        }
    };

    //-----------------------------------------------------------------------------
    // Parsed types
    //-----------------------------------------------------------------------------

    struct TensorShape
    {
        std::vector<int64_t> dims; /// -1 = unknown dim
        bool unknown_rank = false;
    };

    struct TensorInfo
    {
        std::string name; // graph node name, e.g. "serving_default_x:0"
        int dtype;        // TF_DataType value
        TensorShape shape;
    };

    struct SignatureDef
    {
        std::string method_name; // e.g. "tensorflow/serving/predict"
        std::unordered_map<std::string, TensorInfo> inputs;
        std::unordered_map<std::string, TensorInfo> outputs;
    };

    // Map of signature key -> SignatureDef
    using SignatureMap = std::unordered_map<std::string, SignatureDef>;

    // -----------------------------------------------------------------------------
    // Internal parsers
    //------------------------------------------------------------------------------

    namespace detail
    {
        // Parse TensorShapeProto.Dim: { int64 size = 1; string name = 2; }
        inline bool parse_shape_dim(ProtoReader r, int64_t &size)
        {
            size = -1; // default: unknown
            while (r.ok())
            {
                uint32_t field;
                WireType wtype;
                if (!r.read_tag(field, wtype))
                    break;
                if (field == 1 && wtype == WIRE_VARINT)
                {
                    uint64_t v;
                    if (!r.read_varint(v))
                        return false;
                    size = static_cast<int64_t>(v);
                }
                else
                {
                    if (!r.skip(wtype))
                        return false;
                }
            }
            return true;
        }

        // Parse TensorShapeProto: { repeated Dim dim = 2; bool unknown_rank = 3; }
        inline bool parse_shape(ProtoReader r, TensorShape &out)
        {
            while (r.ok())
            {
                uint32_t field;
                WireType wtype;
                if (!r.read_tag(field, wtype))
                    break;
                if (field == 2 && wtype == WIRE_LEN_DELIMITED)
                {
                    const uint8_t *data;
                    size_t len;
                    if (!r.read_bytes(data, len))
                        return false;
                    int64_t size = -1;
                    if (!parse_shape_dim(ProtoReader::sub(data, len), size))
                        return false;
                    out.dims.push_back(size);
                }
                else if (field == 3 && wtype == WIRE_VARINT)
                {
                    uint64_t v;
                    if (!r.read_varint(v))
                        return false;
                    out.unknown_rank = (v != 0);
                }
                else
                {
                    if (!r.skip(wtype))
                        return false;
                }
            }
            return true;
        }

        // Parse TensorInfo: { string name = 1; int32 dtype = 2; TensorShapeProto shape = 3; }
        inline bool parse_tensor_info(ProtoReader r, TensorInfo &out)
        {
            while (r.ok())
            {
                uint32_t field;
                WireType wtype;
                if (!r.read_tag(field, wtype))
                    break;
                if (field == 1 && wtype == WIRE_LEN_DELIMITED)
                {
                    if (!r.read_string(out.name))
                        return false;
                }
                else if (field == 2 && wtype == WIRE_VARINT)
                {
                    uint64_t v;
                    if (!r.read_varint(v))
                        return false;
                    out.dtype = static_cast<int>(v);
                }
                else if (field == 3 && wtype == WIRE_LEN_DELIMITED)
                {
                    const uint8_t *data;
                    size_t len;
                    if (!r.read_bytes(data, len))
                        return false;
                    if (!parse_shape(ProtoReader::sub(data, len), out.shape))
                        return false;
                }
                else
                {
                    if (!r.skip(wtype))
                        return false;
                }
            }
            return true;
        }

        // Parse a map<string, TensorInfo> entry:
        //   message MapEntry { string key = 1; TensorInfo value = 2; }
        inline bool parse_tensor_map_entry(ProtoReader r, std::unordered_map<std::string, TensorInfo> &map)
        {
            std::string key;
            TensorInfo value;
            while (r.ok())
            {
                uint32_t field;
                WireType wtype;
                if (!r.read_tag(field, wtype))
                    break;
                if (field == 1 && wtype == WIRE_LEN_DELIMITED)
                {
                    if (!r.read_string(key))
                        return false;
                }
                else if (field == 2 && wtype == WIRE_LEN_DELIMITED)
                {
                    const uint8_t *data;
                    size_t len;
                    if (!r.read_bytes(data, len))
                        return false;
                    if (!parse_tensor_info(ProtoReader::sub(data, len), value))
                        return false;
                }
                else
                {
                    if (!r.skip(wtype))
                        return false;
                }
            }
            if (!key.empty())
                map.emplace(std::move(key), std::move(value));
            return true;
        }

        // Parse SignatureDef:
        //   { map<string,TensorInfo> inputs = 1;
        //     map<string,TensorInfo> outputs = 2;
        //     string method_name = 3; }
        inline bool parse_signature_def(ProtoReader r, SignatureDef &out)
        {
            while (r.ok())
            {
                uint32_t field;
                WireType wtype;
                if (!r.read_tag(field, wtype))
                    break;
                if (wtype == WIRE_LEN_DELIMITED &&
                    (field == 1 || field == 2))
                {
                    const uint8_t *data;
                    size_t len;
                    if (!r.read_bytes(data, len))
                        return false;
                    auto &map = (field == 1) ? out.inputs : out.outputs;
                    if (!parse_tensor_map_entry(ProtoReader::sub(data, len), map))
                        return false;
                }
                else if (field == 3 && wtype == WIRE_LEN_DELIMITED)
                {
                    if (!r.read_string(out.method_name))
                        return false;
                }
                else
                {
                    if (!r.skip(wtype))
                        return false;
                }
            }
            return true;
        }

        // Parse a map<string, SignatureDef> entry:
        //   message MapEntry { string key = 1; SignatureDef value = 2; }
        inline bool parse_signature_map_entry(ProtoReader r, SignatureMap &map)
        {
            std::string key;
            SignatureDef value;
            while (r.ok())
            {
                uint32_t field;
                WireType wtype;
                if (!r.read_tag(field, wtype))
                    break;
                if (field == 1 && wtype == WIRE_LEN_DELIMITED)
                {
                    if (!r.read_string(key))
                        return false;
                }
                else if (field == 2 && wtype == WIRE_LEN_DELIMITED)
                {
                    const uint8_t *data;
                    size_t len;
                    if (!r.read_bytes(data, len))
                        return false;
                    if (!parse_signature_def(ProtoReader::sub(data, len), value))
                        return false;
                }
                else
                {
                    if (!r.skip(wtype))
                        return false;
                }
            }
            if (!key.empty())
                map[key] = std::move(value);
            return true;
        }

        // Parse MetaGraphDef — only field 5 (signature_def) is extracted.
        inline bool parse_meta_graph(ProtoReader r, SignatureMap &sigs)
        {
            while (r.ok())
            {
                uint32_t field;
                WireType wtype;
                if (!r.read_tag(field, wtype))
                    break;
                if (field == 5 && wtype == WIRE_LEN_DELIMITED)
                {
                    const uint8_t *data;
                    size_t len;
                    if (!r.read_bytes(data, len))
                        return false;
                    if (!parse_signature_map_entry(ProtoReader::sub(data, len), sigs))
                        return false;
                }
                else
                {
                    if (!r.skip(wtype))
                        return false;
                }
            }
            return true;
        }
    } // namespace detail

    // -----------------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------------

    // Parse a SavedModel binary protobuf and return all SignatureDefs.
    // Returns false if the binary is malformed or contains no meta_graphs.
    inline bool parse_saved_model(const uint8_t *data, size_t len, SignatureMap &out)
    {
        // SavedModel: { int64 saved_model_schema_version = 1;
        //               repeated MetaGraphDef meta_graphs = 2; }
        ProtoReader r(data, len);
        bool found_any = false;

        while (r.ok())
        {
            uint32_t field;
            WireType wtype;
            if (!r.read_tag(field, wtype))
                break;
            if (field == 2 && wtype == WIRE_LEN_DELIMITED)
            {
                const uint8_t *mdata;
                size_t mlen;

                if (!r.read_bytes(mdata, mlen))
                    return false;
                if (!detail::parse_meta_graph(ProtoReader::sub(mdata, mlen), out))
                    return false;
                found_any = true;
            }
            else
            {
                if (!r.skip(wtype))
                    return false;
            }
        }
        return found_any;
    }

    // Convenience: pick the best signature from a SignatureMap.
    // Prefers "serving_default", then any signature with method_name containing
    // "predict", then the first available.
    // Cache the result or use a simpler strategy:
    inline const SignatureDef *pick_signature(const SignatureMap &sigs)
    {
        if (sigs.empty())
            return nullptr;

        // Try serving_default first (most common)
        auto it = sigs.find("serving_default");
        if (it != sigs.end())
            return &it->second;

        // Fall back to first signature with "predict"
        // This implementation is already optimal for a map
        for (auto &[key, sig] : sigs)
        {
            if (sig.method_name.find("predict") != std::string::npos)
                return &sig;
        }

        return &sigs.begin()->second;
    }
} // namespace jude_tf