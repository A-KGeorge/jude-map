#pragma once
#include "seqlock.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------
// Dtype
// ---------------------------------------------------

enum class DType : uint32_t
{
    FLOAT32 = 0,
    FLOAT64 = 1,
    INT32 = 2,
    INT64 = 3,
    UINT8 = 4,
    INT8 = 5,
    UINT16 = 6,
    INT16 = 7,
    BOOL = 8,
    UNKNOWN = 0xFFFFFFFF, // ← add this line
};

inline size_t dtype_itemsize(DType dt)
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
        throw std::invalid_argument("Unsupported dtype");
    }
}

inline const char *dtype_name(DType dt)
{
    switch (dt)
    {
    case DType::FLOAT32:
        return "float32";
    case DType::FLOAT64:
        return "float64";
    case DType::INT32:
        return "int32";
    case DType::INT64:
        return "int64";
    case DType::UINT8:
        return "uint8";
    case DType::INT8:
        return "int8";
    case DType::UINT16:
        return "uint16";
    case DType::INT16:
        return "int16";
    case DType::BOOL:
        return "bool";
    default:
        throw std::invalid_argument("Unsupported dtype");
    }
}

// ---------------------------------------------------
// Segment layout
//
// [ SegmentHeader ][ tensor bytes ... ]
//
// SegmentHeader:
//  [0..63]     Seqlock     (cache line 0)
//  [64..127]   TensorMeta  (cache line 1)
//
// Tensor data follows at DATA_OFFSET = sizeof(SegmentHeader).
// SegmentHeader is a fixed 128 bytes so DATA_OFFSET is a compile-time constant.
// ---------------------------------------------------

static constexpr uint32_t MAX_DIMS = 8;

// Metadata block - isolated to its own cache line so readers can load it
// without the seqlock counter causing a false-share invalidation.
struct alignas(64) TensorMeta
{
    uint32_t ndim;            // 4 bytes - number of dimensions [0..MAX_DIMS]
    DType dtype;              // 4 bytes - element type
    uint64_t shape[MAX_DIMS]; // 64 bytes - dimension sizes (unused dims = 0)
    uint64_t byte_length;     // 8 bytes - total bytes of valid tensor data
    uint8_t _pad[48];         // 48 bytes - pad to 128 (two cache lines)
    // total 128 bytes
};

static_assert(sizeof(TensorMeta) == 128, "TensorMeta must be exactly 128 bytes (two cache lines) to avoid false sharing with the seqlock counter");

struct SegmentHeader
{
    Seqlock seqlock;      // 64 bytes - cache line 0
    TensorMeta meta;      // 128 bytes - cache line 1
    uint8_t _gpu_pad[64]; // 64 bytes - pad to 256 for CUDA DMA alignment
};

static_assert(sizeof(SegmentHeader) == 256, "SegmentHeader must be exactly 256 bytes");
static_assert(offsetof(SegmentHeader, meta) == 64, "meta must start at chace line boundary");
static_assert(offsetof(SegmentHeader, _gpu_pad) == 192, "_gpu_pad must start at byte 192");

// Byte offset where tensor data begins.
// 256-byte alignment satisfies CUDA async DMA requirements. When this buffer
// is wrapped via TF_NewTensor(..., TF_DONT_DEALLOCATE_CONTENTS, ...) the
// gpu_private host threads can initiate H2D transfers without a staging copy,
// provided the mapping is also page-locked (see platform_mlock).
static constexpr size_t DATA_OFFSET = sizeof(SegmentHeader); // 256 bytes

inline uint8_t *segment_data_ptr(void *base)
{
    return reinterpret_cast<uint8_t *>(base) + DATA_OFFSET;
}

inline const uint8_t *segment_data_ptr(const void *base)
{
    return reinterpret_cast<const uint8_t *>(base) + DATA_OFFSET;
}

// Minimum mmap size needed to hold a tensor of 'byte_length' bytes.
inline size_t segment_min_size(size_t byte_length)
{
    return DATA_OFFSET + byte_length;
}
