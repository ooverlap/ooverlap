#pragma once

#include "ooverlap/comm.h"

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace utils {

inline bool reduce_op_supported_for_dtype(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16 ||
               dtype == OO_DTYPE_FLOAT32;
    }

    if (op == OO_REDUCE_MIN || op == OO_REDUCE_MAX) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16;
    }

    return false;
}

inline bool dtype_supported_for_copy_collective(oo_dtype_t dtype) {
    return dtype == OO_DTYPE_FLOAT16 ||
           dtype == OO_DTYPE_BFLOAT16 ||
           dtype == OO_DTYPE_FLOAT32;
}

inline bool rank_partition(
    size_t count,
    int rank,
    int world_size,
    size_t* out_begin,
    size_t* out_count) {
    if (out_begin == nullptr || out_count == nullptr) {
        return false;
    }

    *out_begin = 0;
    *out_count = 0;

    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        return false;
    }

    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);

    const size_t base = count / world;
    const size_t rem = count % world;

    *out_begin = r * base + ((r < rem) ? r : rem);
    *out_count = base + ((r < rem) ? 1 : 0);

    return true;
}

inline const void* offset_const_ptr(
    const void* ptr,
    size_t byte_offset) {
    return reinterpret_cast<const void*>(
        reinterpret_cast<const unsigned char*>(ptr) + byte_offset);
}

inline void* offset_ptr(
    void* ptr,
    size_t byte_offset) {
    return reinterpret_cast<void*>(
        reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

} // namespace utils
} // namespace comm
} // namespace ooverlap
