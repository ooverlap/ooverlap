#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace utils {

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

} // namespace utils
} // namespace comm
} // namespace ooverlap
