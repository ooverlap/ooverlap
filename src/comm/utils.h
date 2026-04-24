#pragma once

#include <cuda_runtime.h>

namespace ooverlap {
namespace comm {
namespace utils {

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ int ceil_div_int64_to_int(
    size_t num,
    size_t den) {
    return static_cast<int>((num + den - 1) / den);
}

__host__ __device__ __forceinline__ int min_int(int a, int b) {
    return (a < b) ? a : b;
}

}
}
}
