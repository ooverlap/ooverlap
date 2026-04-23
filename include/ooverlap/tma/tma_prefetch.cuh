#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "ooverlap/tma/tma.cuh"

namespace ooverlap {
namespace tma {

// -----------------------------------------------------------------------------
// PREFETCH HELPERS
// -----------------------------------------------------------------------------
//
// PTX:
//
//   cp.async.bulk.prefetch.L2.global [srcMem], size;
//   cp.async.bulk.prefetch.L2.global.L2::cache_hint [srcMem], size, cache_policy;
//
// Notes:
// - srcMem must be 16-byte aligned
// - size_bytes must be a multiple of 16
// - this is only a performance hint
// - there is no commit/wait mechanism for this instruction
// -----------------------------------------------------------------------------

__device__ __forceinline__ void prefetch_L2(
    const void* src_gmem,
    uint32_t size_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.prefetch.L2.global [%0], %1;\n"
        :
        : "l"(cvta_to_global_u64(src_gmem)),
          "r"(size_bytes)
        : "memory");
#else
    (void)src_gmem;
    (void)size_bytes;
#endif
}

__device__ __forceinline__ void prefetch_L2_cache_hint(
    const void* src_gmem,
    uint32_t size_bytes,
    uint64_t cache_policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n"
        :
        : "l"(cvta_to_global_u64(src_gmem)),
          "r"(size_bytes),
          "l"(cache_policy)
        : "memory");
#else
    (void)src_gmem;
    (void)size_bytes;
    (void)cache_policy;
#endif
}

} // namespace tma
} // namespace ooverlap
