#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "ooverlap/tma/tma.cuh"

namespace ooverlap {
namespace tma {

__device__ __forceinline__ void reduce_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.commit_group;\n");
#endif
}

template <int N = 0>
__device__ __forceinline__ void reduce_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.wait_group %0;\n"
        :
        : "n"(N)
        : "memory");
#endif
}

template <int N = 0>
__device__ __forceinline__ void reduce_async_read_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.wait_group.read %0;\n"
        :
        : "n"(N)
        : "memory");
#endif
}

__device__ __forceinline__ void reduce_add_f16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f16 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
#else
    // No fallback here on purpose; reduction fallback should happen at callsite.
    (void)dst_gmem;
    (void)src_smem;
    (void)size_bytes;
#endif
}

__device__ __forceinline__ void reduce_add_noftz_f16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
#else
    (void)dst_gmem;
    (void)src_smem;
    (void)size_bytes;
#endif
}

} // namespace tma
} // namespace ooverlap
