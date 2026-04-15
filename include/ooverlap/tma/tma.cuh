#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "ooverlap/sync/sync.cuh"

namespace ooverlap {
namespace tma {

__device__ __forceinline__ uint64_t cvta_to_global_u64(const void* ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint64_t out;
    asm volatile(
        "cvta.to.global.u64 %0, %1;\n"
        : "=l"(out)
        : "l"(ptr));
    return out;
#else
    return reinterpret_cast<uint64_t>(ptr);
#endif
}

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// -----------------------------------------------------------------------------
// LOAD-SIDE HELPERS (global -> shared)
// -----------------------------------------------------------------------------
//
// IMPORTANT:
// Hopper bulk-TMA load uses shared::cluster on the destination side.
// Using shared::cta here causes the "State space incorrect for instruction
// 'cp.async.bulk'" ptxas error you just hit.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void expect_bytes(sync::semaphore& bar, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const uint32_t bar_ptr = cvta_to_shared_u32(&bar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(bar_ptr), "r"(bytes)
        : "memory");
#else
    (void)bar;
    (void)bytes;
#endif
}

__device__ __forceinline__ void load_async(
    void* dst_smem,
    const void* src_gmem,
    uint32_t size_bytes,
    sync::semaphore& bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const uint32_t dst_smem_addr = cvta_to_shared_u32(dst_smem);
    const uint32_t bar_addr = cvta_to_shared_u32(&bar);
    const uint64_t src_gmem_addr = cvta_to_global_u64(src_gmem);

    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :
        : "r"(dst_smem_addr),
          "l"(src_gmem_addr),
          "r"(size_bytes),
          "r"(bar_addr)
        : "memory");
#else
    (void)bar;
    char* d = reinterpret_cast<char*>(dst_smem);
    const char* s = reinterpret_cast<const char*>(src_gmem);
    for (uint32_t i = 0; i < size_bytes; ++i) {
        d[i] = s[i];
    }
#endif
}

// -----------------------------------------------------------------------------
// STORE-SIDE HELPERS (shared -> global)
// -----------------------------------------------------------------------------

__device__ __forceinline__ void store_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.commit_group;\n");
#endif
}

template <int N = 0>
__device__ __forceinline__ void store_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.wait_group %0;\n"
        :
        : "n"(N)
        : "memory");
#endif
}

template <int N = 0>
__device__ __forceinline__ void store_async_read_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.wait_group.read %0;\n"
        :
        : "n"(N)
        : "memory");
#endif
}

__device__ __forceinline__ void store_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    store_commit_group();
#else
    char* d = reinterpret_cast<char*>(dst_gmem);
    const char* s = reinterpret_cast<const char*>(src_smem);
    for (uint32_t i = 0; i < size_bytes; ++i) {
        d[i] = s[i];
    }
#endif
}

} // namespace tma
} // namespace ooverlap
