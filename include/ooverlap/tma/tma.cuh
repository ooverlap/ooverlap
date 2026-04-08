#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "sync/sync.cuh"

namespace ooverlap {
namespace tma {

__device__ __forceinline__ void expect_bytes(sync::semaphore& bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(bar_ptr), "r"(bytes));
}

__device__ __forceinline__ void store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n");
}

template <int N = 0>
__device__ __forceinline__ void store_async_wait() {
    asm volatile(
        "cp.async.bulk.wait_group %0;\n"
        :
        : "n"(N)
        : "memory");
}

template <int N = 0>
__device__ __forceinline__ void store_async_read_wait() {
    asm volatile(
        "cp.async.bulk.wait_group.read %0;\n"
        :
        : "n"(N)
        : "memory");
}

__device__ __forceinline__ void load_async(
    void* dst_smem,
    const void* src_gmem,
    uint32_t size_bytes,
    sync::semaphore& bar) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem))),
          "l"(src_gmem),
          "r"(size_bytes),
          "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&bar)))
        : "memory");
}

__device__ __forceinline__ void store_async(
    void* dst_gmem,
    const void* src_smem,
    uint32_t size_bytes) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group "
        "[%0], [%1], %2;\n"
        :
        : "l"(dst_gmem),
          "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_smem))),
          "r"(size_bytes)
        : "memory");
    store_commit_group();
}

} // namespace tma
} // namespace ooverlap
