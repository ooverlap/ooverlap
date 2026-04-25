#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "ooverlap/tma/tma.cuh"

namespace ooverlap {
namespace tma {

// -----------------------------------------------------------------------------
// SM90 TMA bulk reduction helpers
// -----------------------------------------------------------------------------
//
// These wrappers intentionally do not provide a non-SM90 fallback.
// If this file is compiled for an unsupported architecture or unsupported PTX
// operation/type combination, we want the compiler/assembler to fail loudly.
//
// TMA bulk reduce form:
//
//   cp.reduce.async.bulk.global.shared::cta.bulk_group.<op>.<type>
//
// Source is CTA shared memory.
// Destination is global memory.
//
// Exposed floating-point reductions:
//
//   add:
//     f16
//     add.noftz.f16
//     bf16
//     f32
//
//   min:
//     f16
//     bf16
//     f32
//
//   max:
//     f16
//     bf16
//     f32
//
// Notes:
//   - add is the sum operation.
//   - sub is not exposed because TMA reduce has no native subtraction op.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void reduce_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N = 0>
__device__ __forceinline__ void reduce_async_wait() {
    asm volatile(
        "cp.async.bulk.wait_group %0;\n"
        :
        : "n"(N)
        : "memory");
}

template <int N = 0>
__device__ __forceinline__ void reduce_async_read_wait() {
    asm volatile(
        "cp.async.bulk.wait_group.read %0;\n"
        :
        : "n"(N)
        : "memory");
}

// -----------------------------------------------------------------------------
// add
// -----------------------------------------------------------------------------

__device__ __forceinline__ void reduce_add_f16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
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
}

__device__ __forceinline__ void reduce_add_noftz_f16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
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
}

__device__ __forceinline__ void reduce_add_bf16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.bf16 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
}

__device__ __forceinline__ void reduce_add_f32_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
}

// -----------------------------------------------------------------------------
// min
// -----------------------------------------------------------------------------

__device__ __forceinline__ void reduce_min_f16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
}

__device__ __forceinline__ void reduce_min_bf16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
}

__device__ __forceinline__ void reduce_min_f32_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f32 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
}

// -----------------------------------------------------------------------------
// max
// -----------------------------------------------------------------------------

__device__ __forceinline__ void reduce_max_f16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f16 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
}

__device__ __forceinline__ void reduce_max_bf16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
}

__device__ __forceinline__ void reduce_max_f32_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f32 "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
    reduce_commit_group();
}

} // namespace tma
} // namespace ooverlap
