#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace ooverlap {
namespace tma {

// We keep these wrappers separate from plain store_async() because the semantics
// are different: this is "reduce into destination" rather than "overwrite destination".

enum class reduce_op : int {
    add = 0,
    // later: min, max, and, or, xor ... only if/when PTX support is verified
};

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void reduce_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n");
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
// IMPORTANT
// -----------------------------------------------------------------------------
// This file is intentionally split into:
//   1) stable wrapper API
//   2) one PTX-emission point per op/type
//
// The repo currently does NOT contain a verified reduction-form PTX mnemonic,
// so do not guess that line. Fill the asm string only after checking the exact
// Hopper PTX ISA spelling and the supported dtype/op combination you want.
//
// The outer API is what you can already adopt in the codebase.
// -----------------------------------------------------------------------------

template <typename T>
struct reduce_add_async_impl;

// Example specialization point.
// Fill this only after verifying the exact PTX instruction form for your type.
//
// template <>
// struct reduce_add_async_impl<uint32_t> {
//     __device__ __forceinline__ static void run(
//         void* dst_gmem,
//         void* src_smem,
//         uint32_t size_bytes) {
//         asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
//         asm volatile(
//             "TODO_VERIFIED_REDUCTION_PTX [%0], [%1], %2;\n"
//             :
//             : "l"(dst_gmem),
//               "r"(smem_ptr_u32(src_smem)),
//               "r"(size_bytes)
//             : "memory");
//         reduce_commit_group();
//     }
// };

template <typename T>
__device__ __forceinline__ void reduce_add_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    reduce_add_async_impl<T>::run(dst_gmem, src_smem, size_bytes);
}

} // namespace tma
} // namespace ooverlap
