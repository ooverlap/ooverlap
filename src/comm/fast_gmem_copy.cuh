#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace fast_copy {

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

template <typename VecT>
__device__ __forceinline__ VecT load_copy_vec(
    const VecT* __restrict__ ptr) {
    return ptr[0];
}

template <typename VecT>
__device__ __forceinline__ void store_copy_vec(
    VecT* __restrict__ ptr,
    VecT value) {
    ptr[0] = value;
}

template <>
__device__ __forceinline__ uint4 load_copy_vec<uint4>(
    const uint4* __restrict__ ptr) {
    uint4 value;

    asm volatile(
        "{\n"
        "  .reg .u64 addr;\n"
        "  cvta.to.global.u64 addr, %4;\n"
        "  ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [addr];\n"
        "}\n"
        : "=r"(value.x),
          "=r"(value.y),
          "=r"(value.z),
          "=r"(value.w)
        : "l"(ptr));

    return value;
}

template <>
__device__ __forceinline__ void store_copy_vec<uint4>(
    uint4* __restrict__ ptr,
    uint4 value) {
    asm volatile(
        "{\n"
        "  .reg .u64 addr;\n"
        "  cvta.to.global.u64 addr, %0;\n"
        "  st.global.L1::no_allocate.v4.u32 [addr], {%1, %2, %3, %4};\n"
        "}\n"
        :
        : "l"(ptr),
          "r"(value.x),
          "r"(value.y),
          "r"(value.z),
          "r"(value.w)
        : "memory");
}

/*
 * Copy a contiguous VecT span with a caller-selected thread group.
 *
 * This is the piece we will reuse inside the future allreduce kernel for the
 * warp-specialized consumer copy path. The caller chooses lane/lane_count, so
 * this can be driven by a full CTA, a single warp, or any contiguous subgroup.
 */
template <typename VecT, int Unroll>
__device__ __forceinline__ void copy_vec_span(
    const VecT* __restrict__ src_vec,
    VecT* __restrict__ dst_vec,
    size_t begin_vec,
    size_t end_vec,
    size_t lane,
    size_t lane_count) {
    const size_t step = lane_count * static_cast<size_t>(Unroll);

    size_t base = begin_vec + lane;

    for (; base + static_cast<size_t>(Unroll - 1) * lane_count < end_vec;
         base += step) {
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const size_t i = base + static_cast<size_t>(u) * lane_count;
            const VecT value = load_copy_vec<VecT>(src_vec + i);
            store_copy_vec<VecT>(dst_vec + i, value);
        }
    }

    for (; base < end_vec; base += lane_count) {
        const VecT value = load_copy_vec<VecT>(src_vec + base);
        store_copy_vec<VecT>(dst_vec + base, value);
    }
}

/*
 * Copy a byte range using VecT vector traffic plus a byte tail.
 *
 * byte_offset should be VecT-aligned for the vectorized path. The pivot design
 * will naturally use chunk boundaries, so that should hold for uint4.
 */
template <typename VecT, int Unroll>
__device__ __forceinline__ void copy_byte_range(
    const void* __restrict__ src,
    void* __restrict__ dst,
    size_t byte_offset,
    size_t byte_count,
    size_t lane,
    size_t lane_count) {
    const unsigned char* __restrict__ src_u8 =
        reinterpret_cast<const unsigned char*>(src);
    unsigned char* __restrict__ dst_u8 =
        reinterpret_cast<unsigned char*>(dst);

    const size_t vec_size = sizeof(VecT);
    const size_t vec_begin = byte_offset / vec_size;
    const size_t vec_count = byte_count / vec_size;
    const size_t vec_end = vec_begin + vec_count;

    const VecT* __restrict__ src_vec =
        reinterpret_cast<const VecT*>(src);
    VecT* __restrict__ dst_vec = reinterpret_cast<VecT*>(dst);

    copy_vec_span<VecT, Unroll>(
        src_vec,
        dst_vec,
        vec_begin,
        vec_end,
        lane,
        lane_count);

    const size_t tail_begin = byte_offset + vec_count * vec_size;
    const size_t tail_end = byte_offset + byte_count;

    for (size_t i = tail_begin + lane; i < tail_end; i += lane_count) {
        dst_u8[i] = src_u8[i];
    }
}

/*
 * Standalone benchmark kernel. Each CTA owns one contiguous slice. Within the
 * CTA, consecutive lanes access consecutive VecT elements, and each thread
 * batches Unroll loads before stores. This matches the fast copy variant that
 * performed best for the local-to-peer path in the benchmark.
 */
template <typename VecT, int Unroll = 8>
__global__ void gmem_copy_coalesced_kernel(
    const void* __restrict__ src,
    void* __restrict__ dst,
    size_t total_bytes) {
    const unsigned char* __restrict__ src_u8 =
        reinterpret_cast<const unsigned char*>(src);
    unsigned char* __restrict__ dst_u8 =
        reinterpret_cast<unsigned char*>(dst);

    const size_t total_vec = total_bytes / sizeof(VecT);

    const size_t block = static_cast<size_t>(blockIdx.x);
    const size_t nblocks = static_cast<size_t>(gridDim.x);
    const size_t lane = static_cast<size_t>(threadIdx.x);
    const size_t block_threads = static_cast<size_t>(blockDim.x);

    constexpr size_t kWarpElems = 32;

    const size_t raw_vecs_per_block =
        (total_vec + nblocks - 1) / nblocks;

    const size_t vecs_per_block =
        ((raw_vecs_per_block + kWarpElems - 1) / kWarpElems) * kWarpElems;

    const size_t block_begin = block * vecs_per_block;
    const size_t block_end = min_sz(total_vec, block_begin + vecs_per_block);

    const VecT* __restrict__ src_vec =
        reinterpret_cast<const VecT*>(src);
    VecT* __restrict__ dst_vec = reinterpret_cast<VecT*>(dst);

    copy_vec_span<VecT, Unroll>(
        src_vec,
        dst_vec,
        block_begin,
        block_end,
        lane,
        block_threads);

    const size_t tail_begin = total_vec * sizeof(VecT);

    if (blockIdx.x == 0) {
        for (size_t i = tail_begin + lane;
             i < total_bytes;
             i += block_threads) {
            dst_u8[i] = src_u8[i];
        }
    }
}

} // namespace fast_copy
} // namespace comm
} // namespace ooverlap
