#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace kernels {
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
 * Add packed f16x2 values stored in one u32 register.
 *
 * Each uint4 contains 4 x u32 = 8 x fp16 values.
 */
__device__ __forceinline__ uint32_t add_f16x2_bits(
    uint32_t a,
    uint32_t b) {
    uint32_t out;

    asm volatile(
        "{\n"
        "  add.rn.f16x2 %0, %1, %2;\n"
        "}\n"
        : "=r"(out)
        : "r"(a), "r"(b));

    return out;
}

__device__ __forceinline__ uint4 add_f16x8_vec(
    uint4 a,
    uint4 b) {
    uint4 out;
    out.x = add_f16x2_bits(a.x, b.x);
    out.y = add_f16x2_bits(a.y, b.y);
    out.z = add_f16x2_bits(a.z, b.z);
    out.w = add_f16x2_bits(a.w, b.w);
    return out;
}

/*
 * Copy a contiguous VecT span with a caller-selected thread group.
 *
 * Important:
 * For Unroll == 8, this intentionally loads all 8 values first and only then
 * stores all 8 values. This matches the fast benchmark kernel. Do not turn this
 * into load/store pairs; that was slower for the local-to-peer push path.
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

    if constexpr (Unroll == 8) {
        for (; base + static_cast<size_t>(7) * lane_count < end_vec;
             base += step) {
            const VecT v0 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(0) * lane_count);
            const VecT v1 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(1) * lane_count);
            const VecT v2 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(2) * lane_count);
            const VecT v3 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(3) * lane_count);
            const VecT v4 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(4) * lane_count);
            const VecT v5 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(5) * lane_count);
            const VecT v6 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(6) * lane_count);
            const VecT v7 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(7) * lane_count);

            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(0) * lane_count,
                v0);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(1) * lane_count,
                v1);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(2) * lane_count,
                v2);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(3) * lane_count,
                v3);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(4) * lane_count,
                v4);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(5) * lane_count,
                v5);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(6) * lane_count,
                v6);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(7) * lane_count,
                v7);
        }
    } else {
        for (; base + static_cast<size_t>(Unroll - 1) * lane_count < end_vec;
             base += step) {
#pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const size_t i =
                    base + static_cast<size_t>(u) * lane_count;
                const VecT value = load_copy_vec<VecT>(src_vec + i);
                store_copy_vec<VecT>(dst_vec + i, value);
            }
        }
    }

    for (; base < end_vec; base += lane_count) {
        const VecT value = load_copy_vec<VecT>(src_vec + base);
        store_copy_vec<VecT>(dst_vec + base, value);
    }
}

/*
 * Register-only fp16 add over a uint4 span:
 *
 *     dst[i] = dst[i] + src[i]
 *
 * No smem. No atomics. One writer per element is assumed.
 *
 * For Unroll == 4, each thread loads 4 src uint4 values and 4 dst uint4 values,
 * then writes 4 uint4 results. This is lower-register than Unroll == 8 while
 * still giving enough independent memory operations.
 */
template <int Unroll>
__device__ __forceinline__ void add_f16_u128_vec_span(
    const uint4* __restrict__ src_vec,
    uint4* __restrict__ dst_vec,
    size_t begin_vec,
    size_t end_vec,
    size_t lane,
    size_t lane_count) {
    const size_t step = lane_count * static_cast<size_t>(Unroll);

    size_t base = begin_vec + lane;

    if constexpr (Unroll == 4) {
        for (; base + static_cast<size_t>(3) * lane_count < end_vec;
             base += step) {
            const size_t i0 = base + static_cast<size_t>(0) * lane_count;
            const size_t i1 = base + static_cast<size_t>(1) * lane_count;
            const size_t i2 = base + static_cast<size_t>(2) * lane_count;
            const size_t i3 = base + static_cast<size_t>(3) * lane_count;

            const uint4 s0 = load_copy_vec<uint4>(src_vec + i0);
            const uint4 s1 = load_copy_vec<uint4>(src_vec + i1);
            const uint4 s2 = load_copy_vec<uint4>(src_vec + i2);
            const uint4 s3 = load_copy_vec<uint4>(src_vec + i3);

            const uint4 d0 = load_copy_vec<uint4>(dst_vec + i0);
            const uint4 d1 = load_copy_vec<uint4>(dst_vec + i1);
            const uint4 d2 = load_copy_vec<uint4>(dst_vec + i2);
            const uint4 d3 = load_copy_vec<uint4>(dst_vec + i3);

            store_copy_vec<uint4>(dst_vec + i0, add_f16x8_vec(d0, s0));
            store_copy_vec<uint4>(dst_vec + i1, add_f16x8_vec(d1, s1));
            store_copy_vec<uint4>(dst_vec + i2, add_f16x8_vec(d2, s2));
            store_copy_vec<uint4>(dst_vec + i3, add_f16x8_vec(d3, s3));
        }
    } else {
        for (; base + static_cast<size_t>(Unroll - 1) * lane_count < end_vec;
             base += step) {
#pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const size_t i =
                    base + static_cast<size_t>(u) * lane_count;

                const uint4 s = load_copy_vec<uint4>(src_vec + i);
                const uint4 d = load_copy_vec<uint4>(dst_vec + i);

                store_copy_vec<uint4>(dst_vec + i, add_f16x8_vec(d, s));
            }
        }
    }

    for (; base < end_vec; base += lane_count) {
        const uint4 s = load_copy_vec<uint4>(src_vec + base);
        const uint4 d = load_copy_vec<uint4>(dst_vec + base);

        store_copy_vec<uint4>(dst_vec + base, add_f16x8_vec(d, s));
    }
}

/*
 * Copy a byte range using VecT vector traffic plus a byte tail.
 *
 * byte_offset should be VecT-aligned for the vectorized path.
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
    VecT* __restrict__ dst_vec =
        reinterpret_cast<VecT*>(dst);

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
 * Register-only fp16 add over a byte range:
 *
 *     dst_half[i] = dst_half[i] + src_half[i]
 *
 * Vector path uses uint4 / f16x2. Tail path uses scalar half add.
 */
template <int Unroll>
__device__ __forceinline__ void add_f16_u128_byte_range(
    const void* __restrict__ src,
    void* __restrict__ dst,
    size_t byte_offset,
    size_t byte_count,
    size_t lane,
    size_t lane_count) {
    const size_t vec_size = sizeof(uint4);
    const size_t vec_begin = byte_offset / vec_size;
    const size_t vec_count = byte_count / vec_size;
    const size_t vec_end = vec_begin + vec_count;

    const uint4* __restrict__ src_vec =
        reinterpret_cast<const uint4*>(src);
    uint4* __restrict__ dst_vec =
        reinterpret_cast<uint4*>(dst);

    add_f16_u128_vec_span<Unroll>(
        src_vec,
        dst_vec,
        vec_begin,
        vec_end,
        lane,
        lane_count);

    const size_t tail_begin = byte_offset + vec_count * vec_size;
    const size_t tail_bytes = byte_count - vec_count * vec_size;
    const size_t tail_half = tail_bytes / sizeof(half);

    const unsigned char* __restrict__ src_u8 =
        reinterpret_cast<const unsigned char*>(src);
    unsigned char* __restrict__ dst_u8 =
        reinterpret_cast<unsigned char*>(dst);

    const half* __restrict__ src_h =
        reinterpret_cast<const half*>(src_u8 + tail_begin);
    half* __restrict__ dst_h =
        reinterpret_cast<half*>(dst_u8 + tail_begin);

    for (size_t i = lane; i < tail_half; i += lane_count) {
        dst_h[i] = __hadd(dst_h[i], src_h[i]);
    }
}

/*
 * Standalone benchmark copy kernel.
 *
 * This is intentionally written in the same shape as the fast version:
 * - each CTA owns one contiguous slice,
 * - each thread loads 8 VecT values first,
 * - then stores those 8 VecT values.
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
    const size_t block_end =
        min_sz(total_vec, block_begin + vecs_per_block);

    const VecT* __restrict__ src_vec =
        reinterpret_cast<const VecT*>(src);
    VecT* __restrict__ dst_vec =
        reinterpret_cast<VecT*>(dst);

    const size_t step = block_threads * static_cast<size_t>(Unroll);

    size_t base = block_begin + lane;

    if constexpr (Unroll == 8) {
        for (; base + static_cast<size_t>(7) * block_threads < block_end;
             base += step) {
            const VecT v0 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(0) * block_threads);
            const VecT v1 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(1) * block_threads);
            const VecT v2 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(2) * block_threads);
            const VecT v3 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(3) * block_threads);
            const VecT v4 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(4) * block_threads);
            const VecT v5 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(5) * block_threads);
            const VecT v6 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(6) * block_threads);
            const VecT v7 = load_copy_vec<VecT>(
                src_vec + base + static_cast<size_t>(7) * block_threads);

            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(0) * block_threads,
                v0);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(1) * block_threads,
                v1);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(2) * block_threads,
                v2);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(3) * block_threads,
                v3);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(4) * block_threads,
                v4);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(5) * block_threads,
                v5);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(6) * block_threads,
                v6);
            store_copy_vec<VecT>(
                dst_vec + base + static_cast<size_t>(7) * block_threads,
                v7);
        }
    } else {
        for (; base + static_cast<size_t>(Unroll - 1) * block_threads <
               block_end;
             base += step) {
#pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const size_t i =
                    base + static_cast<size_t>(u) * block_threads;
                const VecT value = load_copy_vec<VecT>(src_vec + i);
                store_copy_vec<VecT>(dst_vec + i, value);
            }
        }
    }

    for (; base < block_end; base += block_threads) {
        const VecT value = load_copy_vec<VecT>(src_vec + base);
        store_copy_vec<VecT>(dst_vec + base, value);
    }

    const size_t tail_begin = total_vec * sizeof(VecT);

    if (blockIdx.x == 0) {
        for (size_t i = tail_begin + lane;
             i < total_bytes;
             i += block_threads) {
            dst_u8[i] = src_u8[i];
        }
    }
}

/*
 * Standalone benchmark add kernel:
 *
 *     dst = dst + src
 *
 * This is the register-only path to compare against TMA reduce.
 */
template <int Unroll = 4>
__global__ void gmem_add_f16_u128_kernel(
    const void* __restrict__ src,
    void* __restrict__ dst,
    size_t total_bytes) {
    const size_t total_vec = total_bytes / sizeof(uint4);

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
    const size_t block_end =
        min_sz(total_vec, block_begin + vecs_per_block);

    const uint4* __restrict__ src_vec =
        reinterpret_cast<const uint4*>(src);
    uint4* __restrict__ dst_vec =
        reinterpret_cast<uint4*>(dst);

    add_f16_u128_vec_span<Unroll>(
        src_vec,
        dst_vec,
        block_begin,
        block_end,
        lane,
        block_threads);

    const size_t tail_begin = total_vec * sizeof(uint4);
    const size_t tail_bytes = total_bytes - tail_begin;
    const size_t tail_half = tail_bytes / sizeof(half);

    const unsigned char* __restrict__ src_u8 =
        reinterpret_cast<const unsigned char*>(src);
    unsigned char* __restrict__ dst_u8 =
        reinterpret_cast<unsigned char*>(dst);

    const half* __restrict__ src_h =
        reinterpret_cast<const half*>(src_u8 + tail_begin);
    half* __restrict__ dst_h =
        reinterpret_cast<half*>(dst_u8 + tail_begin);

    if (blockIdx.x == 0) {
        for (size_t i = lane; i < tail_half; i += block_threads) {
            dst_h[i] = __hadd(dst_h[i], src_h[i]);
        }
    }
}

} // namespace fast_copy
} // namespace kernels
} // namespace comm
} // namespace ooverlap
