#pragma once

#include "ooverlap/mem/fast_copy.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace kernels {
namespace fast_add {

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
            const size_t i0 = base + 0 * lane_count;
            const size_t i1 = base + 1 * lane_count;
            const size_t i2 = base + 2 * lane_count;
            const size_t i3 = base + 3 * lane_count;

            const uint4 s0 =
                fast_copy::load_copy_vec<uint4>(src_vec + i0);
            const uint4 s1 =
                fast_copy::load_copy_vec<uint4>(src_vec + i1);
            const uint4 s2 =
                fast_copy::load_copy_vec<uint4>(src_vec + i2);
            const uint4 s3 =
                fast_copy::load_copy_vec<uint4>(src_vec + i3);

            const uint4 d0 =
                fast_copy::load_copy_vec<uint4>(dst_vec + i0);
            const uint4 d1 =
                fast_copy::load_copy_vec<uint4>(dst_vec + i1);
            const uint4 d2 =
                fast_copy::load_copy_vec<uint4>(dst_vec + i2);
            const uint4 d3 =
                fast_copy::load_copy_vec<uint4>(dst_vec + i3);

            fast_copy::store_copy_vec<uint4>(
                dst_vec + i0,
                add_f16x8_vec(d0, s0));

            fast_copy::store_copy_vec<uint4>(
                dst_vec + i1,
                add_f16x8_vec(d1, s1));

            fast_copy::store_copy_vec<uint4>(
                dst_vec + i2,
                add_f16x8_vec(d2, s2));

            fast_copy::store_copy_vec<uint4>(
                dst_vec + i3,
                add_f16x8_vec(d3, s3));
        }
    } else {
        for (; base + static_cast<size_t>(Unroll - 1) * lane_count < end_vec;
             base += step) {
#pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const size_t i =
                    base + static_cast<size_t>(u) * lane_count;

                const uint4 s =
                    fast_copy::load_copy_vec<uint4>(src_vec + i);

                const uint4 d =
                    fast_copy::load_copy_vec<uint4>(dst_vec + i);

                fast_copy::store_copy_vec<uint4>(
                    dst_vec + i,
                    add_f16x8_vec(d, s));
            }
        }
    }

    for (; base < end_vec; base += lane_count) {
        const uint4 s =
            fast_copy::load_copy_vec<uint4>(src_vec + base);

        const uint4 d =
            fast_copy::load_copy_vec<uint4>(dst_vec + base);

        fast_copy::store_copy_vec<uint4>(
            dst_vec + base,
            add_f16x8_vec(d, s));
    }
}

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

    const size_t tail_begin =
        byte_offset + vec_count * vec_size;

    const size_t tail_bytes =
        byte_count - vec_count * vec_size;

    const size_t tail_half =
        tail_bytes / sizeof(half);

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

} // namespace fast_add
} // namespace kernels
} // namespace comm
} // namespace ooverlap
