#pragma once

#include "comm/kernels/fast_add.cuh"
#include "comm/kernels/fast_copy.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace kernels {
namespace fast_copy {

template <int Unroll = 4>
__global__ void gmem_add_f16_u128_kernel(
    const void* __restrict__ src,
    void* __restrict__ dst,
    size_t total_bytes) {
    const size_t total_vec =
        total_bytes / sizeof(uint4);

    const size_t block =
        static_cast<size_t>(blockIdx.x);

    const size_t nblocks =
        static_cast<size_t>(gridDim.x);

    const size_t lane =
        static_cast<size_t>(threadIdx.x);

    const size_t block_threads =
        static_cast<size_t>(blockDim.x);

    constexpr size_t kWarpElems = 32;

    const size_t raw_vecs_per_block =
        (total_vec + nblocks - 1) / nblocks;

    const size_t vecs_per_block =
        ((raw_vecs_per_block + kWarpElems - 1) / kWarpElems) * kWarpElems;

    const size_t block_begin =
        block * vecs_per_block;

    const size_t block_end =
        min_sz(total_vec, block_begin + vecs_per_block);

    const uint4* __restrict__ src_vec =
        reinterpret_cast<const uint4*>(src);

    uint4* __restrict__ dst_vec =
        reinterpret_cast<uint4*>(dst);

    fast_add::add_f16_u128_vec_span<Unroll>(
        src_vec,
        dst_vec,
        block_begin,
        block_end,
        lane,
        block_threads);

    const size_t tail_begin =
        total_vec * sizeof(uint4);

    const size_t tail_bytes =
        total_bytes - tail_begin;

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
