#pragma once

#include "ooverlap/mem/fast_copy.cuh"
#include "comm/kernels/fast_gmem_add.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace kernels {
namespace fast_copy {

template <typename VecT, int Unroll = 8>
__global__ void gmem_copy_coalesced_kernel(
    const void* __restrict__ src,
    void* __restrict__ dst,
    size_t total_bytes) {
    const unsigned char* __restrict__ src_u8 =
        reinterpret_cast<const unsigned char*>(src);

    unsigned char* __restrict__ dst_u8 =
        reinterpret_cast<unsigned char*>(dst);

    const size_t total_vec =
        total_bytes / sizeof(VecT);

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

    const VecT* __restrict__ src_vec =
        reinterpret_cast<const VecT*>(src);

    VecT* __restrict__ dst_vec =
        reinterpret_cast<VecT*>(dst);

    copy_vec_span<VecT, Unroll>(
        src_vec,
        dst_vec,
        block_begin,
        block_end,
        lane,
        block_threads);

    const size_t tail_begin =
        total_vec * sizeof(VecT);

    if (blockIdx.x == 0) {
        for (size_t i = tail_begin + lane;
             i < total_bytes;
             i += block_threads) {
            dst_u8[i] = src_u8[i];
        }
    }
}

} // namespace fast_copy
} // namespace kernels
} // namespace comm
} // namespace ooverlap
