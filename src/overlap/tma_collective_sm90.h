#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace ooverlap {

// Simple local accumulation primitive used above bulk-TMA transport.
cudaError_t enqueue_fp16_add_inplace_sm90(
    half* dst,
    const half* src,
    size_t numel,
    cudaStream_t stream);

// 2-GPU same-process all-reduce baseline:
// each GPU bulk-TMA sends its full buffer to the peer inbox,
// then each GPU locally accumulates inbox into its local tensor.
cudaError_t enqueue_two_gpu_all_reduce_tma_sm90(
    half* rank0_buf,
    half* rank1_buf,
    half* rank0_inbox,
    half* rank1_inbox,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1);

// 2-GPU same-process all-gather baseline:
// each GPU owns one shard; both full outputs are assembled by
// local placement + bulk-TMA send of the peer shard.
cudaError_t enqueue_two_gpu_all_gather_tma_sm90(
    const half* rank0_shard,
    const half* rank1_shard,
    half* rank0_full_out,
    half* rank1_full_out,
    size_t shard_numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1);

// End-to-end smoke tests using VMM peer buffers + bulk TMA transport.
bool tma_two_gpu_all_reduce_smoke_test(
    int64_t numel,
    int dev0 = 0,
    int dev1 = 1);

bool tma_two_gpu_all_gather_smoke_test(
    int64_t shard_numel,
    int dev0 = 0,
    int dev1 = 1);

} // namespace ooverlap
