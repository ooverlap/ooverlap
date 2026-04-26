#pragma once

#include "ooverlap/comm.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_seq_fastcopy_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch);

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_overlap_fastcopy_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch);

/*
 * Compatibility wrapper for existing benchmark code.
 *
 * This no longer implements pivot behavior:
 *   pivot_numerator <= 0 -> sequential TMA reduce then fast copy
 *   pivot_numerator >  0 -> overlapped TMA reduce + fast copy with signals
 *
 * pivot_denominator is ignored and kept only to avoid changing callers yet.
 */
cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_pivot_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int pivot_numerator,
    int pivot_denominator);

} // namespace ooverlap
