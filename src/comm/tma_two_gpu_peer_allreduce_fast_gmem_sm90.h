#pragma once

#include "comm/launch_config.h"
#include "ooverlap/comm.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {

/*
 * Sequential fast-gmem variant:
 *
 *   1. TMA reduce local_in into peer_buf.
 *   2. Fast global-memory copy peer_buf back into local_buf.
 */
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
    int collective_epoch,
    comm::LaunchConfig launch_config);

/*
 * Overlapped fast-gmem variant:
 *
 *   CTA role 0: TMA reduce local_in into peer_buf and publish progress.
 *   CTA role 1: Fast global-memory copy completed peer_buf windows into local_buf.
 */
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
    int collective_epoch,
    comm::LaunchConfig launch_config);

} // namespace ooverlap
