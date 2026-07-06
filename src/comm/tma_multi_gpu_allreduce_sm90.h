#pragma once

#include "comm/launch_config.h"
#include "comm/plan/transfer_plan_distribution.h"
#include "ooverlap/comm.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {

/*
 * Naive rank-local multi-GPU allreduce.
 *
 * Inputs:
 *
 *   local_in:
 *     Local rank input pointer.
 *
 *   local_buf:
 *     Local rank output pointer.
 *
 *   peer_bufs:
 *     Pointers to every other rank's buffer, excluding this local rank.
 *     Length must be peer_count.
 *
 *   peer_count:
 *     Number of peer buffers. world_size must equal peer_count + 1.
 *
 *   rank:
 *     Local rank in [0, world_size).
 *
 *   world_size:
 *     Total number of ranks/devices.
 *
 * Algorithm:
 *
 *   For rank r:
 *
 *     1. Compute r's contiguous partition of the tensor.
 *     2. Reduce every peer's matching partition into local_buf.
 *     3. Copy the finalized local partition back to every peer buffer.
 *
 * Example for 4 GPUs:
 *
 *   Rank 0 owns quarter 0.
 *
 *     reduce GPU1 quarter0 -> GPU0 quarter0
 *     reduce GPU2 quarter0 -> GPU0 quarter0
 *     reduce GPU3 quarter0 -> GPU0 quarter0
 *
 *     copy GPU0 quarter0 -> GPU1 quarter0
 *     copy GPU0 quarter0 -> GPU2 quarter0
 *     copy GPU0 quarter0 -> GPU3 quarter0
 *
 * Every rank does the same for its own quarter.
 *
 * This is intentionally simple and not optimized. It is a correctness-first
 * generalization path before adding better collectives/plans.
 */
cudaError_t enqueue_tma_multi_gpu_allreduce_rank_sm90(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config = comm::default_launch_config());

cudaError_t enqueue_tma_multi_gpu_allreduce_rank_sm90_transfer_plan(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    const int* peer_ranks,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config,
    const comm::plan::AllreduceTransferPlan& transfer_plan);
} // namespace ooverlap
