#pragma once

#include "comm/launch_config.h"
#include "comm/plan/transfer_plan_distribution.h"
#include "ooverlap/comm.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {

/*
 * Naive rank-local multi-GPU all-gather.
 *
 * Inputs:
 *
 *   local_in:
 *     Local rank input pointer over the full logical tensor.
 *
 *   local_buf:
 *     Local rank output pointer over the full logical tensor.
 *
 *     This first implementation uses this rank's natural partition inside the
 *     full logical tensor. It does not use compact local shards yet.
 *
 *   peer_bufs:
 *     Pointers to every other rank's full logical buffer, excluding this local
 *     rank. Length must be peer_count.
 *
 *   count:
 *     Full logical tensor element count per rank.
 *
 * Algorithm:
 *
 *   For rank r:
 *
 *     1. Compute r's contiguous partition of the full tensor.
 *     2. If local_in != local_buf, copy local input partition into local output
 *        partition.
 *     3. Copy this finalized local partition to every peer buffer.
 *
 * Example for 4 GPUs:
 *
 *   Rank 0 owns quarter 0.
 *
 *     copy GPU0 quarter0 -> GPU1 quarter0
 *     copy GPU0 quarter0 -> GPU2 quarter0
 *     copy GPU0 quarter0 -> GPU3 quarter0
 *
 *   No reduce phase.
 */
cudaError_t enqueue_tma_multi_gpu_all_gather_rank_sm90(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config = comm::default_launch_config());

cudaError_t enqueue_tma_multi_gpu_all_gather_rank_sm90_transfer_plan(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    const int* peer_ranks,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config,
    const comm::plan::AllGatherTransferPlan& transfer_plan);

} // namespace ooverlap
