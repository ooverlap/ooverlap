#pragma once

#include "comm/launch_config.h"
#include "ooverlap/comm.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {

/*
 * Configure the default SM90 multi-GPU reduce-scatter kernel variant once.
 *
 * This exists for warmup / code that wants CUDA function attributes configured
 * before timing.
 */
void tma_multi_gpu_reduce_scatter_configure_kernel_once(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device);

/*
 * Naive rank-local multi-GPU reduce-scatter.
 *
 * Inputs:
 *
 *   local_in:
 *     Local rank input pointer over the full logical tensor.
 *
 *   local_buf:
 *     Local rank output pointer over the full logical tensor.
 *
 *     This first implementation writes the result into this rank's natural
 *     partition of local_buf, not compactly at local_buf[0].
 *
 *   peer_bufs:
 *     Pointers to every other rank's full logical input/output buffer,
 *     excluding this local rank. Length must be peer_count.
 *
 *   count:
 *     Full logical tensor element count per rank.
 *
 * Algorithm:
 *
 *   For rank r:
 *
 *     1. Compute r's contiguous partition of the full tensor.
 *     2. Initialize local output partition from local input partition if
 *        local_in != local_buf.
 *     3. Reduce every peer's matching partition into local_buf's partition.
 *     4. Stop. Unlike allreduce, there is no copy/broadcast phase.
 *
 * Example for 4 GPUs:
 *
 *   Rank 0 owns quarter 0.
 *
 *     reduce GPU1 quarter0 -> GPU0 quarter0
 *     reduce GPU2 quarter0 -> GPU0 quarter0
 *     reduce GPU3 quarter0 -> GPU0 quarter0
 *
 *   No copy back to GPU1/GPU2/GPU3.
 */
cudaError_t enqueue_tma_multi_gpu_reduce_scatter_rank_sm90(
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

} // namespace ooverlap
