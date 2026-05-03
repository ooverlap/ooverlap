#pragma once

#include "comm/tma_variant_config.h"
#include "comm/window_task.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {
namespace comm {

template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxTasks>
cudaError_t launch_window_task_executor_sm90(
    WindowTaskExecutorPlan<MaxTasks> plan,
    int num_blocks,
    int threads,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    cudaStream_t stream) {
    using Variant = TmaPipelineVariant<ChunkBytes, StageDepth>;

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    if (threads <= 0 || threads > 1024 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }

    window_task_executor_kernel_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxTasks><<<
            num_blocks,
            threads,
            Variant::dynamic_shared_bytes,
            stream>>>(
                plan,
                local_ready_signal,
                peer_ready_signal,
                collective_epoch);

    return cudaGetLastError();
}

} // namespace comm
} // namespace ooverlap
