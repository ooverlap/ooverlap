#pragma once

#include "comm/kernels/window_task_executor.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"

#include <cuda_runtime.h>

namespace ooverlap {
namespace comm {

template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxTasks>
cudaError_t launch_window_task_executor_sm90(
    comm::plan::WindowTaskExecutorPlan<MaxTasks> plan,
    int num_blocks,
    int threads,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    cudaStream_t stream) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    if (threads <= 0 || threads > 1024 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }

    comm::kernels::window_task_executor_kernel_sm90<
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
