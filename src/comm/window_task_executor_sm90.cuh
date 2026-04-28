#pragma once

#include "comm/tma_variant_config.h"
#include "comm/window_task.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {
namespace comm {

template <int MaxTasks>
struct WindowTaskExecutorPlan {
    static_assert(MaxTasks > 0, "MaxTasks must be > 0");

    int total_tasks = 0;
    int tasks_per_cta = 0;
    WindowTask tasks[MaxTasks];
};

template <int MaxTasks>
__host__ __forceinline__ void window_task_executor_plan_clear(
    WindowTaskExecutorPlan<MaxTasks>* plan) {
    if (plan == nullptr) {
        return;
    }

    plan->total_tasks = 0;
    plan->tasks_per_cta = 0;

    for (int i = 0; i < MaxTasks; ++i) {
        plan->tasks[i] = WindowTask{};
    }
}

template <int MaxTasks>
__host__ __forceinline__ bool window_task_executor_plan_set(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    int task_idx,
    const WindowTask& task) {
    if (plan == nullptr || task_idx < 0 || task_idx >= MaxTasks) {
        return false;
    }

    plan->tasks[task_idx] = task;
    return true;
}

/*
 * Stupid task executor kernel.
 *
 * This kernel intentionally does not know:
 *
 *   rank ownership
 *   CTA window ranges
 *   in-place vs out-of-place policy
 *   reduce/copy ordering
 *   local/peer pointer meaning
 *
 * The host builds a WindowTaskExecutorPlan. The kernel only waits for the peer
 * collective rendezvous and executes the static task stripe assigned to blockIdx.x.
 */
template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxTasks>
__global__ void window_task_executor_kernel_sm90(
    WindowTaskExecutorPlan<MaxTasks> plan,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch) {
    using Variant = TmaPipelineVariant<ChunkBytes, StageDepth>;

    window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[Variant::barrier_count];

    execute_window_task_stripe<
        Variant::stage_depth,
        Variant::stage_gap,
        Variant::chunk_bytes,
        ReduceApply>(
            plan.tasks,
            plan.total_tasks,
            plan.tasks_per_cta,
            static_cast<int>(blockIdx.x),
            shared_raw,
            barriers);
}

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
