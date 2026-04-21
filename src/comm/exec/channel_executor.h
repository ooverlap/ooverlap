#pragma once

#include "comm/channel_worker.h"
#include "comm/exec/chunk_pipeline.h"

namespace ooverlap {
namespace comm {
namespace exec {

template <int QueueCapacity, int StageDepth, typename LoadOp, typename ReduceOp>
struct ChannelExecutor {
    int owner_rank = -1;
    int src_rank = -1;
    int dst_rank = -1;

    transport::DeviceDispatchQueueHandle dispatch_queue{};
    transport::DeviceDirectReduceControlHandle direct_control{};

    ChunkScheduler<QueueCapacity> scheduler{};
    ChunkPipeline<StageDepth, ChunkScheduler<QueueCapacity>, LoadOp, ReduceOp> pipeline{};
};

template <int QueueCapacity, int StageDepth, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void channel_executor_init(
    ChannelExecutor<QueueCapacity, StageDepth, LoadOp, ReduceOp>* exec,
    const ::ooverlap::comm::ChannelWorker* worker) {
    exec->owner_rank = worker->owner_rank;
    exec->src_rank = worker->src_rank;
    exec->dst_rank = worker->dst_rank;
    exec->dispatch_queue = worker->dispatch_queue;
    exec->direct_control = worker->direct_control;

    chunk_scheduler_init(&exec->scheduler, &exec->dispatch_queue);
    chunk_pipeline_init(&exec->pipeline, &exec->scheduler);
}

template <int QueueCapacity, int StageDepth, size_t StageBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void channel_executor_bind_stage_storage(
    ChannelExecutor<QueueCapacity, StageDepth, LoadOp, ReduceOp>* exec,
    unsigned char* smem_base,
    sync::semaphore* barriers) {
    chunk_pipeline_bind_stage_storage<StageDepth, StageBytes>(
        &exec->pipeline,
        smem_base,
        barriers);
}

template <int QueueCapacity, int StageDepth, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ bool channel_executor_try_prime(
    ChannelExecutor<QueueCapacity, StageDepth, LoadOp, ReduceOp>* exec) {
    return chunk_pipeline_try_prime(&exec->pipeline);
}

template <int QueueCapacity, int StageDepth, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void channel_executor_wait_for_work(
    ChannelExecutor<QueueCapacity, StageDepth, LoadOp, ReduceOp>* exec) {
    chunk_pipeline_wait_for_work(&exec->pipeline);
}

template <int QueueCapacity, int StageDepth, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ bool channel_executor_has_current(
    const ChannelExecutor<QueueCapacity, StageDepth, LoadOp, ReduceOp>* exec) {
    return chunk_pipeline_has_current(&exec->pipeline);
}

template <int QueueCapacity, int StageDepth, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void channel_executor_advance(
    ChannelExecutor<QueueCapacity, StageDepth, LoadOp, ReduceOp>* exec) {
    chunk_pipeline_advance(&exec->pipeline);
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
