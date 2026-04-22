#pragma once

#include "comm/exec/chunk_scheduler.h"
#include "comm/exec/pipeline_stage.h"

namespace ooverlap {
namespace comm {
namespace exec {

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
struct ChunkPipeline {
    PipelineStage stages[StageDepth];

    Scheduler scheduler{};
    LoadOp load_op{};
    ApplyOp apply_op{};

    int local_iter = 0;
    bool current_issued = false;
};

template <int StageDepth, size_t StageBytes, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_bind_stage_storage(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe,
    unsigned char* smem_base,
    sync::semaphore* barriers) {
    for (int i = 0; i < StageDepth; ++i) {
        pipeline_stage_bind<StageBytes>(
            &pipe->stages[i],
            smem_base,
            &barriers[i],
            i);
    }
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_init(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe,
    const Scheduler* scheduler) {
    pipe->scheduler = *scheduler;
    pipe->local_iter = 0;
    pipe->current_issued = false;
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ PipelineStage* chunk_pipeline_current_stage(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    return &pipe->stages[pipe->local_iter % StageDepth];
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ const PipelineStage* chunk_pipeline_current_stage(
    const ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    return &pipe->stages[pipe->local_iter % StageDepth];
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ bool chunk_pipeline_try_prime(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    if (!chunk_scheduler_try_prime_current(&pipe->scheduler)) {
        return false;
    }

    if (pipe->current_issued) {
        return true;
    }

    PipelineStage* current = chunk_pipeline_current_stage(pipe);
    pipeline_stage_set_chunk(current, chunk_scheduler_current(&pipe->scheduler));

    if (threadIdx.x == 0) {
        pipe->load_op.issue(current);
    }

    pipe->current_issued = true;
    return true;
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ bool chunk_pipeline_has_current(
    const ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    return chunk_scheduler_has_current(&pipe->scheduler);
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_wait_current_stage(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    if (threadIdx.x == 0) {
        pipe->load_op.wait_ready(chunk_pipeline_current_stage(pipe));
    }
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_issue_current_apply(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    if (threadIdx.x == 0) {
        pipe->apply_op.issue_bulk(chunk_pipeline_current_stage(pipe));
    }
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_finish_current_apply(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    pipe->apply_op.finish_tail(chunk_pipeline_current_stage(pipe));
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_wait_current_complete(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    if (threadIdx.x == 0) {
        pipe->apply_op.wait_complete(chunk_pipeline_current_stage(pipe));
    }
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_schedule_next_load(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    (void)pipe;
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_retire_current(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    chunk_scheduler_retire_current(&pipe->scheduler);
    ++pipe->local_iter;
    pipe->current_issued = false;
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
