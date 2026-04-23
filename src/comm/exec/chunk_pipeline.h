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

    uint32_t retire_iter = 0;
    uint32_t issue_iter = 0;
    uint32_t issued_count = 0;
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
    pipe->retire_iter = 0;
    pipe->issue_iter = 0;
    pipe->issued_count = 0;
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ PipelineStage* chunk_pipeline_current_stage(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    return &pipe->stages[pipe->retire_iter % StageDepth];
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ const PipelineStage* chunk_pipeline_current_stage(
    const ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    return &pipe->stages[pipe->retire_iter % StageDepth];
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ PipelineStage* chunk_pipeline_next_issue_stage(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    return &pipe->stages[pipe->issue_iter % StageDepth];
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ bool chunk_pipeline_try_prime(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe,
    uint32_t max_issued = static_cast<uint32_t>(StageDepth)) {
    if (max_issued == 0u) {
        return pipe->issued_count > 0u;
    }

    if (max_issued > static_cast<uint32_t>(StageDepth)) {
        max_issued = static_cast<uint32_t>(StageDepth);
    }

    while (pipe->issued_count < max_issued) {
        chunk_scheduler_refill_ready_cache(&pipe->scheduler);

        if (!chunk_scheduler_try_prime_current(&pipe->scheduler)) {
            break;
        }

        PipelineStage* next = chunk_pipeline_next_issue_stage(pipe);
        pipeline_stage_set_chunk(
            next,
            chunk_scheduler_current(&pipe->scheduler),
            chunk_scheduler_active_step(&pipe->scheduler));

        pipe->load_op.issue(next);

        // The scheduler now consumes deterministic static work, not a queue.
        // Once the chunk is handed to a pipeline stage, the scheduler advances
        // only by cursor state; retirement publishes monotonic progress.
        chunk_scheduler_handoff_current(&pipe->scheduler);

        ++pipe->issue_iter;
        ++pipe->issued_count;
    }

    return pipe->issued_count > 0u;
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ bool chunk_pipeline_has_current(
    const ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    return pipe->issued_count > 0 &&
           chunk_is_valid(&chunk_pipeline_current_stage(pipe)->chunk);
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
    chunk_pipeline_try_prime(pipe);
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ApplyOp>
__device__ __forceinline__ void chunk_pipeline_retire_current(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ApplyOp>* pipe) {
    PipelineStage* current = chunk_pipeline_current_stage(pipe);

    chunk_scheduler_retire_stage(
        &pipe->scheduler,
        static_cast<uint32_t>(current->chunk.chunk_idx),
        current->step);

    pipeline_stage_reset(current);
    ++pipe->retire_iter;
    --pipe->issued_count;
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
