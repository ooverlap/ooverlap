#pragma once

#include "comm/chunk_scheduler.h"
#include "comm/pipeline_stage.h"
#include "comm/pipeline_load.h"
#include "comm/pipeline_reduce.h"

namespace ooverlap {
namespace comm {

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
struct ChunkPipeline {
    PipelineStage stages[StageDepth];

    Scheduler scheduler{};
    LoadOp load_op{};
    ReduceOp reduce_op{};

    int local_iter = 0;
};

template <int StageDepth, size_t StageBytes, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void chunk_pipeline_bind_stage_storage(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe,
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

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void chunk_pipeline_init(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe,
    const Scheduler* scheduler) {
    pipe->scheduler = *scheduler;
    pipe->local_iter = 0;
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ bool chunk_pipeline_try_prime(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    if (!chunk_scheduler_try_prime_current(&pipe->scheduler)) {
        return false;
    }

    PipelineStage* first = &pipe->stages[0];
    pipeline_stage_set_chunk(first, chunk_scheduler_current(&pipe->scheduler));

    if (threadIdx.x == 0) {
        pipe->load_op.issue(first);
    }

    return true;
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void chunk_pipeline_wait_for_work(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    while (!chunk_pipeline_try_prime(pipe)) {
    }
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ bool chunk_pipeline_has_current(
    const ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    return chunk_scheduler_has_current(&pipe->scheduler);
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ PipelineStage* chunk_pipeline_current_stage(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    return &pipe->stages[pipe->local_iter % StageDepth];
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ const PipelineStage* chunk_pipeline_current_stage(
    const ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    return &pipe->stages[pipe->local_iter % StageDepth];
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void chunk_pipeline_wait_current_stage(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    if (threadIdx.x == 0) {
        pipe->load_op.wait_ready(chunk_pipeline_current_stage(pipe));
    }
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void chunk_pipeline_issue_current_reduce(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    if (threadIdx.x == 0) {
        pipe->reduce_op.issue_bulk(chunk_pipeline_current_stage(pipe));
    }
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void chunk_pipeline_schedule_next_load(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    Chunk next_chunk{};
    if (!chunk_scheduler_peek_next(&pipe->scheduler, &next_chunk)) {
        return;
    }

    if (threadIdx.x == 0) {
        // TODO(keyvand): revisit deeper prefetch / multi-stage priming.
        // The simple one-step lookahead is currently faster for this kernel.
        // If we retry deeper lookahead later, we should model stage retirement
        // explicitly instead of reusing the same stage ring heuristically.
        if ((pipe->local_iter + 1) >= StageDepth) {
            pipe->reduce_op.template wait_before_stage_reuse<StageDepth>();
        }

        PipelineStage* next_stage =
            &pipe->stages[(pipe->local_iter + 1) % StageDepth];

        pipeline_stage_set_chunk(next_stage, &next_chunk);
        pipe->load_op.issue(next_stage);
    }
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void chunk_pipeline_finish_current_tail(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    pipe->reduce_op.finish_tail(chunk_pipeline_current_stage(pipe));
}

template <int StageDepth, typename Scheduler, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void chunk_pipeline_advance(
    ChunkPipeline<StageDepth, Scheduler, LoadOp, ReduceOp>* pipe) {
    chunk_scheduler_advance(&pipe->scheduler);
    ++pipe->local_iter;

    if (chunk_scheduler_has_current(&pipe->scheduler)) {
        PipelineStage* next_current = chunk_pipeline_current_stage(pipe);
        pipeline_stage_set_chunk(next_current, chunk_scheduler_current(&pipe->scheduler));
    }
}

} // namespace comm
} // namespace ooverlap
