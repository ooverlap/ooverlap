#pragma once

#include "comm/pipeline_stage.h"
#include "comm/pipeline_load.h"
#include "comm/pipeline_reduce.h"

#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
struct PersistentPipeline {
    PipelineStage stages[StageDepth];

    LoadOp load_op{};
    ReduceOp reduce_op{};

    const unsigned char* src_bytes = nullptr;
    unsigned char* dst_bytes = nullptr;
    size_t total_bytes = 0;

    int num_chunks = 0;
    int chunk_stride = 0;

    int local_iter = 0;
    int cur_chunk = -1;
    int next_chunk_to_load = -1;
};

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void persistent_pipeline_bind_stage_storage(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe,
    unsigned char* smem_base,
    sync::semaphore* barriers) {
    for (int i = 0; i < StageDepth; ++i) {
        persistent_stage_bind<ChunkBytes>(
            &pipe->stages[i],
            smem_base,
            &barriers[i],
            i);
    }
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void persistent_pipeline_init(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe,
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    size_t total_bytes,
    int start_chunk,
    int chunk_stride,
    int num_chunks) {
    pipe->src_bytes = src_bytes;
    pipe->dst_bytes = dst_bytes;
    pipe->total_bytes = total_bytes;
    pipe->num_chunks = num_chunks;
    pipe->chunk_stride = chunk_stride;
    pipe->local_iter = 0;
    pipe->cur_chunk = start_chunk;
    pipe->next_chunk_to_load = start_chunk + chunk_stride;
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ bool persistent_pipeline_active(
    const PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    return pipe->cur_chunk < pipe->num_chunks;
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ PipelineStage* persistent_pipeline_current_stage(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    return &pipe->stages[pipe->local_iter % StageDepth];
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ const PipelineStage* persistent_pipeline_current_stage(
    const PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    return &pipe->stages[pipe->local_iter % StageDepth];
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void persistent_pipeline_prime(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    if (!persistent_pipeline_active(pipe)) {
        return;
    }

    PipelineStage* first = &pipe->stages[0];
    persistent_stage_set_chunk<ChunkBytes>(
        first,
        pipe->cur_chunk,
        pipe->total_bytes);

    if (threadIdx.x == 0) {
        pipe->load_op.issue(first, pipe->src_bytes);
    }
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void persistent_pipeline_wait_current_stage(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    if (threadIdx.x == 0) {
        pipe->load_op.wait_ready(persistent_pipeline_current_stage(pipe));
    }
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void persistent_pipeline_issue_current_reduce(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    if (threadIdx.x == 0) {
        pipe->reduce_op.issue_bulk(
            persistent_pipeline_current_stage(pipe),
            pipe->dst_bytes);
    }
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void persistent_pipeline_schedule_next_load(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    if (pipe->next_chunk_to_load >= pipe->num_chunks) {
        return;
    }

    if (threadIdx.x == 0) {
        if ((pipe->local_iter + 1) >= StageDepth) {
            pipe->reduce_op.template wait_before_stage_reuse<StageDepth>();
        }

        PipelineStage* next_stage =
            &pipe->stages[(pipe->local_iter + 1) % StageDepth];

        persistent_stage_set_chunk<ChunkBytes>(
            next_stage,
            pipe->next_chunk_to_load,
            pipe->total_bytes);

        pipe->load_op.issue(next_stage, pipe->src_bytes);
    }
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void persistent_pipeline_finish_current_tail(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    pipe->reduce_op.finish_tail(
        persistent_pipeline_current_stage(pipe),
        reinterpret_cast<half*>(pipe->dst_bytes));
}

template <int StageDepth, size_t ChunkBytes, typename LoadOp, typename ReduceOp>
__device__ __forceinline__ void persistent_pipeline_advance(
    PersistentPipeline<StageDepth, ChunkBytes, LoadOp, ReduceOp>* pipe) {
    pipe->cur_chunk = pipe->next_chunk_to_load;
    pipe->next_chunk_to_load += pipe->chunk_stride;
    ++pipe->local_iter;
}

} // namespace comm
} // namespace ooverlap
