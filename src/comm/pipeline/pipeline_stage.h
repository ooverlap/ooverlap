#pragma once

#include "ooverlap/sync/sync.cuh"

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace pipeline {

struct PipelineChunk {
    const unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    size_t bytes = 0;
};

struct PipelineStage {
    PipelineChunk chunk{};
    unsigned char* smem = nullptr;
    sync::semaphore* load_barrier = nullptr;
};

__host__ __device__ __forceinline__ size_t pipeline_stage_bulk_bytes(
    const PipelineStage* stage) {
    return stage->chunk.bytes & ~static_cast<size_t>(0xF);
}

__host__ __device__ __forceinline__ size_t pipeline_stage_tail_bytes(
    const PipelineStage* stage) {
    return stage->chunk.bytes - pipeline_stage_bulk_bytes(stage);
}

__host__ __device__ __forceinline__ PipelineChunk make_pipeline_chunk(
    const void* src,
    void* dst,
    size_t bytes) {
    PipelineChunk chunk{};
    chunk.src = reinterpret_cast<const unsigned char*>(src);
    chunk.dst = reinterpret_cast<unsigned char*>(dst);
    chunk.bytes = bytes;
    return chunk;
}

__host__ __device__ __forceinline__ PipelineStage make_pipeline_stage(
    PipelineChunk chunk,
    void* smem,
    sync::semaphore* load_barrier) {
    PipelineStage stage{};
    stage.chunk = chunk;
    stage.smem = reinterpret_cast<unsigned char*>(smem);
    stage.load_barrier = load_barrier;
    return stage;
}

} // namespace pipeline
} // namespace comm
} // namespace ooverlap
