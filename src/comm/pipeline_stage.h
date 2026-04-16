#pragma once

#include "ooverlap/sync/sync.cuh"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {

struct PipelineStage {
    unsigned char* smem = nullptr;
    sync::semaphore* load_barrier = nullptr;

    int chunk_idx = -1;
    size_t chunk_offset_bytes = 0;
    size_t chunk_bytes = 0;
};

__host__ __device__ __forceinline__ int pipeline_compute_num_chunks(
    size_t total_bytes,
    size_t chunk_bytes) {
    return static_cast<int>((total_bytes + chunk_bytes - 1) / chunk_bytes);
}

__host__ __device__ __forceinline__ size_t pipeline_compute_chunk_offset_bytes(
    int chunk_idx,
    size_t chunk_bytes) {
    return static_cast<size_t>(chunk_idx) * chunk_bytes;
}

__host__ __device__ __forceinline__ size_t pipeline_compute_chunk_bytes(
    int chunk_idx,
    size_t total_bytes,
    size_t chunk_bytes) {
    return utils::pipeline_min_sz(
        chunk_bytes,
        total_bytes - pipeline_compute_chunk_offset_bytes(chunk_idx, chunk_bytes));
}

__host__ __device__ __forceinline__ size_t pipeline_stage_bulk_bytes(
    const PipelineStage* stage) {
    return stage->chunk_bytes & ~static_cast<size_t>(0xF);
}

__host__ __device__ __forceinline__ size_t pipeline_stage_tail_bytes(
    const PipelineStage* stage) {
    return stage->chunk_bytes - pipeline_stage_bulk_bytes(stage);
}

__device__ __forceinline__ void pipeline_stage_reset(
    PipelineStage* stage) {
    stage->chunk_idx = -1;
    stage->chunk_offset_bytes = 0;
    stage->chunk_bytes = 0;
}

template <size_t ChunkBytes>
__device__ __forceinline__ void pipeline_stage_bind(
    PipelineStage* stage,
    unsigned char* smem_base,
    sync::semaphore* barrier,
    int stage_idx) {
    stage->smem = smem_base + static_cast<size_t>(stage_idx) * ChunkBytes;
    stage->load_barrier = barrier;
    pipeline_stage_reset(stage);
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ void pipeline_stage_set_chunk(
    PipelineStage* stage,
    int chunk_idx,
    size_t total_bytes) {
    stage->chunk_idx = chunk_idx;
    stage->chunk_offset_bytes =
        pipeline_compute_chunk_offset_bytes(chunk_idx, ChunkBytes);
    stage->chunk_bytes =
        pipeline_compute_chunk_bytes(chunk_idx, total_bytes, ChunkBytes);
}

} // namespace comm
} // namespace ooverlap
