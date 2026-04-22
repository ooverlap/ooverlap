#pragma once

#include "comm/exec/chunk.h"
#include "ooverlap/sync/sync.cuh"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {

struct PipelineStage {
    unsigned char* smem = nullptr;
    sync::semaphore* load_barrier = nullptr;
    Chunk chunk{};
    uint32_t step = 0;
};

__host__ __device__ __forceinline__ size_t pipeline_stage_bulk_bytes(
    const PipelineStage* stage) {
    return stage->chunk.bytes & ~static_cast<size_t>(0xF);
}

__host__ __device__ __forceinline__ size_t pipeline_stage_tail_bytes(
    const PipelineStage* stage) {
    return stage->chunk.bytes - pipeline_stage_bulk_bytes(stage);
}

__host__ __device__ __forceinline__ unsigned char* pipeline_stage_smem_ptr(
    const PipelineStage* stage,
    size_t offset_bytes = 0) {
    return stage->smem + offset_bytes;
}

__device__ __forceinline__ void pipeline_stage_reset(
    PipelineStage* stage) {
    chunk_clear(&stage->chunk);
    stage->step = 0;
}

template <size_t StageBytes>
__device__ __forceinline__ void pipeline_stage_bind(
    PipelineStage* stage,
    unsigned char* smem_base,
    sync::semaphore* barrier,
    int stage_idx) {
    stage->smem = smem_base + static_cast<size_t>(stage_idx) * StageBytes;
    stage->load_barrier = barrier;
    pipeline_stage_reset(stage);
}

__host__ __device__ __forceinline__ void pipeline_stage_set_chunk(
    PipelineStage* stage,
    const Chunk* chunk,
    uint32_t step) {
    stage->chunk = *chunk;
    stage->step = step;
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
