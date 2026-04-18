#pragma once

#include "comm/chunk.h"

namespace ooverlap {
namespace comm {

struct RangeChunkScheduler {
    const unsigned char* src_base = nullptr;
    unsigned char* dst_base = nullptr;
    size_t total_bytes = 0;
    size_t chunk_bytes = 0;

    int start_chunk = 0;
    int chunk_stride = 1;
    int current_chunk_idx = -1;

    Chunk current{};
};

__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    RangeChunkScheduler* sched) {
    sched->src_base = nullptr;
    sched->dst_base = nullptr;
    sched->total_bytes = 0;
    sched->chunk_bytes = 0;
    sched->start_chunk = 0;
    sched->chunk_stride = 1;
    sched->current_chunk_idx = -1;
    chunk_clear(&sched->current);
}

__host__ __device__ __forceinline__ void chunk_scheduler_init(
    RangeChunkScheduler* sched,
    const unsigned char* src_base,
    unsigned char* dst_base,
    size_t total_bytes,
    size_t chunk_bytes,
    int start_chunk,
    int chunk_stride) {
    sched->src_base = src_base;
    sched->dst_base = dst_base;
    sched->total_bytes = total_bytes;
    sched->chunk_bytes = chunk_bytes;
    sched->start_chunk = start_chunk;
    sched->chunk_stride = chunk_stride;
    sched->current_chunk_idx = -1;
    chunk_clear(&sched->current);
}

__host__ __device__ __forceinline__ int chunk_scheduler_num_chunks(
    const RangeChunkScheduler* sched) {
    if (sched == nullptr || sched->chunk_bytes == 0 || sched->total_bytes == 0) {
        return 0;
    }
    return static_cast<int>((sched->total_bytes + sched->chunk_bytes - 1) / sched->chunk_bytes);
}

__host__ __device__ __forceinline__ bool chunk_scheduler_make_chunk(
    const RangeChunkScheduler* sched,
    int chunk_idx,
    Chunk* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_clear(out);

    if (sched == nullptr ||
        sched->src_base == nullptr ||
        sched->dst_base == nullptr ||
        sched->chunk_bytes == 0 ||
        sched->total_bytes == 0) {
        return false;
    }

    const int num_chunks = chunk_scheduler_num_chunks(sched);
    if (chunk_idx < 0 || chunk_idx >= num_chunks) {
        return false;
    }

    const size_t offset = static_cast<size_t>(chunk_idx) * sched->chunk_bytes;
    const size_t bytes = utils::min_sz(sched->chunk_bytes, sched->total_bytes - offset);

    out->src = sched->src_base + offset;
    out->dst = sched->dst_base + offset;
    out->bytes = bytes;
    out->span_ticket = 0;
    out->user_tag = static_cast<uint64_t>(chunk_idx);
    out->chunk_idx = chunk_idx;
    out->span_offset_bytes = offset;
    out->op = ChunkOpKind::kReduceAddNoFtzF16;
    return true;
}

__host__ __device__ __forceinline__ bool chunk_scheduler_try_prime_current(
    RangeChunkScheduler* sched) {
    if (chunk_is_valid(&sched->current)) {
        return true;
    }

    sched->current_chunk_idx = sched->start_chunk;
    return chunk_scheduler_make_chunk(sched, sched->current_chunk_idx, &sched->current);
}

__host__ __device__ __forceinline__ bool chunk_scheduler_has_current(
    const RangeChunkScheduler* sched) {
    return chunk_is_valid(&sched->current);
}

__host__ __device__ __forceinline__ const Chunk* chunk_scheduler_current(
    const RangeChunkScheduler* sched) {
    return &sched->current;
}

__host__ __device__ __forceinline__ bool chunk_scheduler_peek_next(
    const RangeChunkScheduler* sched,
    Chunk* out) {
    if (sched == nullptr) {
        if (out != nullptr) {
            chunk_clear(out);
        }
        return false;
    }
    if (!chunk_is_valid(&sched->current)) {
        if (out != nullptr) {
            chunk_clear(out);
        }
        return false;
    }

    const int next_chunk_idx = sched->current_chunk_idx + sched->chunk_stride;
    return chunk_scheduler_make_chunk(sched, next_chunk_idx, out);
}

__host__ __device__ __forceinline__ void chunk_scheduler_advance(
    RangeChunkScheduler* sched) {
    if (sched == nullptr || !chunk_is_valid(&sched->current)) {
        chunk_clear(&sched->current);
        sched->current_chunk_idx = -1;
        return;
    }

    sched->current_chunk_idx += sched->chunk_stride;
    if (!chunk_scheduler_make_chunk(sched, sched->current_chunk_idx, &sched->current)) {
        chunk_clear(&sched->current);
        sched->current_chunk_idx = -1;
    }
}

} // namespace comm
} // namespace ooverlap
