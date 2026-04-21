#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {

enum class ChunkOpKind : uint8_t {
    kInvalid = 0,
    kCopy = 1,
    kReduceAddNoFtzF16 = 2,
};

static constexpr int kChunkMaxTileSpans = 16;

struct ChunkTileSpan {
    const unsigned char* src = nullptr;
    size_t bytes = 0;
    size_t dst_offset_bytes = 0;

    uint64_t tile_id = 0;
    uint64_t queue_ticket = 0;

    uint32_t dim0 = 0;
    uint32_t dim1 = 0;
    uint32_t dim2 = 0;
};

struct Chunk {
    // Destination staging area / channel buffer base.
    unsigned char* dst = nullptr;

    // Total bytes represented by this chunk.
    size_t bytes = 0;

    // Metadata for routing / tracing.
    uint32_t queue_id = 0;
    int dst_rank = -1;

    // The first queue ticket included in this chunk.
    uint64_t span_ticket = 0;

    // Usually the first tile id in this chunk.
    uint64_t user_tag = 0;

    // Kept for compatibility with older pipeline helpers.
    int chunk_idx = 0;
    size_t span_offset_bytes = 0;

    ChunkOpKind op = ChunkOpKind::kInvalid;

    int num_tile_spans = 0;
    ChunkTileSpan tile_spans[kChunkMaxTileSpans];
};

__host__ __device__ __forceinline__ void chunk_tile_span_clear(
    ChunkTileSpan* span) {
    span->src = nullptr;
    span->bytes = 0;
    span->dst_offset_bytes = 0;
    span->tile_id = 0;
    span->queue_ticket = 0;
    span->dim0 = 0;
    span->dim1 = 0;
    span->dim2 = 0;
}

__host__ __device__ __forceinline__ bool chunk_tile_span_is_valid(
    const ChunkTileSpan* span) {
    return span != nullptr &&
           span->src != nullptr &&
           span->bytes > 0;
}

__host__ __device__ __forceinline__ void chunk_clear(
    Chunk* chunk) {
    chunk->dst = nullptr;
    chunk->bytes = 0;
    chunk->queue_id = 0;
    chunk->dst_rank = -1;
    chunk->span_ticket = 0;
    chunk->user_tag = 0;
    chunk->chunk_idx = 0;
    chunk->span_offset_bytes = 0;
    chunk->op = ChunkOpKind::kInvalid;
    chunk->num_tile_spans = 0;

    for (int i = 0; i < kChunkMaxTileSpans; ++i) {
        chunk_tile_span_clear(&chunk->tile_spans[i]);
    }
}

__host__ __device__ __forceinline__ bool chunk_is_valid(
    const Chunk* chunk) {
    if (chunk == nullptr ||
        chunk->dst == nullptr ||
        chunk->bytes == 0 ||
        chunk->op == ChunkOpKind::kInvalid ||
        chunk->num_tile_spans <= 0 ||
        chunk->num_tile_spans > kChunkMaxTileSpans) {
        return false;
    }

    size_t summed_bytes = 0;
    for (int i = 0; i < chunk->num_tile_spans; ++i) {
        if (!chunk_tile_span_is_valid(&chunk->tile_spans[i])) {
            return false;
        }
        summed_bytes += chunk->tile_spans[i].bytes;
    }

    return summed_bytes == chunk->bytes;
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
