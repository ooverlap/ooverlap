#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comm/utils.h"

namespace ooverlap {
namespace comm {
namespace exec {

enum class ChunkOpKind : uint8_t {
    kInvalid = 0,
    kCopy = 1,
    kReduceAddNoFtzF16 = 2,
};

struct WorkSpan {
    const unsigned char* src_base = nullptr;
    unsigned char* dst_base = nullptr;
    size_t total_bytes = 0;
    ChunkOpKind op = ChunkOpKind::kInvalid;
    uint64_t user_tag = 0;
};

struct Chunk {
    const unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    size_t bytes = 0;

    uint64_t span_ticket = 0;
    uint64_t user_tag = 0;

    int chunk_idx = -1;
    size_t span_offset_bytes = 0;

    ChunkOpKind op = ChunkOpKind::kInvalid;
};

__host__ __device__ __forceinline__ void work_span_clear(WorkSpan* span) {
    span->src_base = nullptr;
    span->dst_base = nullptr;
    span->total_bytes = 0;
    span->op = ChunkOpKind::kInvalid;
    span->user_tag = 0;
}

__host__ __device__ __forceinline__ bool work_span_is_valid(const WorkSpan* span) {
    return span != nullptr &&
           span->src_base != nullptr &&
           span->dst_base != nullptr &&
           span->total_bytes > 0 &&
           span->op != ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ void chunk_clear(Chunk* chunk) {
    chunk->src = nullptr;
    chunk->dst = nullptr;
    chunk->bytes = 0;
    chunk->span_ticket = 0;
    chunk->user_tag = 0;
    chunk->chunk_idx = -1;
    chunk->span_offset_bytes = 0;
    chunk->op = ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool chunk_is_valid(const Chunk* chunk) {
    return chunk != nullptr &&
           chunk->src != nullptr &&
           chunk->dst != nullptr &&
           chunk->bytes > 0 &&
           chunk->chunk_idx >= 0 &&
           chunk->op != ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool chunk_make_from_span(
    const WorkSpan* span,
    uint64_t span_ticket,
    size_t span_offset_bytes,
    int chunk_idx,
    size_t chunk_bytes,
    Chunk* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_clear(out);

    if (!work_span_is_valid(span)) {
        return false;
    }
    if (chunk_bytes == 0) {
        return false;
    }
    if (span_offset_bytes >= span->total_bytes) {
        return false;
    }

    const size_t bytes =
        utils::min_sz(chunk_bytes, span->total_bytes - span_offset_bytes);

    out->src = span->src_base + span_offset_bytes;
    out->dst = span->dst_base + span_offset_bytes;
    out->bytes = bytes;
    out->span_ticket = span_ticket;
    out->user_tag = span->user_tag;
    out->chunk_idx = chunk_idx;
    out->span_offset_bytes = span_offset_bytes;
    out->op = span->op;
    return true;
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
