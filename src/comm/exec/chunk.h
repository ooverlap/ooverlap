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

struct Chunk {
    // Contiguous source range for this chunk.
    const unsigned char* src = nullptr;

    // Contiguous destination range for this chunk.
    unsigned char* dst = nullptr;

    // Bytes in this chunk.
    size_t bytes = 0;

    // Generic metadata.
    uint32_t op_id = 0;
    int dst_rank = -1;
    uint64_t user_tag = 0;

    // 0-based chunk index inside the submitted range.
    int chunk_idx = 0;

    // Byte offset of this chunk inside the full submitted range.
    size_t range_offset_bytes = 0;

    ChunkOpKind op = ChunkOpKind::kInvalid;
};

__host__ __device__ __forceinline__ void chunk_clear(
    Chunk* chunk) {
    chunk->src = nullptr;
    chunk->dst = nullptr;
    chunk->bytes = 0;
    chunk->op_id = 0;
    chunk->dst_rank = -1;
    chunk->user_tag = 0;
    chunk->chunk_idx = 0;
    chunk->range_offset_bytes = 0;
    chunk->op = ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool chunk_is_valid(
    const Chunk* chunk) {
    return chunk != nullptr &&
           chunk->src != nullptr &&
           chunk->dst != nullptr &&
           chunk->bytes > 0 &&
           chunk->op != ChunkOpKind::kInvalid;
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
