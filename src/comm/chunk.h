#pragma once

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {

enum class ChunkOpKind : uint8_t {
    kInvalid = 0,
    kCopy = 1,
    kReduceAddNoFtzF16 = 2,
};

struct Chunk {
    const unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    size_t bytes = 0;

    int registration_idx = -1;
    int chunk_idx = -1;

    ChunkOpKind op = ChunkOpKind::kInvalid;
};

struct ChunkRegistration {
    const unsigned char* src_base = nullptr;
    unsigned char* dst_base = nullptr;

    size_t total_bytes = 0;
    size_t chunk_bytes = 0;

    ChunkOpKind op = ChunkOpKind::kInvalid;
};

__host__ __device__ __forceinline__ void chunk_clear(Chunk* chunk) {
    chunk->src = nullptr;
    chunk->dst = nullptr;
    chunk->bytes = 0;
    chunk->registration_idx = -1;
    chunk->chunk_idx = -1;
    chunk->op = ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool chunk_is_valid(const Chunk* chunk) {
    return chunk != nullptr &&
           chunk->bytes > 0 &&
           chunk->registration_idx >= 0 &&
           chunk->chunk_idx >= 0;
}

__host__ __device__ __forceinline__ int chunk_registration_num_chunks(
    const ChunkRegistration* reg) {
    if (reg == nullptr || reg->chunk_bytes == 0 || reg->total_bytes == 0) {
        return 0;
    }
    return static_cast<int>((reg->total_bytes + reg->chunk_bytes - 1) / reg->chunk_bytes);
}

__host__ __device__ __forceinline__ bool chunk_registration_resolve_chunk(
    const ChunkRegistration* reg,
    int registration_idx,
    int chunk_idx,
    Chunk* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_clear(out);

    if (reg == nullptr) {
        return false;
    }
    if (reg->chunk_bytes == 0 || reg->total_bytes == 0) {
        return false;
    }

    const int num_chunks = chunk_registration_num_chunks(reg);
    if (chunk_idx < 0 || chunk_idx >= num_chunks) {
        return false;
    }

    const size_t offset = static_cast<size_t>(chunk_idx) * reg->chunk_bytes;
    const size_t bytes = chunk_min_sz(reg->chunk_bytes, reg->total_bytes - offset);

    out->src = (reg->src_base == nullptr) ? nullptr : (reg->src_base + offset);
    out->dst = (reg->dst_base == nullptr) ? nullptr : (reg->dst_base + offset);
    out->bytes = bytes;
    out->registration_idx = registration_idx;
    out->chunk_idx = chunk_idx;
    out->op = reg->op;
    return true;
}

} // namespace comm
} // namespace ooverlap
