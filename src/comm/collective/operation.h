#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comm/exec/chunk.h"

namespace ooverlap {
namespace comm {
namespace collective {

enum : uint32_t {
    kOperationFlagEnabled = 1u << 0,
};

enum : uint32_t {
    kChunkStateFlagSeeded = 1u << 0,
    kChunkStateFlagActive = 1u << 1,
    kChunkStateFlagDone = 1u << 2,
};

// Ring / range operation descriptor.
//
// Important semantic choice:
// - local persistent kernel owns local partial_ptr and local dst_ptr
// - peers should NOT remotely reduce directly into partial_ptr/dst_ptr
// - future transport should use inbox/outbox mailboxes instead
struct OperationDesc {
    uint32_t op_id = 0;
    uint32_t epoch = 0;
    uint32_t flags = 0;
    uint32_t reserved0 = 0;

    int rank = -1;
    int world_size = 0;
    int prev_rank = -1;
    int next_rank = -1;

    size_t total_bytes = 0;
    size_t chunk_bytes = 0;
    uint32_t num_chunks = 0;
    uint32_t expected_contributions = 0;

    exec::ChunkOpKind op = exec::ChunkOpKind::kInvalid;
    uint32_t reserved1 = 0;
    uint64_t user_tag = 0;

    // Local source range for this rank.
    uint64_t src_ptr = 0;

    // Final destination owned by this rank.
    uint64_t dst_ptr = 0;

    // Local accumulator / partial storage for all chunks.
    // Must be at least total_bytes bytes.
    uint64_t partial_ptr = 0;
    size_t partial_bytes = 0;

    // Reserved for future transport/mailbox phase.
    uint64_t inbox_ptr = 0;
    uint64_t outbox_ptr = 0;
};

struct ChunkState {
    uint32_t op_id = 0;
    uint32_t epoch = 0;
    uint32_t chunk_idx = 0;
    uint32_t flags = 0;

    uint32_t contributions_seen = 0;
    uint32_t expected_contributions = 0;
    uint32_t step = 0;
    uint32_t send_count = 0;

    size_t offset_bytes = 0;
    size_t bytes = 0;
};

struct ChunkStateTable {
    ChunkState* records = nullptr;  // device pointer
    uint32_t capacity = 0;
    int device = -1;
};

__host__ __device__ __forceinline__ uint32_t operation_desc_compute_num_chunks(
    size_t total_bytes,
    size_t chunk_bytes) {
    if (total_bytes == 0 || chunk_bytes == 0) {
        return 0;
    }
    return static_cast<uint32_t>(
        (total_bytes + chunk_bytes - 1) / chunk_bytes);
}

__host__ __device__ __forceinline__ void operation_desc_clear(
    OperationDesc* op) {
    op->op_id = 0;
    op->epoch = 0;
    op->flags = 0;
    op->reserved0 = 0;
    op->rank = -1;
    op->world_size = 0;
    op->prev_rank = -1;
    op->next_rank = -1;
    op->total_bytes = 0;
    op->chunk_bytes = 0;
    op->num_chunks = 0;
    op->expected_contributions = 0;
    op->op = exec::ChunkOpKind::kInvalid;
    op->reserved1 = 0;
    op->user_tag = 0;
    op->src_ptr = 0;
    op->dst_ptr = 0;
    op->partial_ptr = 0;
    op->partial_bytes = 0;
    op->inbox_ptr = 0;
    op->outbox_ptr = 0;
}

__host__ __device__ __forceinline__ bool operation_desc_is_enabled(
    const OperationDesc* op) {
    return op != nullptr &&
           (op->flags & kOperationFlagEnabled) != 0u;
}

__host__ __device__ __forceinline__ bool operation_desc_is_valid(
    const OperationDesc* op) {
    return op != nullptr &&
           op->op_id != 0 &&
           op->rank >= 0 &&
           op->world_size > 0 &&
           op->prev_rank >= 0 &&
           op->next_rank >= 0 &&
           op->total_bytes > 0 &&
           op->chunk_bytes > 0 &&
           op->num_chunks ==
               operation_desc_compute_num_chunks(op->total_bytes, op->chunk_bytes) &&
           op->expected_contributions > 0 &&
           op->op != exec::ChunkOpKind::kInvalid &&
           op->src_ptr != 0 &&
           op->dst_ptr != 0 &&
           op->partial_ptr != 0 &&
           op->partial_bytes >= op->total_bytes;
}

__host__ __device__ __forceinline__ bool operation_desc_is_active(
    const OperationDesc* op) {
    return operation_desc_is_valid(op) &&
           operation_desc_is_enabled(op);
}

__host__ __device__ __forceinline__ size_t operation_desc_chunk_offset_bytes(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    return static_cast<size_t>(chunk_idx) * op->chunk_bytes;
}

__host__ __device__ __forceinline__ size_t operation_desc_chunk_bytes_at(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    if (!operation_desc_is_valid(op) || chunk_idx >= op->num_chunks) {
        return 0;
    }

    const size_t offset = operation_desc_chunk_offset_bytes(op, chunk_idx);
    const size_t remaining = op->total_bytes - offset;
    return (remaining < op->chunk_bytes) ? remaining : op->chunk_bytes;
}

__host__ __device__ __forceinline__ const unsigned char* operation_desc_src_base(
    const OperationDesc* op) {
    return reinterpret_cast<const unsigned char*>(op->src_ptr);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_dst_base(
    const OperationDesc* op) {
    return reinterpret_cast<unsigned char*>(op->dst_ptr);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_partial_base(
    const OperationDesc* op) {
    return reinterpret_cast<unsigned char*>(op->partial_ptr);
}

__host__ __device__ __forceinline__ const unsigned char* operation_desc_src_chunk_ptr(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    return operation_desc_src_base(op) +
           operation_desc_chunk_offset_bytes(op, chunk_idx);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_dst_chunk_ptr(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    return operation_desc_dst_base(op) +
           operation_desc_chunk_offset_bytes(op, chunk_idx);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_partial_chunk_ptr(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    return operation_desc_partial_base(op) +
           operation_desc_chunk_offset_bytes(op, chunk_idx);
}

// Simple reduce-scatter ownership rule for now:
// final owner = chunk_idx % world_size
__host__ __device__ __forceinline__ int operation_desc_final_owner_rank(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    if (!operation_desc_is_valid(op) || op->world_size <= 0) {
        return -1;
    }
    return static_cast<int>(chunk_idx % static_cast<uint32_t>(op->world_size));
}

__host__ __device__ __forceinline__ bool operation_desc_matches_submission(
    const OperationDesc* op,
    const exec::RangeSchedulerSubmission* sub) {
    return operation_desc_is_active(op) &&
           exec::range_scheduler_submission_is_valid(sub) &&
           sub->src == operation_desc_src_base(op) &&
           sub->bytes == op->total_bytes &&
           sub->op == op->op;
}

__host__ __device__ __forceinline__ void chunk_state_clear(
    ChunkState* st) {
    st->op_id = 0;
    st->epoch = 0;
    st->chunk_idx = 0;
    st->flags = 0;
    st->contributions_seen = 0;
    st->expected_contributions = 0;
    st->step = 0;
    st->send_count = 0;
    st->offset_bytes = 0;
    st->bytes = 0;
}

__host__ __device__ __forceinline__ bool chunk_state_is_seeded(
    const ChunkState* st) {
    return st != nullptr &&
           (st->flags & kChunkStateFlagSeeded) != 0u;
}

__host__ __device__ __forceinline__ bool chunk_state_is_active(
    const ChunkState* st) {
    return st != nullptr &&
           (st->flags & kChunkStateFlagActive) != 0u;
}

__host__ __device__ __forceinline__ bool chunk_state_is_done(
    const ChunkState* st) {
    return st != nullptr &&
           (st->flags & kChunkStateFlagDone) != 0u;
}

__host__ __device__ __forceinline__ bool chunk_state_table_is_configured(
    const ChunkStateTable* table) {
    return table != nullptr &&
           table->records != nullptr &&
           table->capacity > 0 &&
           table->device >= 0;
}

bool chunk_state_table_init(
    ChunkStateTable* table,
    int device,
    uint32_t capacity);

void chunk_state_table_reset(
    ChunkStateTable* table);

void chunk_state_table_destroy(
    ChunkStateTable* table);

} // namespace collective
} // namespace comm
} // namespace ooverlap
