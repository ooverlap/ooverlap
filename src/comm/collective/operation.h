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
    kChunkStateFlagInitialized = 1u << 0,
    kChunkStateFlagInFlight = 1u << 1,
    kChunkStateFlagDone = 1u << 2,
};

static constexpr uint32_t kOperationInboundStepInvalid = 0xffffffffu;

// Minimal ring all-reduce operation descriptor using per-rank local progress.
//
// Each rank owns:
// - accum_ptr                : local full buffer
// - inbound_steps_ptr        : local inbound step mailbox (polled locally)
// - done_ptr                 : local done flags (polled locally / by host)
// - chunk_states_ptr         : local per-chunk bookkeeping
//
// Previous rank writes remotely into:
// - next_accum_ptr
// - next_inbound_steps_ptr
// - next_done_ptr
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
    uint32_t reserved1 = 0;

    exec::ChunkOpKind op = exec::ChunkOpKind::kInvalid;
    uint32_t reserved2 = 0;
    uint64_t user_tag = 0;

    // Local full buffer: starts as local partial, ends as full all-reduce result.
    uint64_t accum_ptr = 0;
    size_t accum_bytes = 0;

    // Mapped pointer to next rank's full buffer.
    uint64_t next_accum_ptr = 0;
    size_t next_accum_bytes = 0;

    // Local inbound step mailbox: one uint32 per chunk.
    uint64_t inbound_steps_ptr = 0;
    size_t inbound_steps_bytes = 0;

    // Mapped pointer to next rank's local inbound mailbox.
    uint64_t next_inbound_steps_ptr = 0;
    size_t next_inbound_steps_bytes = 0;

    // Local done flags: one uint32 per chunk.
    uint64_t done_ptr = 0;
    size_t done_bytes = 0;

    // Mapped pointer to next rank's local done flags.
    uint64_t next_done_ptr = 0;
    size_t next_done_bytes = 0;

    // Local per-rank bookkeeping, one ChunkState per chunk.
    uint64_t chunk_states_ptr = 0;
};

struct ChunkState {
    uint32_t chunk_idx = 0;
    uint32_t last_step_started = 0;
    uint32_t last_step_completed = 0;
    uint32_t flags = 0;

    size_t offset_bytes = 0;
    size_t bytes = 0;
};

struct ChunkStateTable {
    ChunkState* records = nullptr;  // device pointer
    uint32_t capacity = 0;
    int device = -1;
};

struct DeviceOperationDesc {
    OperationDesc* ptr = nullptr;  // device pointer
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

__host__ __device__ __forceinline__ uint32_t operation_desc_total_ring_steps(
    const OperationDesc* op) {
    if (op == nullptr || op->world_size <= 1) {
        return 0;
    }
    return static_cast<uint32_t>(2 * (op->world_size - 1));
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
    op->reserved1 = 0;
    op->op = exec::ChunkOpKind::kInvalid;
    op->reserved2 = 0;
    op->user_tag = 0;

    op->accum_ptr = 0;
    op->accum_bytes = 0;
    op->next_accum_ptr = 0;
    op->next_accum_bytes = 0;

    op->inbound_steps_ptr = 0;
    op->inbound_steps_bytes = 0;
    op->next_inbound_steps_ptr = 0;
    op->next_inbound_steps_bytes = 0;

    op->done_ptr = 0;
    op->done_bytes = 0;
    op->next_done_ptr = 0;
    op->next_done_bytes = 0;

    op->chunk_states_ptr = 0;
}

__host__ __device__ __forceinline__ bool operation_desc_is_enabled(
    const OperationDesc* op) {
    return op != nullptr &&
           (op->flags & kOperationFlagEnabled) != 0u;
}

__host__ __device__ __forceinline__ bool operation_desc_is_valid(
    const OperationDesc* op) {
    if (op == nullptr) {
        return false;
    }

    const size_t num_chunks =
        static_cast<size_t>(operation_desc_compute_num_chunks(op->total_bytes, op->chunk_bytes));
    const size_t progress_bytes = num_chunks * sizeof(uint32_t);

    if (op->op_id == 0 ||
        op->rank < 0 ||
        op->world_size <= 0 ||
        op->prev_rank < 0 ||
        op->next_rank < 0 ||
        op->total_bytes == 0 ||
        op->chunk_bytes == 0 ||
        op->num_chunks != num_chunks ||
        op->op == exec::ChunkOpKind::kInvalid ||
        op->accum_ptr == 0 ||
        op->accum_bytes < op->total_bytes ||
        op->inbound_steps_ptr == 0 ||
        op->inbound_steps_bytes < progress_bytes ||
        op->done_ptr == 0 ||
        op->done_bytes < progress_bytes ||
        op->chunk_states_ptr == 0) {
        return false;
    }

    if (op->world_size > 1) {
        if (op->next_accum_ptr == 0 ||
            op->next_accum_bytes < op->total_bytes ||
            op->next_inbound_steps_ptr == 0 ||
            op->next_inbound_steps_bytes < progress_bytes ||
            op->next_done_ptr == 0 ||
            op->next_done_bytes < progress_bytes) {
            return false;
        }
    }

    return true;
}

__host__ __device__ __forceinline__ bool operation_desc_is_active(
    const OperationDesc* op) {
    return operation_desc_is_valid(op) &&
           operation_desc_is_enabled(op);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_accum_base(
    const OperationDesc* op) {
    return reinterpret_cast<unsigned char*>(op->accum_ptr);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_next_accum_base(
    const OperationDesc* op) {
    return reinterpret_cast<unsigned char*>(op->next_accum_ptr);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_local_inbound_steps(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->inbound_steps_ptr);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_next_inbound_steps(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->next_inbound_steps_ptr);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_local_done(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->done_ptr);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_next_done(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->next_done_ptr);
}

__host__ __device__ __forceinline__ ChunkState* operation_desc_chunk_states(
    const OperationDesc* op) {
    return reinterpret_cast<ChunkState*>(op->chunk_states_ptr);
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

__host__ __device__ __forceinline__ unsigned char* operation_desc_accum_chunk_ptr(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    return operation_desc_accum_base(op) +
           operation_desc_chunk_offset_bytes(op, chunk_idx);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_next_accum_chunk_ptr(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    return operation_desc_next_accum_base(op) +
           operation_desc_chunk_offset_bytes(op, chunk_idx);
}

__host__ __device__ __forceinline__ int operation_desc_chunk_owner_rank(
    const OperationDesc* op,
    uint32_t chunk_idx) {
    if (!operation_desc_is_valid(op) || op->world_size <= 0) {
        return -1;
    }
    return static_cast<int>(chunk_idx % static_cast<uint32_t>(op->world_size));
}

__host__ __device__ __forceinline__ int operation_desc_actor_rank_for_step(
    const OperationDesc* op,
    uint32_t chunk_idx,
    uint32_t step) {
    if (!operation_desc_is_valid(op)) {
        return -1;
    }

    const int owner = operation_desc_chunk_owner_rank(op, chunk_idx);
    const uint32_t reduce_steps =
        (op->world_size <= 1) ? 0u : static_cast<uint32_t>(op->world_size - 1);

    if (step < reduce_steps) {
        const int start = (owner + 1) % op->world_size;
        return (start + static_cast<int>(step)) % op->world_size;
    }

    const uint32_t gather_step = step - reduce_steps;
    return (owner + static_cast<int>(gather_step)) % op->world_size;
}

__host__ __device__ __forceinline__ bool operation_desc_step_is_reduce_phase(
    const OperationDesc* op,
    uint32_t step) {
    if (!operation_desc_is_valid(op) || op->world_size <= 1) {
        return false;
    }
    return step < static_cast<uint32_t>(op->world_size - 1);
}

__host__ __device__ __forceinline__ exec::ChunkOpKind operation_desc_chunk_op_for_step(
    const OperationDesc* op,
    uint32_t step) {
    return operation_desc_step_is_reduce_phase(op, step)
        ? op->op
        : exec::ChunkOpKind::kCopy;
}

__host__ __device__ __forceinline__ void chunk_state_clear(
    ChunkState* st) {
    st->chunk_idx = 0;
    st->last_step_started = 0;
    st->last_step_completed = 0;
    st->flags = 0;
    st->offset_bytes = 0;
    st->bytes = 0;
}

__host__ __device__ __forceinline__ bool chunk_state_is_initialized(
    const ChunkState* st) {
    return st != nullptr &&
           (st->flags & kChunkStateFlagInitialized) != 0u;
}

__host__ __device__ __forceinline__ bool chunk_state_is_in_flight(
    const ChunkState* st) {
    return st != nullptr &&
           (st->flags & kChunkStateFlagInFlight) != 0u;
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

__host__ __device__ __forceinline__ bool device_operation_desc_is_configured(
    const DeviceOperationDesc* op) {
    return op != nullptr &&
           op->ptr != nullptr &&
           op->device >= 0;
}

bool operation_desc_build_ring_allreduce(
    OperationDesc* out,
    int rank,
    int world_size,
    size_t total_bytes,
    size_t chunk_bytes,
    exec::ChunkOpKind op,
    void* accum_ptr,
    void* next_accum_ptr,
    void* inbound_steps_ptr,
    void* next_inbound_steps_ptr,
    void* done_ptr,
    void* next_done_ptr,
    ChunkState* chunk_states_ptr,
    uint32_t op_id,
    uint32_t epoch = 1,
    bool enabled = true,
    uint64_t user_tag = 0);

bool operation_desc_reset_local_state(
    int device,
    const OperationDesc* desc);

bool device_operation_desc_create(
    DeviceOperationDesc* storage,
    int device,
    const OperationDesc* host_desc);

bool device_operation_desc_write(
    DeviceOperationDesc* storage,
    const OperationDesc* host_desc);

void device_operation_desc_destroy(
    DeviceOperationDesc* storage);

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
