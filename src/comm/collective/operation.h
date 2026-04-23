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

struct ReadyItem {
    uint32_t chunk_idx = 0;
    uint32_t step = kOperationInboundStepInvalid;
};

// Ring all-reduce operation descriptor.
//
// New semantics:
// - scheduling is deterministic from (chunk_idx, step)
// - signaling is monotonic per-chunk progress
//
// Compatibility note:
// - ready_queue_* fields are retained so the surrounding host code can migrate
//   incrementally, but the new scheduler no longer consumes or produces queue
//   items on the hot path.
// - done_ptr now stores LOCAL progress counters, one uint32 per chunk.
// - next_done_ptr now stores PREV-RANK progress counters, one uint32 per chunk,
//   visible from this rank.
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

    uint64_t accum_ptr = 0;
    size_t accum_bytes = 0;

    uint64_t next_accum_ptr = 0;
    size_t next_accum_bytes = 0;

    // Deprecated hot-path storage. Retained for compatibility during migration.
    uint64_t ready_queue_ptr = 0;
    size_t ready_queue_bytes = 0;

    uint64_t next_ready_queue_ptr = 0;
    size_t next_ready_queue_bytes = 0;

    // Progress storage:
    //   done_ptr      : local progress[chunk] = completed global step count
    //   next_done_ptr : previous-rank progress visible from this rank
    uint64_t done_ptr = 0;
    size_t done_bytes = 0;

    uint64_t next_done_ptr = 0;
    size_t next_done_bytes = 0;

    uint64_t chunk_states_ptr = 0;

    uint64_t completion_count_ptr = 0;
    uint64_t completion_flag_ptr = 0;
    uint32_t completion_target = 0;
    uint32_t reserved3 = 0;
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
    ChunkState* records = nullptr;
    uint32_t capacity = 0;
    int device = -1;
};

struct DeviceOperationDesc {
    OperationDesc* ptr = nullptr;
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

__host__ __device__ __forceinline__ size_t operation_desc_ready_queue_storage_bytes(
    const OperationDesc* op) {
    if (op == nullptr) {
        return 0;
    }
    return 2 * sizeof(uint32_t) +
           static_cast<size_t>(op->num_chunks) * sizeof(ReadyItem);
}

__host__ __device__ __forceinline__ size_t operation_desc_progress_storage_bytes(
    const OperationDesc* op) {
    if (op == nullptr) {
        return 0;
    }
    return static_cast<size_t>(op->num_chunks) * sizeof(uint32_t);
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

    op->ready_queue_ptr = 0;
    op->ready_queue_bytes = 0;
    op->next_ready_queue_ptr = 0;
    op->next_ready_queue_bytes = 0;

    op->done_ptr = 0;
    op->done_bytes = 0;
    op->next_done_ptr = 0;
    op->next_done_bytes = 0;

    op->chunk_states_ptr = 0;

    op->completion_count_ptr = 0;
    op->completion_flag_ptr = 0;
    op->completion_target = 0;
    op->reserved3 = 0;
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
        static_cast<size_t>(
            operation_desc_compute_num_chunks(op->total_bytes, op->chunk_bytes));
    const size_t queue_bytes =
        2 * sizeof(uint32_t) + num_chunks * sizeof(ReadyItem);
    const size_t progress_bytes =
        num_chunks * sizeof(uint32_t);

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
        op->done_ptr == 0 ||
        op->done_bytes < progress_bytes ||
        op->chunk_states_ptr == 0) {
        return false;
    }

    if (op->ready_queue_ptr != 0 &&
        op->ready_queue_bytes < queue_bytes) {
        return false;
    }

    if (op->world_size > 1) {
        if (op->next_accum_ptr == 0 ||
            op->next_accum_bytes < op->total_bytes ||
            op->next_done_ptr == 0 ||
            op->next_done_bytes < progress_bytes) {
            return false;
        }

        if (op->next_ready_queue_ptr != 0 &&
            op->next_ready_queue_bytes < queue_bytes) {
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

__host__ __device__ __forceinline__ unsigned char* operation_desc_local_ready_queue_base(
    const OperationDesc* op) {
    return reinterpret_cast<unsigned char*>(op->ready_queue_ptr);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_next_ready_queue_base(
    const OperationDesc* op) {
    return reinterpret_cast<unsigned char*>(op->next_ready_queue_ptr);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_local_ready_head(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->ready_queue_ptr);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_local_ready_tail(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->ready_queue_ptr) + 1;
}

__host__ __device__ __forceinline__ ReadyItem* operation_desc_local_ready_items(
    const OperationDesc* op) {
    return reinterpret_cast<ReadyItem*>(
        reinterpret_cast<uint32_t*>(op->ready_queue_ptr) + 2);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_next_ready_tail(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->next_ready_queue_ptr) + 1;
}

__host__ __device__ __forceinline__ ReadyItem* operation_desc_next_ready_items(
    const OperationDesc* op) {
    return reinterpret_cast<ReadyItem*>(
        reinterpret_cast<uint32_t*>(op->next_ready_queue_ptr) + 2);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_local_progress(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->done_ptr);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_prev_progress(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->next_done_ptr);
}

// Backward-compatible aliases. Old callers that still think of these as
// "done flags" will compile, but the new runtime semantics are monotonic
// progress counters.
__host__ __device__ __forceinline__ uint32_t* operation_desc_local_done(
    const OperationDesc* op) {
    return operation_desc_local_progress(op);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_next_done(
    const OperationDesc* op) {
    return operation_desc_prev_progress(op);
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

__host__ __device__ __forceinline__ uint32_t* operation_desc_completion_count(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->completion_count_ptr);
}

__host__ __device__ __forceinline__ uint32_t* operation_desc_completion_flag(
    const OperationDesc* op) {
    return reinterpret_cast<uint32_t*>(op->completion_flag_ptr);
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
    return static_cast<int>(
        chunk_idx % static_cast<uint32_t>(op->world_size));
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

__host__ __device__ __forceinline__ bool operation_desc_step_ready_from_prev(
    const OperationDesc* op,
    uint32_t chunk_idx,
    uint32_t step,
    uint32_t prev_progress_value) {
    if (!operation_desc_is_valid(op) || chunk_idx >= op->num_chunks) {
        return false;
    }
    if (step == 0u) {
        return true;
    }
    return prev_progress_value >= step;
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

__host__ __device__ __forceinline__ uint32_t operation_desc_local_completion_target(
    const OperationDesc* op) {
    if (!operation_desc_is_valid(op)) {
        return 0;
    }

    const uint32_t total_steps = operation_desc_total_ring_steps(op);
    if (total_steps == 0) {
        return op->num_chunks;
    }

    const uint32_t final_step = total_steps - 1u;
    uint32_t target = 0;
    for (uint32_t idx = 0; idx < op->num_chunks; ++idx) {
        if (operation_desc_actor_rank_for_step(op, idx, final_step) == op->rank) {
            ++target;
        }
    }
    return target;
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
    void* ready_queue_ptr,
    void* next_ready_queue_ptr,
    void* progress_ptr,
    void* prev_progress_ptr,
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
