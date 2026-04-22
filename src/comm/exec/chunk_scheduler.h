#pragma once

#include "comm/collective/operation.h"
#include "comm/exec/chunk.h"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {

struct RangeSchedulerSubmission {
    const unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    size_t bytes = 0;

    ChunkOpKind op = ChunkOpKind::kInvalid;

    uint32_t op_id = 0;
    int dst_rank = -1;
    uint64_t user_tag = 0;
};

__host__ __device__ __forceinline__ void range_scheduler_submission_clear(
    RangeSchedulerSubmission* sub) {
    sub->src = nullptr;
    sub->dst = nullptr;
    sub->bytes = 0;
    sub->op = ChunkOpKind::kInvalid;
    sub->op_id = 0;
    sub->dst_rank = -1;
    sub->user_tag = 0;
}

__host__ __device__ __forceinline__ bool range_scheduler_submission_is_valid(
    const RangeSchedulerSubmission* sub) {
    return sub != nullptr &&
           sub->src != nullptr &&
           sub->dst != nullptr &&
           sub->bytes > 0 &&
           sub->op != ChunkOpKind::kInvalid;
}

enum class ChunkSchedulerMode : uint8_t {
    kInvalid = 0,
    kRange = 1,
    kOperation = 2,
};

template <int WatchThreads>
struct ChunkSchedulerScratch {
    int unused = 0;
};

template <int WatchThreads = 1>
struct ChunkScheduler {
    ChunkSchedulerMode mode = ChunkSchedulerMode::kInvalid;

    const RangeSchedulerSubmission* submission = nullptr;
    const collective::OperationDesc* operation = nullptr;

    size_t chunk_bytes = 0;

    size_t next_offset_bytes = 0;

    bool has_active_chunk = false;
    size_t active_offset_bytes = 0;
    int active_chunk_idx = -1;
    uint32_t active_step = 0;

    Chunk current{};

    ChunkSchedulerScratch<WatchThreads>* scratch = nullptr;
};

template <int WatchThreads>
__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler<WatchThreads>* sched) {
    sched->mode = ChunkSchedulerMode::kInvalid;
    sched->submission = nullptr;
    sched->operation = nullptr;
    sched->chunk_bytes = 0;
    sched->next_offset_bytes = 0;
    sched->has_active_chunk = false;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = -1;
    sched->active_step = 0;
    chunk_clear(&sched->current);
    sched->scratch = nullptr;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ void chunk_scheduler_bind_scratch(
    ChunkScheduler<WatchThreads>* sched,
    ChunkSchedulerScratch<WatchThreads>* scratch) {
    sched->scratch = scratch;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ void chunk_scheduler_init(
    ChunkScheduler<WatchThreads>* sched,
    const RangeSchedulerSubmission* submission,
    size_t chunk_bytes,
    ChunkSchedulerScratch<WatchThreads>* scratch = nullptr) {
    chunk_scheduler_reset(sched);
    sched->mode = ChunkSchedulerMode::kRange;
    sched->submission = submission;
    sched->chunk_bytes = chunk_bytes;
    sched->scratch = scratch;
}

template <int WatchThreads>
__device__ __forceinline__ void chunk_scheduler_init_operation(
    ChunkScheduler<WatchThreads>* sched,
    const collective::OperationDesc* operation,
    ChunkSchedulerScratch<WatchThreads>* scratch = nullptr) {
    chunk_scheduler_reset(sched);
    sched->mode = ChunkSchedulerMode::kOperation;
    sched->operation = operation;
    sched->chunk_bytes = (operation != nullptr) ? operation->chunk_bytes : 0;
    sched->scratch = scratch;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ bool chunk_scheduler_has_current(
    const ChunkScheduler<WatchThreads>* sched) {
    return chunk_is_valid(&sched->current);
}

template <int WatchThreads>
__host__ __device__ __forceinline__ const Chunk* chunk_scheduler_current(
    const ChunkScheduler<WatchThreads>* sched) {
    return &sched->current;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ uint32_t chunk_scheduler_active_step(
    const ChunkScheduler<WatchThreads>* sched) {
    return (sched != nullptr) ? sched->active_step : 0u;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ uint32_t chunk_scheduler_active_queue_id(
    const ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr || !sched->has_active_chunk) {
        return 0;
    }

    if (sched->mode == ChunkSchedulerMode::kRange) {
        return (sched->submission != nullptr) ? sched->submission->op_id : 0u;
    }

    if (sched->mode == ChunkSchedulerMode::kOperation) {
        return (sched->operation != nullptr) ? sched->operation->op_id : 0u;
    }

    return 0;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ int chunk_scheduler_active_dst_rank(
    const ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr || !sched->has_active_chunk) {
        return -1;
    }

    if (sched->mode == ChunkSchedulerMode::kRange) {
        return (sched->submission != nullptr) ? sched->submission->dst_rank : -1;
    }

    if (sched->mode == ChunkSchedulerMode::kOperation) {
        return (sched->operation != nullptr) ? sched->operation->next_rank : -1;
    }

    return -1;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ bool chunk_scheduler_build_chunk_at_offset(
    const ChunkScheduler<WatchThreads>* sched,
    size_t offset_bytes,
    Chunk* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_clear(out);

    if (sched == nullptr ||
        sched->submission == nullptr ||
        !range_scheduler_submission_is_valid(sched->submission) ||
        sched->chunk_bytes == 0) {
        return false;
    }

    const RangeSchedulerSubmission& sub = *sched->submission;

    if (offset_bytes >= sub.bytes) {
        return false;
    }

    const size_t remaining = sub.bytes - offset_bytes;
    const size_t this_chunk_bytes =
        (remaining < sched->chunk_bytes) ? remaining : sched->chunk_bytes;
    const int chunk_idx =
        static_cast<int>(offset_bytes / sched->chunk_bytes);

    out->src = sub.src + offset_bytes;
    out->dst = sub.dst + offset_bytes;
    out->bytes = this_chunk_bytes;
    out->op_id = sub.op_id;
    out->dst_rank = sub.dst_rank;
    out->user_tag = sub.user_tag;
    out->chunk_idx = chunk_idx;
    out->range_offset_bytes = offset_bytes;
    out->op = sub.op;

    return true;
}

template <int WatchThreads>
__device__ __forceinline__ uint32_t chunk_scheduler_atomic_load_u32(
    volatile uint32_t* ptr) {
    return atomicAdd(
        reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(ptr)),
        0u);
}

template <int WatchThreads>
__device__ __forceinline__ void chunk_scheduler_atomic_store_u32(
    volatile uint32_t* ptr,
    uint32_t value) {
    atomicExch(
        reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(ptr)),
        value);
}

template <int WatchThreads>
__device__ __forceinline__ void chunk_scheduler_build_operation_chunk(
    const collective::OperationDesc* op,
    uint32_t chunk_idx,
    uint32_t step,
    Chunk* out) {
    chunk_clear(out);

    out->src =
        collective::operation_desc_accum_chunk_ptr(op, chunk_idx);
    out->dst =
        collective::operation_desc_next_accum_chunk_ptr(op, chunk_idx);
    out->bytes =
        collective::operation_desc_chunk_bytes_at(op, chunk_idx);
    out->op_id = op->op_id;
    out->dst_rank = op->next_rank;
    out->user_tag = op->user_tag;
    out->chunk_idx = static_cast<int>(chunk_idx);
    out->range_offset_bytes =
        collective::operation_desc_chunk_offset_bytes(op, chunk_idx);
    out->op =
        collective::operation_desc_chunk_op_for_step(op, step);
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_try_activate_next_range_chunk(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr) {
        return false;
    }

    if (sched->has_active_chunk) {
        return true;
    }

    if (!chunk_scheduler_build_chunk_at_offset(
            sched,
            sched->next_offset_bytes,
            &sched->current)) {
        return false;
    }

    sched->has_active_chunk = true;
    sched->active_offset_bytes = sched->next_offset_bytes;
    sched->active_chunk_idx = sched->current.chunk_idx;
    sched->active_step = 0;
    return true;
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_try_activate_next_operation_chunk(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr || sched->operation == nullptr) {
        return false;
    }

    if (sched->has_active_chunk) {
        return true;
    }

    const collective::OperationDesc* op = sched->operation;
    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(op);

    volatile uint32_t* inbound_steps =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_inbound_steps(op));
    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(op);

    for (uint32_t idx = 0; idx < op->num_chunks; ++idx) {
        if (collective::chunk_state_is_in_flight(&chunk_states[idx])) {
            continue;
        }

        const uint32_t step =
            chunk_scheduler_atomic_load_u32<WatchThreads>(&inbound_steps[idx]);

        if (step == collective::kOperationInboundStepInvalid) {
            continue;
        }
        if (step >= total_steps) {
            continue;
        }

        const int actor =
            collective::operation_desc_actor_rank_for_step(op, idx, step);
        if (actor != op->rank) {
            continue;
        }

        chunk_scheduler_build_operation_chunk(op, idx, step, &sched->current);

        sched->has_active_chunk = true;
        sched->active_offset_bytes = sched->current.range_offset_bytes;
        sched->active_chunk_idx = static_cast<int>(idx);
        sched->active_step = step;

        chunk_states[idx].last_step_started = step;
        chunk_states[idx].flags |= collective::kChunkStateFlagInFlight;
        return true;
    }

    return false;
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_try_activate_next_chunk(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr) {
        return false;
    }

    if (sched->mode == ChunkSchedulerMode::kRange) {
        return chunk_scheduler_try_activate_next_range_chunk(sched);
    }
    if (sched->mode == ChunkSchedulerMode::kOperation) {
        return chunk_scheduler_try_activate_next_operation_chunk(sched);
    }
    return false;
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_try_prime_current(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr) {
        return false;
    }
    if (chunk_is_valid(&sched->current)) {
        return true;
    }
    return chunk_scheduler_try_activate_next_chunk(sched);
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_peek_next(
    const ChunkScheduler<WatchThreads>* sched,
    Chunk* out) {
    if (sched == nullptr || out == nullptr) {
        return false;
    }

    if (sched->mode != ChunkSchedulerMode::kRange) {
        // Operation mode cannot safely look ahead until current retirement
        // publishes the next step tokens.
        return false;
    }

    const size_t next_offset =
        sched->has_active_chunk
            ? (sched->active_offset_bytes + sched->current.bytes)
            : sched->next_offset_bytes;

    return chunk_scheduler_build_chunk_at_offset(sched, next_offset, out);
}

template <int WatchThreads>
__device__ __forceinline__ void chunk_scheduler_retire_range_current(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr) {
        return;
    }

    if (!sched->has_active_chunk) {
        chunk_clear(&sched->current);
        return;
    }

    sched->next_offset_bytes = sched->active_offset_bytes + sched->current.bytes;
    sched->has_active_chunk = false;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = -1;
    sched->active_step = 0;
    chunk_clear(&sched->current);

    chunk_scheduler_try_activate_next_range_chunk(sched);
}

template <int WatchThreads>
__device__ __forceinline__ void chunk_scheduler_retire_operation_current(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr || sched->operation == nullptr || !sched->has_active_chunk) {
        return;
    }

    const collective::OperationDesc* op = sched->operation;
    const uint32_t chunk_idx = static_cast<uint32_t>(sched->active_chunk_idx);
    const uint32_t current_step = sched->active_step;
    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(op);
    const uint32_t reduce_steps =
        static_cast<uint32_t>(op->world_size - 1);
    const uint32_t next_step = current_step + 1u;

    volatile uint32_t* local_inbound =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_inbound_steps(op));
    volatile uint32_t* local_done =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_done(op));
    volatile uint32_t* next_inbound =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_next_inbound_steps(op));
    volatile uint32_t* next_done =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_next_done(op));
    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(op);

    chunk_scheduler_atomic_store_u32<WatchThreads>(
        &local_inbound[chunk_idx],
        collective::kOperationInboundStepInvalid);

    __threadfence_system();

    if (current_step < reduce_steps) {
        if (next_step == reduce_steps) {
            chunk_scheduler_atomic_store_u32<WatchThreads>(&next_done[chunk_idx], 1u);
        }
        if (next_step < total_steps) {
            chunk_scheduler_atomic_store_u32<WatchThreads>(&next_inbound[chunk_idx], next_step);
        }
    } else {
        chunk_scheduler_atomic_store_u32<WatchThreads>(&next_done[chunk_idx], 1u);
        if (next_step < total_steps) {
            chunk_scheduler_atomic_store_u32<WatchThreads>(&next_inbound[chunk_idx], next_step);
        }
    }

    if (next_step >= total_steps) {
        chunk_scheduler_atomic_store_u32<WatchThreads>(&local_done[chunk_idx], 1u);
    }

    __threadfence_system();

    chunk_states[chunk_idx].last_step_completed = next_step;
    chunk_states[chunk_idx].flags &= ~collective::kChunkStateFlagInFlight;

    if (chunk_scheduler_atomic_load_u32<WatchThreads>(&local_done[chunk_idx]) == 1u) {
        chunk_states[chunk_idx].flags |= collective::kChunkStateFlagDone;
    }

    sched->has_active_chunk = false;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = -1;
    sched->active_step = 0;
    chunk_clear(&sched->current);

    chunk_scheduler_try_activate_next_operation_chunk(sched);
}

template <int WatchThreads>
__device__ __forceinline__ void chunk_scheduler_retire_current(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr) {
        return;
    }

    if (sched->mode == ChunkSchedulerMode::kRange) {
        chunk_scheduler_retire_range_current(sched);
        return;
    }
    if (sched->mode == ChunkSchedulerMode::kOperation) {
        chunk_scheduler_retire_operation_current(sched);
        return;
    }
}

template <int WatchThreads>
__device__ __forceinline__ void chunk_scheduler_advance(
    ChunkScheduler<WatchThreads>* sched) {
    chunk_scheduler_retire_current(sched);
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
