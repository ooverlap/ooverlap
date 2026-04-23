#pragma once

#include "comm/collective/operation.h"
#include "comm/exec/chunk.h"

#include <cstddef>
#include <cstdint>

#ifndef OOVERLAP_ENDPOINT_DEBUG
#define OOVERLAP_ENDPOINT_DEBUG 0
#endif

#if OOVERLAP_ENDPOINT_DEBUG
#include <cstdio>
#define OOVERLAP_SCHED_DBG(...) printf(__VA_ARGS__)
#else
#define OOVERLAP_SCHED_DBG(...) ((void)0)
#endif

namespace ooverlap {
namespace comm {
namespace exec {

struct ChunkScheduler {
    const collective::OperationDesc* operation = nullptr;

    bool has_active_chunk = false;
    int active_chunk_idx = -1;
    uint32_t active_step = 0;

    // Kept for current kernel debug prints.
    uint32_t next_search_idx = 0;
    uint32_t ready_count = 0;

    Chunk current{};
};

__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler* sched) {
    sched->operation = nullptr;
    sched->has_active_chunk = false;
    sched->active_chunk_idx = -1;
    sched->active_step = 0;
    sched->next_search_idx = 0;
    sched->ready_count = 0;
    chunk_clear(&sched->current);
}

__device__ __forceinline__ void chunk_scheduler_init_operation(
    ChunkScheduler* sched,
    const collective::OperationDesc* operation) {
    chunk_scheduler_reset(sched);
    sched->operation = operation;
}

__host__ __device__ __forceinline__ bool chunk_scheduler_has_current(
    const ChunkScheduler* sched) {
    return chunk_is_valid(&sched->current);
}

__host__ __device__ __forceinline__ const Chunk* chunk_scheduler_current(
    const ChunkScheduler* sched) {
    return &sched->current;
}

__host__ __device__ __forceinline__ uint32_t chunk_scheduler_active_step(
    const ChunkScheduler* sched) {
    return (sched != nullptr) ? sched->active_step : 0u;
}

__host__ __device__ __forceinline__ uint32_t chunk_scheduler_active_queue_id(
    const ChunkScheduler* sched) {
    if (sched == nullptr || !sched->has_active_chunk || sched->operation == nullptr) {
        return 0u;
    }
    return sched->operation->op_id;
}

__host__ __device__ __forceinline__ int chunk_scheduler_active_dst_rank(
    const ChunkScheduler* sched) {
    if (sched == nullptr || !sched->has_active_chunk || sched->operation == nullptr) {
        return -1;
    }
    return sched->operation->next_rank;
}

__device__ __forceinline__ uint32_t chunk_scheduler_atomic_load_u32(
    volatile uint32_t* ptr) {
    return atomicAdd(
        reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(ptr)),
        0u);
}

__device__ __forceinline__ void chunk_scheduler_atomic_store_u32(
    volatile uint32_t* ptr,
    uint32_t value) {
    atomicExch(
        reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(ptr)),
        value);
}

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

__device__ __forceinline__ bool chunk_scheduler_candidate_is_valid(
    const ChunkScheduler* sched,
    uint32_t chunk_idx,
    uint32_t expected_step) {
    if (sched == nullptr || sched->operation == nullptr) {
        return false;
    }

    const collective::OperationDesc* op = sched->operation;
    if (chunk_idx >= op->num_chunks) {
        return false;
    }

    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(op);

    if (expected_step == collective::kOperationInboundStepInvalid) {
        return false;
    }
    if (expected_step >= total_steps) {
        return false;
    }

    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(op);
    if (collective::chunk_state_is_in_flight(&chunk_states[chunk_idx])) {
        return false;
    }

    const int actor =
        collective::operation_desc_actor_rank_for_step(op, chunk_idx, expected_step);

    return actor == op->rank;
}

__device__ __forceinline__ void chunk_scheduler_refill_ready_cache(
    ChunkScheduler* sched) {
    if (sched == nullptr || sched->operation == nullptr || threadIdx.x != 0) {
        return;
    }

    volatile uint32_t* head_ptr =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_ready_head(sched->operation));
    volatile uint32_t* tail_ptr =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_ready_tail(sched->operation));

    const uint32_t head = chunk_scheduler_atomic_load_u32(head_ptr);
    const uint32_t tail = chunk_scheduler_atomic_load_u32(tail_ptr);
    sched->ready_count = tail - head;
}

__device__ __forceinline__ bool chunk_scheduler_try_activate_next_chunk(
    ChunkScheduler* sched) {
    if (sched == nullptr || sched->operation == nullptr) {
        return false;
    }

    if (sched->has_active_chunk) {
        return true;
    }

    const collective::OperationDesc* op = sched->operation;
    collective::ReadyItem* local_items = collective::operation_desc_local_ready_items(op);
    volatile uint32_t* head_ptr =
        reinterpret_cast<volatile uint32_t*>(collective::operation_desc_local_ready_head(op));
    volatile uint32_t* tail_ptr =
        reinterpret_cast<volatile uint32_t*>(collective::operation_desc_local_ready_tail(op));

    while (true) {
        const uint32_t head = chunk_scheduler_atomic_load_u32(head_ptr);
        const uint32_t tail = chunk_scheduler_atomic_load_u32(tail_ptr);

        sched->ready_count = tail - head;
        if (head == tail) {
            return false;
        }

        const collective::ReadyItem cand = local_items[head % op->num_chunks];
        chunk_scheduler_atomic_store_u32(head_ptr, head + 1u);

        if (!chunk_scheduler_candidate_is_valid(sched, cand.chunk_idx, cand.step)) {
#if OOVERLAP_ENDPOINT_DEBUG
            OOVERLAP_SCHED_DBG(
                "[stale] rank=%d chunk=%u step=%u ready_count=%u\n",
                sched->operation->rank,
                cand.chunk_idx,
                cand.step,
                sched->ready_count);
#endif
            continue;
        }

        collective::ChunkState* chunk_states =
            collective::operation_desc_chunk_states(op);

        if (!collective::chunk_state_is_initialized(&chunk_states[cand.chunk_idx])) {
            collective::chunk_state_clear(&chunk_states[cand.chunk_idx]);
            chunk_states[cand.chunk_idx].chunk_idx = cand.chunk_idx;
            chunk_states[cand.chunk_idx].offset_bytes =
                collective::operation_desc_chunk_offset_bytes(op, cand.chunk_idx);
            chunk_states[cand.chunk_idx].bytes =
                collective::operation_desc_chunk_bytes_at(op, cand.chunk_idx);
            chunk_states[cand.chunk_idx].flags |=
                collective::kChunkStateFlagInitialized;
        }
 

        chunk_scheduler_build_operation_chunk(
            op,
            cand.chunk_idx,
            cand.step,
            &sched->current);

        sched->has_active_chunk = true;
        sched->active_chunk_idx = static_cast<int>(cand.chunk_idx);
        sched->active_step = cand.step;

        chunk_states[cand.chunk_idx].last_step_started = cand.step;
        chunk_states[cand.chunk_idx].flags |= collective::kChunkStateFlagInFlight;

#if OOVERLAP_ENDPOINT_DEBUG
        OOVERLAP_SCHED_DBG(
            "[activate] rank=%d chunk=%u step=%u queue_count=%u\n",
            op->rank,
            cand.chunk_idx,
            cand.step,
            sched->ready_count);
#endif
        return true;
    }
}

__device__ __forceinline__ bool chunk_scheduler_try_prime_current(
    ChunkScheduler* sched) {
    if (sched == nullptr) {
        return false;
    }
    if (chunk_is_valid(&sched->current)) {
        return true;
    }
    return chunk_scheduler_try_activate_next_chunk(sched);
}

__device__ __forceinline__ void chunk_scheduler_handoff_current(
    ChunkScheduler* sched) {
    if (sched == nullptr) {
        return;
    }

    sched->has_active_chunk = false;
    sched->active_chunk_idx = -1;
    sched->active_step = 0;
    chunk_clear(&sched->current);
}

__device__ __forceinline__ void chunk_scheduler_retire_stage(
    ChunkScheduler* sched,
    uint32_t chunk_idx,
    uint32_t current_step) {
    if (sched == nullptr || sched->operation == nullptr) {
        return;
    }

    const collective::OperationDesc* op = sched->operation;
    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(op);
    const uint32_t reduce_steps =
        static_cast<uint32_t>(op->world_size - 1);
    const uint32_t next_step = current_step + 1u;

    volatile uint32_t* local_done =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_done(op));
    volatile uint32_t* next_done =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_next_done(op));
    volatile uint32_t* next_tail =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_next_ready_tail(op));
    collective::ReadyItem* next_items =
        collective::operation_desc_next_ready_items(op);
    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(op);

    if (current_step < reduce_steps) {
        if (next_step == reduce_steps) {
            chunk_scheduler_atomic_store_u32(&next_done[chunk_idx], 1u);
        }
    } else {
        chunk_scheduler_atomic_store_u32(&next_done[chunk_idx], 1u);
    }

    if (next_step < total_steps) {
        const uint32_t tail = chunk_scheduler_atomic_load_u32(next_tail);
        next_items[tail % op->num_chunks].chunk_idx = chunk_idx;
        next_items[tail % op->num_chunks].step = next_step;
        __threadfence();
        chunk_scheduler_atomic_store_u32(next_tail, tail + 1u);
    }

    if (next_step >= total_steps) {
        chunk_scheduler_atomic_store_u32(&local_done[chunk_idx], 1u);
        if (op->completion_count_ptr != 0 &&
           op->completion_flag_ptr != 0 &&
           op->completion_target != 0) {
           volatile uint32_t* completion_count =
               reinterpret_cast<volatile uint32_t*>(
                   collective::operation_desc_completion_count(op));
           volatile uint32_t* completion_flag =
               reinterpret_cast<volatile uint32_t*>(
                   collective::operation_desc_completion_flag(op));
     
           const uint32_t completed =
             atomicAdd(
                 reinterpret_cast<unsigned int*>(
                     const_cast<uint32_t*>(completion_count)),
                 1u) + 1u;
     
           if (completed >= op->completion_target) {
               __threadfence();
               chunk_scheduler_atomic_store_u32(completion_flag, 1u);
           }
        }
    }

    __threadfence();

    chunk_states[chunk_idx].last_step_completed = next_step;
    chunk_states[chunk_idx].flags &= ~collective::kChunkStateFlagInFlight;

    if (chunk_scheduler_atomic_load_u32(&local_done[chunk_idx]) == 1u) {
        chunk_states[chunk_idx].flags |= collective::kChunkStateFlagDone;
    }

#if OOVERLAP_ENDPOINT_DEBUG
    OOVERLAP_SCHED_DBG(
        "[retire] rank=%d chunk=%u cur_step=%u next_step=%u local_done=%u next_tail=%u next_done=%u\n",
        op->rank,
        chunk_idx,
        current_step,
        next_step,
        chunk_scheduler_atomic_load_u32(&local_done[chunk_idx]),
        (next_step < total_steps)
            ? chunk_scheduler_atomic_load_u32(next_tail)
            : 0u,
        chunk_scheduler_atomic_load_u32(&next_done[chunk_idx]));
#endif

    sched->ready_count = 0;
}

__device__ __forceinline__ void chunk_scheduler_retire_current(
    ChunkScheduler* sched) {
    if (sched == nullptr || sched->operation == nullptr || !sched->has_active_chunk) {
        return;
    }

    chunk_scheduler_retire_stage(
        sched,
        static_cast<uint32_t>(sched->active_chunk_idx),
        sched->active_step);

    chunk_scheduler_handoff_current(sched);
}

__device__ __forceinline__ void chunk_scheduler_advance(
    ChunkScheduler* sched) {
    chunk_scheduler_retire_current(sched);
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
