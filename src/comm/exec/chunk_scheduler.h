#pragma once

#include "comm/collective/operation.h"
#include "comm/exec/chunk.h"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {

static constexpr uint32_t kChunkSchedulerReadyCapacity = 32;
static constexpr uint32_t kChunkSchedulerSearchWarp = 32;

struct ReadyCandidate {
    uint32_t chunk_idx = 0;
    uint32_t step = collective::kOperationInboundStepInvalid;
};

struct ChunkScheduler {
    const collective::OperationDesc* operation = nullptr;

    bool has_active_chunk = false;
    int active_chunk_idx = -1;
    uint32_t active_step = 0;

    uint32_t next_search_idx = 0;

    uint32_t ready_head = 0;
    uint32_t ready_tail = 0;
    uint32_t ready_count = 0;
    ReadyCandidate ready[kChunkSchedulerReadyCapacity]{};

    ReadyCandidate search_found[kChunkSchedulerSearchWarp]{};

    Chunk current{};
};

__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler* sched) {
    sched->operation = nullptr;
    sched->has_active_chunk = false;
    sched->active_chunk_idx = -1;
    sched->active_step = 0;
    sched->next_search_idx = 0;
    sched->ready_head = 0;
    sched->ready_tail = 0;
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

__device__ __forceinline__ bool chunk_scheduler_ready_empty(
    const ChunkScheduler* sched) {
    return sched->ready_count == 0;
}

__device__ __forceinline__ bool chunk_scheduler_ready_full(
    const ChunkScheduler* sched) {
    return sched->ready_count >= kChunkSchedulerReadyCapacity;
}

__device__ __forceinline__ void chunk_scheduler_ready_push(
    ChunkScheduler* sched,
    uint32_t chunk_idx,
    uint32_t step) {
    if (chunk_scheduler_ready_full(sched)) {
        return;
    }

    sched->ready[sched->ready_tail].chunk_idx = chunk_idx;
    sched->ready[sched->ready_tail].step = step;

    sched->ready_tail++;
    if (sched->ready_tail == kChunkSchedulerReadyCapacity) {
        sched->ready_tail = 0;
    }
    sched->ready_count++;
}

__device__ __forceinline__ bool chunk_scheduler_ready_pop(
    ChunkScheduler* sched,
    ReadyCandidate* out) {
    if (chunk_scheduler_ready_empty(sched)) {
        return false;
    }

    *out = sched->ready[sched->ready_head];

    sched->ready_head++;
    if (sched->ready_head == kChunkSchedulerReadyCapacity) {
        sched->ready_head = 0;
    }
    sched->ready_count--;
    return true;
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

    volatile uint32_t* inbound_steps =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_inbound_steps(op));

    const uint32_t live_step =
        chunk_scheduler_atomic_load_u32(&inbound_steps[chunk_idx]);

    if (live_step != expected_step) {
        return false;
    }

    const int actor =
        collective::operation_desc_actor_rank_for_step(op, chunk_idx, live_step);

    return actor == op->rank;
}

__device__ __forceinline__ void chunk_scheduler_refill_ready_cache_parallel(
    ChunkScheduler* sched) {
    if (sched == nullptr || sched->operation == nullptr) {
        return;
    }
    if (!chunk_scheduler_ready_empty(sched)) {
        return;
    }

    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;
    if (threadIdx.x >= static_cast<int>(kChunkSchedulerSearchWarp)) {
        return;
    }

    const collective::OperationDesc* op = sched->operation;
    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(op);

    volatile uint32_t* inbound_steps =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_inbound_steps(op));
    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(op);

    uint32_t base = sched->next_search_idx;

    for (uint32_t scanned = 0; scanned < op->num_chunks; scanned += kChunkSchedulerSearchWarp) {
        __syncwarp();

        if (chunk_scheduler_ready_full(sched)) {
            break;
        }

        const bool slot_live = (scanned + lane) < op->num_chunks;
        uint32_t cur = 0;
        uint32_t step = collective::kOperationInboundStepInvalid;
        bool valid = false;

        if (slot_live) {
            cur = (base + lane) % op->num_chunks;

            if (!collective::chunk_state_is_in_flight(&chunk_states[cur])) {
                step = chunk_scheduler_atomic_load_u32(&inbound_steps[cur]);

                if (step != collective::kOperationInboundStepInvalid &&
                    step < total_steps) {
                    const int actor =
                        collective::operation_desc_actor_rank_for_step(op, cur, step);
                    valid = (actor == op->rank);
                }
            }
        }

        sched->search_found[lane].chunk_idx = cur;
        sched->search_found[lane].step =
            valid ? step : collective::kOperationInboundStepInvalid;

        __syncwarp();

        if (threadIdx.x == 0) {
            for (uint32_t src_lane = 0;
                 src_lane < kChunkSchedulerSearchWarp && !chunk_scheduler_ready_full(sched);
                 ++src_lane) {
                const ReadyCandidate cand = sched->search_found[src_lane];
                if (cand.step == collective::kOperationInboundStepInvalid) {
                    continue;
                }
                chunk_scheduler_ready_push(sched, cand.chunk_idx, cand.step);
            }
        }

        __syncwarp();

        base += kChunkSchedulerSearchWarp;
        if (base >= op->num_chunks) {
            base %= op->num_chunks;
        }
    }

    if (threadIdx.x == 0) {
        sched->next_search_idx = base;
    }
}

__device__ __forceinline__ void chunk_scheduler_refill_ready_cache(
    ChunkScheduler* sched) {
    if (sched == nullptr || sched->operation == nullptr) {
        return;
    }
    if (!chunk_scheduler_ready_empty(sched)) {
        return;
    }

    const collective::OperationDesc* op = sched->operation;
    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(op);

    volatile uint32_t* inbound_steps =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_inbound_steps(op));
    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(op);

    uint32_t idx = sched->next_search_idx;

    for (uint32_t scanned = 0;
         scanned < op->num_chunks && !chunk_scheduler_ready_full(sched);
         ++scanned) {

        const uint32_t cur = idx;

        idx++;
        if (idx == op->num_chunks) {
            idx = 0;
        }

        if (collective::chunk_state_is_in_flight(&chunk_states[cur])) {
            continue;
        }

        const uint32_t step =
            chunk_scheduler_atomic_load_u32(&inbound_steps[cur]);

        if (step == collective::kOperationInboundStepInvalid) {
            continue;
        }
        if (step >= total_steps) {
            continue;
        }

        const int actor =
            collective::operation_desc_actor_rank_for_step(op, cur, step);
        if (actor != op->rank) {
            continue;
        }

        chunk_scheduler_ready_push(sched, cur, step);
    }

    sched->next_search_idx = idx;
}

__device__ __forceinline__ bool chunk_scheduler_try_activate_next_chunk(
    ChunkScheduler* sched) {
    if (sched == nullptr || sched->operation == nullptr) {
        return false;
    }

    if (sched->has_active_chunk) {
        return true;
    }

    ReadyCandidate cand{};

    while (true) {
        if (chunk_scheduler_ready_empty(sched)) {
            chunk_scheduler_refill_ready_cache(sched);
        }

        if (!chunk_scheduler_ready_pop(sched, &cand)) {
            return false;
        }

        if (!chunk_scheduler_candidate_is_valid(sched, cand.chunk_idx, cand.step)) {
            continue;
        }

        const collective::OperationDesc* op = sched->operation;
        collective::ChunkState* chunk_states =
            collective::operation_desc_chunk_states(op);

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

__device__ __forceinline__ void chunk_scheduler_retire_current(
    ChunkScheduler* sched) {
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

    chunk_scheduler_atomic_store_u32(
        &local_inbound[chunk_idx],
        collective::kOperationInboundStepInvalid);

    __threadfence_system();

    if (current_step < reduce_steps) {
        if (next_step == reduce_steps) {
            chunk_scheduler_atomic_store_u32(&next_done[chunk_idx], 1u);
        }

        if (next_step < total_steps) {
            chunk_scheduler_atomic_store_u32(&next_inbound[chunk_idx], next_step);
        }
    } else {
        chunk_scheduler_atomic_store_u32(&next_done[chunk_idx], 1u);

        if (next_step < total_steps) {
            chunk_scheduler_atomic_store_u32(&next_inbound[chunk_idx], next_step);
        }
    }

    if (next_step >= total_steps) {
        chunk_scheduler_atomic_store_u32(&local_done[chunk_idx], 1u);
    }

    __threadfence_system();

    chunk_states[chunk_idx].last_step_completed = next_step;
    chunk_states[chunk_idx].flags &= ~collective::kChunkStateFlagInFlight;

    if (chunk_scheduler_atomic_load_u32(&local_done[chunk_idx]) == 1u) {
        chunk_states[chunk_idx].flags |= collective::kChunkStateFlagDone;
    }

    sched->has_active_chunk = false;
    sched->active_chunk_idx = -1;
    sched->active_step = 0;
    chunk_clear(&sched->current);
}

__device__ __forceinline__ void chunk_scheduler_advance(
    ChunkScheduler* sched) {
    chunk_scheduler_retire_current(sched);
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
