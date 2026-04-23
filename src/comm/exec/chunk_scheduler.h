#pragma once

#include "comm/collective/operation.h"
#include "comm/exec/chunk.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifndef OOVERLAP_ENDPOINT_DEBUG
#define OOVERLAP_ENDPOINT_DEBUG 0
#endif

#ifndef OOVERLAP_ENDPOINT_TRACK_CHUNK_STATE
#define OOVERLAP_ENDPOINT_TRACK_CHUNK_STATE 0
#endif

#ifndef OOVERLAP_ENDPOINT_DEBUG_WAIT_PRINT_EVERY
#define OOVERLAP_ENDPOINT_DEBUG_WAIT_PRINT_EVERY 65536u
#endif

#if OOVERLAP_ENDPOINT_DEBUG
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

    uint32_t step_cursor = 0;
    uint32_t next_search_idx = 0;

    uint32_t ready_count = 0;

    // Debug state for blocked progress waits.
    uint32_t blocked_spins = 0;
    int blocked_chunk_idx = -1;
    uint32_t blocked_step = 0;
    uint32_t blocked_prev_progress = 0;

    Chunk current{};
};

__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler* sched) {
    sched->operation = nullptr;
    sched->has_active_chunk = false;
    sched->active_chunk_idx = -1;
    sched->active_step = 0;
    sched->step_cursor = 0;
    sched->next_search_idx = 0;
    sched->ready_count = 0;
    sched->blocked_spins = 0;
    sched->blocked_chunk_idx = -1;
    sched->blocked_step = 0;
    sched->blocked_prev_progress = 0;
    chunk_clear(&sched->current);
}

__device__ __forceinline__ void chunk_scheduler_init_operation(
    ChunkScheduler* sched,
    const collective::OperationDesc* operation) {
    chunk_scheduler_reset(sched);
    sched->operation = operation;

#if OOVERLAP_ENDPOINT_DEBUG
    if (threadIdx.x == 0 && operation != nullptr) {
        OOVERLAP_SCHED_DBG(
            "[sched-init] rank=%d world=%d op_id=%u num_chunks=%u total_bytes=%zu chunk_bytes=%zu total_steps=%u accum=0x%llx next_accum=0x%llx local_progress=0x%llx prev_progress=0x%llx completion_target=%u\n",
            operation->rank,
            operation->world_size,
            operation->op_id,
            operation->num_chunks,
            operation->total_bytes,
            operation->chunk_bytes,
            collective::operation_desc_total_ring_steps(operation),
            static_cast<unsigned long long>(operation->accum_ptr),
            static_cast<unsigned long long>(operation->next_accum_ptr),
            static_cast<unsigned long long>(operation->done_ptr),
            static_cast<unsigned long long>(operation->next_done_ptr),
            operation->completion_target);
    }
#endif
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

__device__ __forceinline__ uint32_t chunk_scheduler_volatile_load_u32(
    const volatile uint32_t* ptr) {
    return *ptr;
}

__device__ __forceinline__ void chunk_scheduler_store_release_u32(
    volatile uint32_t* ptr,
    uint32_t value) {
    *const_cast<uint32_t*>(ptr) = value;
    __threadfence_system();
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

__device__ __forceinline__ uint32_t chunk_scheduler_count_remaining_local_chunks_for_step(
    const collective::OperationDesc* op,
    uint32_t step,
    uint32_t begin_idx) {
    if (op == nullptr || begin_idx >= op->num_chunks) {
        return 0u;
    }

    uint32_t count = 0u;
    for (uint32_t idx = begin_idx; idx < op->num_chunks; ++idx) {
        if (collective::operation_desc_actor_rank_for_step(op, idx, step) == op->rank) {
            ++count;
        }
    }
    return count;
}

__device__ __forceinline__ void chunk_scheduler_refill_ready_cache(
    ChunkScheduler* sched) {
    if (sched == nullptr || sched->operation == nullptr || threadIdx.x != 0) {
        return;
    }

    const collective::OperationDesc* op = sched->operation;
    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(op);

    if (sched->step_cursor >= total_steps) {
        sched->ready_count = 0u;
        return;
    }

    sched->ready_count = chunk_scheduler_count_remaining_local_chunks_for_step(
        op,
        sched->step_cursor,
        sched->next_search_idx);
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
    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(op);

    while (sched->step_cursor < total_steps) {
        while (sched->next_search_idx < op->num_chunks) {
            const uint32_t idx = sched->next_search_idx;
            const int actor =
                collective::operation_desc_actor_rank_for_step(
                    op, idx, sched->step_cursor);

            if (actor != op->rank) {
                ++sched->next_search_idx;
                continue;
            }

            sched->ready_count =
                chunk_scheduler_count_remaining_local_chunks_for_step(
                    op,
                    sched->step_cursor,
                    idx);

            if (sched->step_cursor > 0u) {
                const volatile uint32_t* prev_progress =
                    reinterpret_cast<volatile uint32_t*>(
                        collective::operation_desc_prev_progress(op));

                const uint32_t prev_value =
                    chunk_scheduler_volatile_load_u32(&prev_progress[idx]);

                if (!collective::operation_desc_step_ready_from_prev(
                        op,
                        idx,
                        sched->step_cursor,
                        prev_value)) {
                    sched->blocked_chunk_idx = static_cast<int>(idx);
                    sched->blocked_step = sched->step_cursor;
                    sched->blocked_prev_progress = prev_value;
                    ++sched->blocked_spins;

#if OOVERLAP_ENDPOINT_DEBUG
                    if (sched->blocked_spins <= 8u ||
                        (sched->blocked_spins % OOVERLAP_ENDPOINT_DEBUG_WAIT_PRINT_EVERY) == 0u) {
                        const volatile uint32_t* local_progress =
                            reinterpret_cast<volatile uint32_t*>(
                                collective::operation_desc_local_progress(op));

                        const uint32_t local_value =
                            chunk_scheduler_volatile_load_u32(&local_progress[idx]);

                        OOVERLAP_SCHED_DBG(
                            "[wait] rank=%d chunk=%u step=%u prev_progress=%u local_progress=%u blocked_spins=%u step_cursor=%u next_search_idx=%u ready_count=%u prev_progress_ptr=0x%llx local_progress_ptr=0x%llx\n",
                            op->rank,
                            idx,
                            sched->step_cursor,
                            prev_value,
                            local_value,
                            sched->blocked_spins,
                            sched->step_cursor,
                            sched->next_search_idx,
                            sched->ready_count,
                            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&prev_progress[idx])),
                            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&local_progress[idx])));
                    }
#endif
                    return false;
                }
            }

            if (sched->blocked_chunk_idx == static_cast<int>(idx) &&
                sched->blocked_step == sched->step_cursor &&
                sched->blocked_spins > 0u) {
#if OOVERLAP_ENDPOINT_DEBUG
                OOVERLAP_SCHED_DBG(
                    "[unblock] rank=%d chunk=%u step=%u prev_progress=%u blocked_spins=%u\n",
                    op->rank,
                    idx,
                    sched->step_cursor,
                    sched->blocked_prev_progress,
                    sched->blocked_spins);
#endif
                sched->blocked_spins = 0u;
                sched->blocked_chunk_idx = -1;
                sched->blocked_step = 0u;
                sched->blocked_prev_progress = 0u;
            }

            chunk_scheduler_build_operation_chunk(
                op,
                idx,
                sched->step_cursor,
                &sched->current);

            sched->has_active_chunk = true;
            sched->active_chunk_idx = static_cast<int>(idx);
            sched->active_step = sched->step_cursor;
            ++sched->next_search_idx;

#if OOVERLAP_ENDPOINT_TRACK_CHUNK_STATE
            collective::ChunkState* chunk_states =
                collective::operation_desc_chunk_states(op);
            collective::ChunkState* st = &chunk_states[idx];

            if (!collective::chunk_state_is_initialized(st)) {
                collective::chunk_state_clear(st);
                st->chunk_idx = idx;
                st->offset_bytes =
                    collective::operation_desc_chunk_offset_bytes(op, idx);
                st->bytes =
                    collective::operation_desc_chunk_bytes_at(op, idx);
                st->flags |= collective::kChunkStateFlagInitialized;
            }

            st->last_step_started = sched->step_cursor;
            st->flags |= collective::kChunkStateFlagInFlight;
#endif

#if OOVERLAP_ENDPOINT_DEBUG
            {
                uint32_t prev_value = 0u;
                if (sched->step_cursor > 0u) {
                    const volatile uint32_t* prev_progress =
                        reinterpret_cast<volatile uint32_t*>(
                            collective::operation_desc_prev_progress(op));
                    prev_value = chunk_scheduler_volatile_load_u32(&prev_progress[idx]);
                }

                OOVERLAP_SCHED_DBG(
                    "[activate] rank=%d chunk=%u step=%u next_search_idx=%u remaining=%u prev_progress=%u bytes=%zu src=0x%llx dst=0x%llx op=%d\n",
                    op->rank,
                    idx,
                    sched->step_cursor,
                    sched->next_search_idx,
                    sched->ready_count,
                    prev_value,
                    sched->current.bytes,
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(sched->current.src)),
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(sched->current.dst)),
                    static_cast<int>(sched->current.op));
            }
#endif
            return true;
        }

#if OOVERLAP_ENDPOINT_DEBUG
        OOVERLAP_SCHED_DBG(
            "[advance-step] rank=%d old_step=%u -> new_step=%u\n",
            op->rank,
            sched->step_cursor,
            sched->step_cursor + 1u);
#endif

        ++sched->step_cursor;
        sched->next_search_idx = 0u;
        sched->ready_count = 0u;
    }

#if OOVERLAP_ENDPOINT_DEBUG
    OOVERLAP_SCHED_DBG(
        "[drain] rank=%d no-more-work total_steps=%u\n",
        op->rank,
        total_steps);
#endif

    return false;
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
    sched->active_step = 0u;
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
    const uint32_t next_progress = current_step + 1u;

    volatile uint32_t* local_progress =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_progress(op));

    chunk_scheduler_store_release_u32(&local_progress[chunk_idx], next_progress);

#if OOVERLAP_ENDPOINT_DEBUG
    OOVERLAP_SCHED_DBG(
        "[retire] rank=%d chunk=%u cur_step=%u published_progress=%u total_steps=%u local_progress_ptr=0x%llx\n",
        op->rank,
        chunk_idx,
        current_step,
        next_progress,
        total_steps,
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&local_progress[chunk_idx])));
#endif

    if (next_progress >= total_steps) {
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

#if OOVERLAP_ENDPOINT_DEBUG
            OOVERLAP_SCHED_DBG(
                "[complete-count] rank=%d chunk=%u completed=%u target=%u completion_count_ptr=0x%llx completion_flag_ptr=0x%llx\n",
                op->rank,
                chunk_idx,
                completed,
                op->completion_target,
                static_cast<unsigned long long>(op->completion_count_ptr),
                static_cast<unsigned long long>(op->completion_flag_ptr));
#endif

            if (completed >= op->completion_target) {
                __threadfence_system();
                *const_cast<uint32_t*>(completion_flag) = 1u;
                __threadfence_system();

#if OOVERLAP_ENDPOINT_DEBUG
                OOVERLAP_SCHED_DBG(
                    "[complete-flag] rank=%d chunk=%u completion_flag=1\n",
                    op->rank,
                    chunk_idx);
#endif
            }
        }
    }

#if OOVERLAP_ENDPOINT_TRACK_CHUNK_STATE
    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(op);
    collective::ChunkState* st = &chunk_states[chunk_idx];

    st->last_step_completed = next_progress;
    st->flags &= ~collective::kChunkStateFlagInFlight;

    if (next_progress >= total_steps) {
        st->flags |= collective::kChunkStateFlagDone;
    }
#endif

    sched->ready_count = 0u;
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
