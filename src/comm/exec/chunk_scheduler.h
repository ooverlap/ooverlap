#pragma once

#include "comm/collective/operation.h"
#include "comm/collective/published_tile.h"
#include "comm/exec/chunk.h"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {

struct SchedulerQueueBinding {
    // Producer -> persistent-kernel mailbox on this GPU.
    collective::ReadyTileQueue* queue = nullptr;

    // Policy / routing / destination meaning for this queue.
    collective::OperationDesc* operation = nullptr;
};

template <int WatchThreads>
struct ChunkSchedulerScratch {
    int candidate_binding_idx[WatchThreads];
    int selected_binding_idx = -1;
};

template <int WatchThreads = 16>
struct ChunkScheduler {
    SchedulerQueueBinding* bindings = nullptr;
    int num_bindings = 0;
    size_t chunk_bytes = 0;

    // Round-robin start for fairness across queues.
    int next_binding_start = 0;

    bool has_active_chunk = false;
    int active_binding_idx = -1;
    uint64_t active_start_ticket = 0;
    int active_num_tiles = 0;

    Chunk current{};

    ChunkSchedulerScratch<WatchThreads>* scratch = nullptr;
};

template <int WatchThreads>
__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler<WatchThreads>* sched) {
    sched->bindings = nullptr;
    sched->num_bindings = 0;
    sched->chunk_bytes = 0;
    sched->next_binding_start = 0;

    sched->has_active_chunk = false;
    sched->active_binding_idx = -1;
    sched->active_start_ticket = 0;
    sched->active_num_tiles = 0;

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
    SchedulerQueueBinding* bindings,
    int num_bindings,
    size_t chunk_bytes,
    ChunkSchedulerScratch<WatchThreads>* scratch = nullptr) {
    sched->bindings = bindings;
    sched->num_bindings = num_bindings;
    sched->chunk_bytes = chunk_bytes;
    sched->next_binding_start = 0;

    sched->has_active_chunk = false;
    sched->active_binding_idx = -1;
    sched->active_start_ticket = 0;
    sched->active_num_tiles = 0;

    chunk_clear(&sched->current);
    sched->scratch = scratch;
}

__host__ __device__ __forceinline__ bool scheduler_queue_binding_is_valid(
    const SchedulerQueueBinding* binding) {
    return binding != nullptr &&
           binding->queue != nullptr &&
           binding->operation != nullptr &&
           collective::ready_tile_queue_is_configured(binding->queue) &&
           collective::operation_desc_is_active(binding->operation);
}

__device__ __forceinline__ bool chunk_scheduler_fail_invalid_mapping() {
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
#endif
    return false;
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
__host__ __device__ __forceinline__ uint32_t chunk_scheduler_active_queue_id(
    const ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr ||
        !sched->has_active_chunk ||
        sched->bindings == nullptr ||
        sched->active_binding_idx < 0 ||
        sched->active_binding_idx >= sched->num_bindings) {
        return 0;
    }
    return sched->bindings[sched->active_binding_idx].operation->queue_id;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ int chunk_scheduler_active_dst_rank(
    const ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr ||
        !sched->has_active_chunk ||
        sched->bindings == nullptr ||
        sched->active_binding_idx < 0 ||
        sched->active_binding_idx >= sched->num_bindings) {
        return -1;
    }
    return sched->bindings[sched->active_binding_idx].operation->dst_rank;
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_try_build_chunk_from_binding(
    const ChunkScheduler<WatchThreads>* sched,
    int binding_idx,
    uint64_t start_ticket,
    Chunk* out,
    int* out_num_tiles) {
    if (out == nullptr || out_num_tiles == nullptr) {
        return false;
    }
    chunk_clear(out);
    *out_num_tiles = 0;

    if (sched == nullptr ||
        sched->bindings == nullptr ||
        sched->chunk_bytes == 0 ||
        binding_idx < 0 ||
        binding_idx >= sched->num_bindings) {
        return false;
    }

    const SchedulerQueueBinding& binding = sched->bindings[binding_idx];
    if (!scheduler_queue_binding_is_valid(&binding)) {
        return false;
    }

    const collective::ReadyTileQueue& queue = *binding.queue;
    const collective::OperationDesc& op_desc = *binding.operation;

    collective::ReadyTile tile{};
    if (!collective::device_ready_tile_queue_try_peek_ticket(
            &queue,
            start_ticket,
            &tile)) {
        return false;
    }

    // Temporary invariant:
    // one tile per chunk until we support per-tile destination offsets
    // inside a multi-tile chunk.
    if (tile.bytes > sched->chunk_bytes) {
        return chunk_scheduler_fail_invalid_mapping();
    }

    unsigned char* mapped_dst = nullptr;
    size_t mapped_dst_offset = 0;
    if (!collective::operation_desc_try_map_tile(
            &op_desc,
            tile.tile_id,
            static_cast<size_t>(tile.bytes),
            &mapped_dst,
            &mapped_dst_offset)) {
        return chunk_scheduler_fail_invalid_mapping();
    }

    ChunkTileSpan* span = &out->tile_spans[0];
    span->src = collective::ready_tile_src_bytes(&tile);
    span->bytes = static_cast<size_t>(tile.bytes);
    span->dst_offset_bytes = 0;
    span->tile_id = tile.tile_id;
    span->queue_ticket = tile.queue_ticket;
    span->dim0 = tile.dim0;
    span->dim1 = tile.dim1;
    span->dim2 = tile.dim2;

    out->dst = mapped_dst;
    out->bytes = static_cast<size_t>(tile.bytes);
    out->queue_id = op_desc.queue_id;
    out->dst_rank = op_desc.dst_rank;
    out->span_ticket = start_ticket;
    out->user_tag = tile.tile_id;
    out->chunk_idx = 0;
    out->span_offset_bytes = mapped_dst_offset;
    out->op = op_desc.op;
    out->num_tile_spans = 1;

    *out_num_tiles = 1;
    return true;
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_binding_has_schedulable_work(
    const ChunkScheduler<WatchThreads>* sched,
    int binding_idx) {
    if (sched == nullptr ||
        sched->bindings == nullptr ||
        binding_idx < 0 ||
        binding_idx >= sched->num_bindings) {
        return false;
    }

    const SchedulerQueueBinding& binding = sched->bindings[binding_idx];
    if (!scheduler_queue_binding_is_valid(&binding)) {
        return false;
    }

    collective::ReadyTile tile{};
    if (!collective::device_ready_tile_queue_try_peek_head(binding.queue, &tile)) {
        return false;
    }

    if (tile.bytes > sched->chunk_bytes) {
        return chunk_scheduler_fail_invalid_mapping();
    }

    unsigned char* mapped_dst = nullptr;
    size_t mapped_dst_offset = 0;
    if (!collective::operation_desc_try_map_tile(
            binding.operation,
            tile.tile_id,
            static_cast<size_t>(tile.bytes),
            &mapped_dst,
            &mapped_dst_offset)) {
        return chunk_scheduler_fail_invalid_mapping();
    }

    (void)mapped_dst;
    (void)mapped_dst_offset;
    return true;
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_try_activate_next_chunk_sequential(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched->has_active_chunk) {
        return true;
    }
    if (sched->bindings == nullptr || sched->num_bindings <= 0 || sched->chunk_bytes == 0) {
        return false;
    }

    for (int step = 0; step < sched->num_bindings; ++step) {
        const int binding_idx =
            (sched->next_binding_start + step) % sched->num_bindings;

        if (!chunk_scheduler_binding_has_schedulable_work(sched, binding_idx)) {
            continue;
        }

        const uint64_t start_ticket =
            *(sched->bindings[binding_idx].queue->head);

        Chunk built{};
        int built_num_tiles = 0;
        if (!chunk_scheduler_try_build_chunk_from_binding(
                sched,
                binding_idx,
                start_ticket,
                &built,
                &built_num_tiles)) {
            continue;
        }

        sched->has_active_chunk = true;
        sched->active_binding_idx = binding_idx;
        sched->active_start_ticket = start_ticket;
        sched->active_num_tiles = built_num_tiles;
        sched->current = built;
        sched->next_binding_start = (binding_idx + 1) % sched->num_bindings;
        return true;
    }

    return false;
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_try_activate_next_chunk(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched->has_active_chunk) {
        return true;
    }
    if (sched->bindings == nullptr || sched->num_bindings <= 0 || sched->chunk_bytes == 0) {
        return false;
    }

    if (sched->scratch == nullptr) {
        return chunk_scheduler_try_activate_next_chunk_sequential(sched);
    }

    ChunkSchedulerScratch<WatchThreads>* scratch = sched->scratch;
    const int active_watchers =
        (static_cast<int>(blockDim.x) < WatchThreads)
            ? static_cast<int>(blockDim.x)
            : WatchThreads;
    const int lane = static_cast<int>(threadIdx.x);

    if (lane == 0) {
        for (int i = 0; i < WatchThreads; ++i) {
            scratch->candidate_binding_idx[i] = -1;
        }
        scratch->selected_binding_idx = -1;
    }
    __syncthreads();

    if (lane < active_watchers) {
        for (int step = lane; step < sched->num_bindings; step += active_watchers) {
            const int binding_idx =
                (sched->next_binding_start + step) % sched->num_bindings;

            if (chunk_scheduler_binding_has_schedulable_work(sched, binding_idx)) {
                scratch->candidate_binding_idx[lane] = binding_idx;
                break;
            }
        }
    }
    __syncthreads();

    if (lane == 0) {
        int best_binding_idx = -1;
        int best_distance = sched->num_bindings + 1;

        for (int i = 0; i < active_watchers; ++i) {
            const int binding_idx = scratch->candidate_binding_idx[i];
            if (binding_idx < 0) {
                continue;
            }

            const int distance =
                (binding_idx - sched->next_binding_start + sched->num_bindings) %
                sched->num_bindings;

            if (distance < best_distance) {
                best_distance = distance;
                best_binding_idx = binding_idx;
            }
        }

        scratch->selected_binding_idx = best_binding_idx;
    }
    __syncthreads();

    if (scratch->selected_binding_idx < 0) {
        return false;
    }

    const int binding_idx = scratch->selected_binding_idx;
    const uint64_t start_ticket =
        *(sched->bindings[binding_idx].queue->head);

    Chunk built{};
    int built_num_tiles = 0;
    if (!chunk_scheduler_try_build_chunk_from_binding(
            sched,
            binding_idx,
            start_ticket,
            &built,
            &built_num_tiles)) {
        return false;
    }

    sched->has_active_chunk = true;
    sched->active_binding_idx = binding_idx;
    sched->active_start_ticket = start_ticket;
    sched->active_num_tiles = built_num_tiles;
    sched->current = built;
    sched->next_binding_start = (binding_idx + 1) % sched->num_bindings;
    return true;
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_try_prime_current(
    ChunkScheduler<WatchThreads>* sched) {
    if (chunk_is_valid(&sched->current)) {
        return true;
    }

    return chunk_scheduler_try_activate_next_chunk(sched);
}

template <int WatchThreads>
__device__ __forceinline__ bool chunk_scheduler_peek_next(
    const ChunkScheduler<WatchThreads>* sched,
    Chunk* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_clear(out);

    if (sched == nullptr ||
        !sched->has_active_chunk ||
        sched->bindings == nullptr ||
        sched->active_binding_idx < 0 ||
        sched->active_binding_idx >= sched->num_bindings) {
        return false;
    }

    const uint64_t next_start_ticket =
        sched->active_start_ticket + static_cast<uint64_t>(sched->active_num_tiles);

    int next_num_tiles = 0;
    return chunk_scheduler_try_build_chunk_from_binding(
        sched,
        sched->active_binding_idx,
        next_start_ticket,
        out,
        &next_num_tiles);
}

template <int WatchThreads>
__device__ __forceinline__ void chunk_scheduler_advance(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr || !sched->has_active_chunk) {
        chunk_clear(&sched->current);
        return;
    }

    SchedulerQueueBinding& binding = sched->bindings[sched->active_binding_idx];

    const uint64_t new_head =
        sched->active_start_ticket + static_cast<uint64_t>(sched->active_num_tiles);

    *binding.queue->head = new_head;
    __threadfence();

    sched->has_active_chunk = false;
    sched->active_binding_idx = -1;
    sched->active_start_ticket = 0;
    sched->active_num_tiles = 0;
    chunk_clear(&sched->current);

    chunk_scheduler_try_activate_next_chunk(sched);
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
