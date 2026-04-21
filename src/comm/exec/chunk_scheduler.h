#pragma once

#include "comm/transport/work_queue.h"
#include "comm/exec/chunk.h"

namespace ooverlap {
namespace comm {
namespace exec {

template <int QueueCapacity>
struct ChunkScheduler {
    transport::WorkQueue<QueueCapacity>* queue = nullptr;
    size_t chunk_bytes = 0;

    bool has_active_span = false;
    WorkSpan active_span{};
    uint64_t active_ticket = 0;

    size_t active_offset_bytes = 0;
    int active_chunk_idx = 0;

    Chunk current{};
};

template <int QueueCapacity>
__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler<QueueCapacity>* sched) {
    sched->queue = nullptr;
    sched->chunk_bytes = 0;
    sched->has_active_span = false;
    work_span_clear(&sched->active_span);
    sched->active_ticket = 0;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = 0;
    chunk_clear(&sched->current);
}

template <int QueueCapacity>
__host__ __device__ __forceinline__ void chunk_scheduler_init(
    ChunkScheduler<QueueCapacity>* sched,
    transport::WorkQueue<QueueCapacity>* queue,
    size_t chunk_bytes) {
    sched->queue = queue;
    sched->chunk_bytes = chunk_bytes;
    sched->has_active_span = false;
    work_span_clear(&sched->active_span);
    sched->active_ticket = 0;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = 0;
    chunk_clear(&sched->current);
}

template <int QueueCapacity>
__device__ __forceinline__ bool chunk_scheduler_try_activate_head_span(
    ChunkScheduler<QueueCapacity>* sched) {
    if (sched->has_active_span) {
        return true;
    }
    if (sched->queue == nullptr) {
        return false;
    }

    WorkSpan span{};
    uint64_t ticket = 0;
    if (!work_queue_try_peek_head(sched->queue, &span, &ticket)) {
        return false;
    }

    sched->has_active_span = true;
    sched->active_span = span;
    sched->active_ticket = ticket;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = 0;
    return true;
}

template <int QueueCapacity>
__device__ __forceinline__ bool chunk_scheduler_try_prime_current(
    ChunkScheduler<QueueCapacity>* sched) {
    if (chunk_is_valid(&sched->current)) {
        return true;
    }
    if (!chunk_scheduler_try_activate_head_span(sched)) {
        return false;
    }

    return chunk_make_from_span(
        &sched->active_span,
        sched->active_ticket,
        sched->active_offset_bytes,
        sched->active_chunk_idx,
        sched->chunk_bytes,
        &sched->current);
}

template <int QueueCapacity>
__device__ __forceinline__ bool chunk_scheduler_has_current(
    const ChunkScheduler<QueueCapacity>* sched) {
    return chunk_is_valid(&sched->current);
}

template <int QueueCapacity>
__device__ __forceinline__ const Chunk* chunk_scheduler_current(
    const ChunkScheduler<QueueCapacity>* sched) {
    return &sched->current;
}

template <int QueueCapacity>
__device__ __forceinline__ bool chunk_scheduler_peek_next(
    const ChunkScheduler<QueueCapacity>* sched,
    Chunk* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_clear(out);

    if (!chunk_is_valid(&sched->current) || !sched->has_active_span) {
        return false;
    }

    const size_t next_offset =
        sched->active_offset_bytes + sched->current.bytes;
    const int next_chunk_idx =
        sched->active_chunk_idx + 1;

    if (next_offset < sched->active_span.total_bytes) {
        return chunk_make_from_span(
            &sched->active_span,
            sched->active_ticket,
            next_offset,
            next_chunk_idx,
            sched->chunk_bytes,
            out);
    }

    if (sched->queue == nullptr) {
        return false;
    }

    WorkSpan next_span{};
    const uint64_t next_ticket = sched->active_ticket + 1;
    if (!work_queue_try_peek_ticket(sched->queue, next_ticket, &next_span)) {
        return false;
    }

    return chunk_make_from_span(
        &next_span,
        next_ticket,
        0,
        0,
        sched->chunk_bytes,
        out);
}

template <int QueueCapacity>
__device__ __forceinline__ void chunk_scheduler_advance(
    ChunkScheduler<QueueCapacity>* sched) {
    if (!chunk_is_valid(&sched->current) || !sched->has_active_span) {
        chunk_clear(&sched->current);
        return;
    }

    const size_t next_offset =
        sched->active_offset_bytes + sched->current.bytes;
    const int next_chunk_idx =
        sched->active_chunk_idx + 1;

    if (next_offset < sched->active_span.total_bytes) {
        sched->active_offset_bytes = next_offset;
        sched->active_chunk_idx = next_chunk_idx;
        chunk_make_from_span(
            &sched->active_span,
            sched->active_ticket,
            sched->active_offset_bytes,
            sched->active_chunk_idx,
            sched->chunk_bytes,
            &sched->current);
        return;
    }

    work_queue_release_head(sched->queue, sched->active_ticket);

    sched->has_active_span = false;
    work_span_clear(&sched->active_span);
    sched->active_ticket = 0;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = 0;
    chunk_clear(&sched->current);

    chunk_scheduler_try_prime_current(sched);
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
