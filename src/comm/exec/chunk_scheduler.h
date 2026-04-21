#pragma once

#include "comm/transport/dispatch_queue.h"
#include "comm/exec/chunk.h"

namespace ooverlap {
namespace comm {
namespace exec {

template <int QueueCapacity>
struct ChunkScheduler {
    transport::DeviceDispatchQueueHandle queue{};

    bool has_active_record = false;
    transport::DispatchRecord active_record{};
    uint64_t active_ticket = 0;

    Chunk current{};
};

template <int QueueCapacity>
__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler<QueueCapacity>* sched) {
    sched->queue = transport::DeviceDispatchQueueHandle{};
    sched->has_active_record = false;
    transport::dispatch_record_clear(&sched->active_record);
    sched->active_ticket = 0;
    chunk_clear(&sched->current);
}

template <int QueueCapacity>
__host__ __device__ __forceinline__ void chunk_scheduler_init(
    ChunkScheduler<QueueCapacity>* sched,
    const transport::DeviceDispatchQueueHandle* queue) {
    sched->queue = (queue != nullptr) ? *queue : transport::DeviceDispatchQueueHandle{};
    sched->has_active_record = false;
    transport::dispatch_record_clear(&sched->active_record);
    sched->active_ticket = 0;
    chunk_clear(&sched->current);
}

template <int QueueCapacity>
__host__ __device__ __forceinline__ void chunk_scheduler_init(
    ChunkScheduler<QueueCapacity>* sched,
    const transport::DeviceDispatchQueueHandle* queue,
    size_t /*chunk_bytes_ignored*/) {
    chunk_scheduler_init(sched, queue);
}

template <int QueueCapacity>
__device__ __forceinline__ bool chunk_scheduler_try_activate_head_record(
    ChunkScheduler<QueueCapacity>* sched) {
    if (sched->has_active_record) {
        return true;
    }

    transport::DispatchRecord rec{};
    if (!transport::device_dispatch_queue_try_peek_head(&sched->queue, &rec)) {
        return false;
    }

    sched->has_active_record = true;
    sched->active_record = rec;
    sched->active_ticket = rec.queue_ticket;
    return true;
}

template <int QueueCapacity>
__device__ __forceinline__ bool chunk_scheduler_try_prime_current(
    ChunkScheduler<QueueCapacity>* sched) {
    if (chunk_is_valid(&sched->current)) {
        return true;
    }
    if (!chunk_scheduler_try_activate_head_record(sched)) {
        return false;
    }

    transport::dispatch_record_to_exec_chunk(
        &sched->active_record,
        &sched->current);

    return chunk_is_valid(&sched->current);
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

    if (!chunk_is_valid(&sched->current) || !sched->has_active_record) {
        return false;
    }

    transport::DispatchRecord rec{};
    if (!transport::device_dispatch_queue_try_peek_ticket(
            &sched->queue,
            sched->active_ticket + 1,
            &rec)) {
        return false;
    }

    transport::dispatch_record_to_exec_chunk(&rec, out);
    return chunk_is_valid(out);
}

template <int QueueCapacity>
__device__ __forceinline__ void chunk_scheduler_advance(
    ChunkScheduler<QueueCapacity>* sched) {
    if (!sched->has_active_record) {
        chunk_clear(&sched->current);
        return;
    }

    transport::device_dispatch_queue_release_head(&sched->queue);

    sched->has_active_record = false;
    transport::dispatch_record_clear(&sched->active_record);
    sched->active_ticket = 0;
    chunk_clear(&sched->current);

    chunk_scheduler_try_prime_current(sched);
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
