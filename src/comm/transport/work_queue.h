#pragma once

#include "comm/exec/chunk.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace transport {

enum class WorkQueueSlotState : int {
    kFree = 0,
    kWriting = 1,
    kReady = 2,
};

template <int Capacity>
struct WorkQueueSlot {
    exec::WorkSpan span{};
    uint64_t ticket = 0;
    int state = static_cast<int>(WorkQueueSlotState::kFree);
};

template <int Capacity>
struct WorkQueue {
    WorkQueueSlot<Capacity> slots[Capacity];
    uint64_t head_ticket = 0;
    uint64_t tail_ticket = 0;
};

template <int Capacity>
__host__ __device__ __forceinline__ void work_queue_reset(
    WorkQueue<Capacity>* queue) {
    queue->head_ticket = 0;
    queue->tail_ticket = 0;
    for (int i = 0; i < Capacity; ++i) {
        work_span_clear(&queue->slots[i].span);
        queue->slots[i].ticket = 0;
        queue->slots[i].state = static_cast<int>(WorkQueueSlotState::kFree);
    }
}

template <int Capacity>
__device__ __forceinline__ uint64_t work_queue_push_blocking(
    WorkQueue<Capacity>* queue,
    const exec::WorkSpan* span) {
    const uint64_t ticket =
        static_cast<uint64_t>(atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail_ticket), 1ULL));

    WorkQueueSlot<Capacity>* slot =
        &queue->slots[static_cast<int>(ticket % static_cast<uint64_t>(Capacity))];

    while (atomicCAS(
        &slot->state,
        static_cast<int>(WorkQueueSlotState::kFree),
        static_cast<int>(WorkQueueSlotState::kWriting)) !=
            static_cast<int>(WorkQueueSlotState::kFree)) {
    }

    slot->span = *span;
    slot->ticket = ticket;
    __threadfence();
    atomicExch(&slot->state, static_cast<int>(WorkQueueSlotState::kReady));
    return ticket;
}

template <int Capacity>
__device__ __forceinline__ bool work_queue_try_peek_ticket(
    const WorkQueue<Capacity>* queue,
    uint64_t ticket,
    exec::WorkSpan* out) {
    if (out == nullptr) {
        return false;
    }
    work_span_clear(out);

    const WorkQueueSlot<Capacity>* slot =
        &queue->slots[static_cast<int>(ticket % static_cast<uint64_t>(Capacity))];

    const int state = *((volatile const int*)&slot->state);
    const uint64_t seen_ticket = *((volatile const uint64_t*)&slot->ticket);

    if (state != static_cast<int>(WorkQueueSlotState::kReady)) {
        return false;
    }
    if (seen_ticket != ticket) {
        return false;
    }

    *out = slot->span;
    return work_span_is_valid(out);
}

template <int Capacity>
__device__ __forceinline__ bool work_queue_try_peek_head(
    const WorkQueue<Capacity>* queue,
    exec::WorkSpan* out,
    uint64_t* ticket_out) {
    if (queue == nullptr || out == nullptr || ticket_out == nullptr) {
        return false;
    }

    const uint64_t ticket = *((volatile const uint64_t*)&queue->head_ticket);
    if (!work_queue_try_peek_ticket(queue, ticket, out)) {
        return false;
    }

    *ticket_out = ticket;
    return true;
}

template <int Capacity>
__device__ __forceinline__ void work_queue_release_head(
    WorkQueue<Capacity>* queue,
    uint64_t ticket) {
    WorkQueueSlot<Capacity>* slot =
        &queue->slots[static_cast<int>(ticket % static_cast<uint64_t>(Capacity))];

    work_span_clear(&slot->span);
    slot->ticket = 0;
    __threadfence();
    atomicExch(&slot->state, static_cast<int>(WorkQueueSlotState::kFree));

    queue->head_ticket = ticket + 1;
}

} // namespace transport
} // namespace comm
} // namespace ooverlap
