#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/exec/chunk.h"
#include "comm/transport/buffer.h"

namespace ooverlap {
namespace comm {
namespace transport {

enum : uint32_t {
    kDispatchRecordFlagReady = 1u << 0,
    kDispatchRecordFlagConsumed = 1u << 1,
};

struct DispatchRecord {
    uint64_t queue_ticket = 0;

    uint64_t op_id = 0;
    uint64_t publish_ticket = 0;
    uint64_t logical_dst_offset_bytes = 0;

    uint64_t src_ptr = 0;
    uint64_t dst_ptr = 0;

    uint32_t tile_id = 0;
    uint32_t bytes = 0;

    uint16_t src_rank = 0;
    uint16_t dst_rank = 0;

    uint16_t chunk_idx = 0;
    uint16_t num_chunks = 0;

    uint8_t op_kind = static_cast<uint8_t>(exec::ChunkOpKind::kInvalid);
    uint8_t reserved0 = 0;
    uint16_t reserved1 = 0;

    uint32_t flags = 0;
};

struct DispatchQueue {
    CommBuffer records_buffer;
    CommBuffer head_buffer;      // uint64_t
    CommBuffer tail_buffer;      // uint64_t
    CommBuffer overflow_buffer;  // uint32_t

    uint32_t capacity = 0;
    int owner_rank = -1;
    int src_rank = -1;
    int dst_rank = -1;
};

struct DeviceDispatchQueueHandle {
    DispatchRecord* records = nullptr;
    uint64_t* head = nullptr;
    uint64_t* tail = nullptr;
    uint32_t* overflow = nullptr;

    uint32_t capacity = 0;
    int owner_rank = -1;
    int src_rank = -1;
    int dst_rank = -1;
};

__host__ __device__ __forceinline__ void dispatch_record_clear(
    DispatchRecord* rec) {
    rec->queue_ticket = 0;
    rec->op_id = 0;
    rec->publish_ticket = 0;
    rec->logical_dst_offset_bytes = 0;
    rec->src_ptr = 0;
    rec->dst_ptr = 0;
    rec->tile_id = 0;
    rec->bytes = 0;
    rec->src_rank = 0;
    rec->dst_rank = 0;
    rec->chunk_idx = 0;
    rec->num_chunks = 0;
    rec->op_kind = static_cast<uint8_t>(exec::ChunkOpKind::kInvalid);
    rec->reserved0 = 0;
    rec->reserved1 = 0;
    rec->flags = 0;
}

__host__ __device__ __forceinline__ bool dispatch_record_is_ready(
    const DispatchRecord* rec) {
    return rec != nullptr &&
           (rec->flags & kDispatchRecordFlagReady) != 0u;
}

__host__ __device__ __forceinline__ bool dispatch_record_is_valid(
    const DispatchRecord* rec) {
    return rec != nullptr &&
           rec->op_id != 0 &&
           rec->src_ptr != 0 &&
           rec->dst_ptr != 0 &&
           rec->bytes > 0 &&
           rec->num_chunks > 0 &&
           dispatch_record_is_ready(rec) &&
           static_cast<exec::ChunkOpKind>(rec->op_kind) != exec::ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool dispatch_queue_is_configured(
    const DispatchQueue* q) {
    return q != nullptr &&
           q->records_buffer.ptr != nullptr &&
           q->head_buffer.ptr != nullptr &&
           q->tail_buffer.ptr != nullptr &&
           q->overflow_buffer.ptr != nullptr &&
           q->capacity > 0 &&
           q->owner_rank >= 0;
}

inline DispatchRecord* dispatch_queue_records(
    DispatchQueue* q) {
    return reinterpret_cast<DispatchRecord*>(q->records_buffer.ptr);
}

inline const DispatchRecord* dispatch_queue_records(
    const DispatchQueue* q) {
    return reinterpret_cast<const DispatchRecord*>(q->records_buffer.ptr);
}

inline uint64_t* dispatch_queue_head(
    DispatchQueue* q) {
    return reinterpret_cast<uint64_t*>(q->head_buffer.ptr);
}

inline const uint64_t* dispatch_queue_head(
    const DispatchQueue* q) {
    return reinterpret_cast<const uint64_t*>(q->head_buffer.ptr);
}

inline uint64_t* dispatch_queue_tail(
    DispatchQueue* q) {
    return reinterpret_cast<uint64_t*>(q->tail_buffer.ptr);
}

inline const uint64_t* dispatch_queue_tail(
    const DispatchQueue* q) {
    return reinterpret_cast<const uint64_t*>(q->tail_buffer.ptr);
}

inline uint32_t* dispatch_queue_overflow(
    DispatchQueue* q) {
    return reinterpret_cast<uint32_t*>(q->overflow_buffer.ptr);
}

inline const uint32_t* dispatch_queue_overflow(
    const DispatchQueue* q) {
    return reinterpret_cast<const uint32_t*>(q->overflow_buffer.ptr);
}

bool dispatch_queue_init(
    const std::vector<int>& devices,
    DispatchQueue* q,
    int owner_rank,
    int src_rank,
    int dst_rank,
    uint32_t capacity);

void dispatch_queue_destroy(
    const std::vector<int>& devices,
    DispatchQueue* q);

void dispatch_queue_reset(
    const std::vector<int>& devices,
    DispatchQueue* q);

bool dispatch_queue_push_host_blocking(
    const std::vector<int>& devices,
    DispatchQueue* q,
    const DispatchRecord* rec);

DeviceDispatchQueueHandle dispatch_queue_get_device_handle(
    const DispatchQueue* q);

__device__ __forceinline__ bool device_dispatch_queue_try_peek_ticket(
    const DeviceDispatchQueueHandle* q,
    uint64_t ticket,
    DispatchRecord* out) {
    if (q == nullptr || out == nullptr ||
        q->records == nullptr || q->head == nullptr || q->tail == nullptr) {
        return false;
    }

    dispatch_record_clear(out);

    const uint64_t head_ticket = *((volatile const uint64_t*)q->head);
    const uint64_t tail_ticket = *((volatile const uint64_t*)q->tail);

    if (ticket < head_ticket || ticket >= tail_ticket) {
        return false;
    }

    const DispatchRecord* rec =
        &q->records[static_cast<size_t>(ticket % static_cast<uint64_t>(q->capacity))];

    const uint32_t flags = *((volatile const uint32_t*)&rec->flags);
    const uint64_t seen_ticket = *((volatile const uint64_t*)&rec->queue_ticket);

    if ((flags & kDispatchRecordFlagReady) == 0u) {
        return false;
    }
    if (seen_ticket != ticket) {
        return false;
    }

    *out = *rec;
    return dispatch_record_is_valid(out);
}

__device__ __forceinline__ bool device_dispatch_queue_try_peek_head(
    const DeviceDispatchQueueHandle* q,
    DispatchRecord* out) {
    if (q == nullptr || out == nullptr ||
        q->records == nullptr || q->head == nullptr || q->tail == nullptr) {
        return false;
    }

    dispatch_record_clear(out);

    const uint64_t head_ticket = *((volatile const uint64_t*)q->head);
    const uint64_t tail_ticket = *((volatile const uint64_t*)q->tail);
    if (head_ticket >= tail_ticket) {
        return false;
    }

    const DispatchRecord* rec =
        &q->records[static_cast<size_t>(head_ticket % static_cast<uint64_t>(q->capacity))];

    const uint32_t flags = *((volatile const uint32_t*)&rec->flags);
    const uint64_t seen_ticket = *((volatile const uint64_t*)&rec->queue_ticket);

    if ((flags & kDispatchRecordFlagReady) == 0u) {
        return false;
    }
    if (seen_ticket != head_ticket) {
        return false;
    }

    *out = *rec;
    return dispatch_record_is_valid(out);
}

__device__ __forceinline__ void device_dispatch_queue_release_head(
    DeviceDispatchQueueHandle* q) {
    if (q == nullptr || q->records == nullptr || q->head == nullptr) {
        return;
    }

    const uint64_t head_ticket = *q->head;
    DispatchRecord* rec =
        &q->records[static_cast<size_t>(head_ticket % static_cast<uint64_t>(q->capacity))];

    dispatch_record_clear(rec);
    __threadfence();
    *q->head = head_ticket + 1;
}

__host__ __device__ __forceinline__ void dispatch_record_to_exec_chunk(
    const DispatchRecord* rec,
    exec::Chunk* out) {
    exec::chunk_clear(out);
    if (!dispatch_record_is_valid(rec)) {
        return;
    }

    out->src = reinterpret_cast<const unsigned char*>(rec->src_ptr);
    out->dst = reinterpret_cast<unsigned char*>(rec->dst_ptr);
    out->bytes = rec->bytes;
    out->span_ticket = rec->publish_ticket;
    out->user_tag =
        (static_cast<uint64_t>(rec->tile_id) << 32) |
        static_cast<uint32_t>(rec->chunk_idx);
    out->chunk_idx = static_cast<int>(rec->chunk_idx);
    out->span_offset_bytes = rec->logical_dst_offset_bytes;
    out->op = static_cast<exec::ChunkOpKind>(rec->op_kind);
}

} // namespace transport
} // namespace comm
} // namespace ooverlap
