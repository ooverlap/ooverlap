#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace collective {

enum : uint32_t {
    kReadyTileFlagReady = 1u << 0,
};

struct ReadyTile {
    uint64_t queue_ticket = 0;

    uint64_t tile_id = 0;
    uint64_t src_ptr = 0;

    uint32_t bytes = 0;

    // Optional shape metadata.
    // Leave as 0 if not needed.
    uint32_t dim0 = 0;
    uint32_t dim1 = 0;
    uint32_t dim2 = 0;

    uint32_t flags = 0;
};

// This is the queue.
// Host allocates the device memory behind these pointers.
// GEMM and persistent kernel both use this same struct.
struct ReadyTileQueue {
    ReadyTile* records = nullptr;  // device pointer
    uint64_t* head = nullptr;      // device pointer
    uint64_t* tail = nullptr;      // device pointer

    uint32_t capacity = 0;
    int device = -1;
};

__host__ __device__ __forceinline__ void ready_tile_clear(
    ReadyTile* tile) {
    tile->queue_ticket = 0;
    tile->tile_id = 0;
    tile->src_ptr = 0;
    tile->bytes = 0;
    tile->dim0 = 0;
    tile->dim1 = 0;
    tile->dim2 = 0;
    tile->flags = 0;
}

__host__ __device__ __forceinline__ bool ready_tile_is_ready(
    const ReadyTile* tile) {
    return tile != nullptr &&
           (tile->flags & kReadyTileFlagReady) != 0u;
}

__host__ __device__ __forceinline__ bool ready_tile_is_valid(
    const ReadyTile* tile) {
    return tile != nullptr &&
           tile->src_ptr != 0 &&
           tile->bytes > 0 &&
           ready_tile_is_ready(tile);
}

__host__ __device__ __forceinline__ const unsigned char* ready_tile_src_bytes(
    const ReadyTile* tile) {
    return reinterpret_cast<const unsigned char*>(tile->src_ptr);
}

__host__ __device__ __forceinline__ bool ready_tile_queue_is_configured(
    const ReadyTileQueue* q) {
    return q != nullptr &&
           q->records != nullptr &&
           q->head != nullptr &&
           q->tail != nullptr &&
           q->capacity > 0 &&
           q->device >= 0;
}

bool ready_tile_queue_init(
    ReadyTileQueue* q,
    int device,
    uint32_t capacity);

void ready_tile_queue_reset(
    ReadyTileQueue* q);

void ready_tile_queue_destroy(
    ReadyTileQueue* q);

// Multi-producer, single-consumer bounded ring.
// Producers spin while the queue is full until the consumer advances head.
__device__ __forceinline__ bool device_ready_tile_queue_push_blocking(
    ReadyTileQueue const& q,
    const ReadyTile* in,
    uint64_t* out_ticket = nullptr) {
    if (!ready_tile_queue_is_configured(&q) ||
        in == nullptr ||
        in->src_ptr == 0 ||
        in->bytes == 0) {
        return false;
    }

    uint64_t ticket = 0;

    while (true) {
        const uint64_t head_ticket = *((volatile const uint64_t*)q.head);
        const uint64_t tail_ticket = *((volatile const uint64_t*)q.tail);

        if ((tail_ticket - head_ticket) >= static_cast<uint64_t>(q.capacity)) {
            // Full: spin until consumer pops something.
            continue;
        }

        const unsigned long long seen = atomicCAS(
            reinterpret_cast<unsigned long long*>(q.tail),
            static_cast<unsigned long long>(tail_ticket),
            static_cast<unsigned long long>(tail_ticket + 1));

        if (seen == static_cast<unsigned long long>(tail_ticket)) {
            ticket = tail_ticket;
            break;
        }
    }

    ReadyTile staged = *in;
    staged.queue_ticket = ticket;
    staged.flags = 0;

    ReadyTile* dst =
        &q.records[static_cast<size_t>(ticket % static_cast<uint64_t>(q.capacity))];

    *dst = staged;
    __threadfence();
    dst->flags = kReadyTileFlagReady;

    if (out_ticket != nullptr) {
        *out_ticket = ticket;
    }
    return true;
}

__device__ __forceinline__ bool device_publish_ready_tile(
    ReadyTileQueue const& q,
    uint64_t tile_id,
    const void* src,
    uint32_t bytes,
    uint32_t dim0 = 0,
    uint32_t dim1 = 0,
    uint32_t dim2 = 0,
    uint64_t* out_ticket = nullptr) {
    if (!ready_tile_queue_is_configured(&q) ||
        src == nullptr ||
        bytes == 0) {
        return false;
    }

    ReadyTile tile{};
    tile.tile_id = tile_id;
    tile.src_ptr = reinterpret_cast<uint64_t>(src);
    tile.bytes = bytes;
    tile.dim0 = dim0;
    tile.dim1 = dim1;
    tile.dim2 = dim2;
    tile.flags = 0;

    return device_ready_tile_queue_push_blocking(q, &tile, out_ticket);
}

// Consumer-side helpers.
// Assumes a single consumer advances head.

__device__ __forceinline__ bool device_ready_tile_queue_try_peek_ticket(
    const ReadyTileQueue* q,
    uint64_t ticket,
    ReadyTile* out) {
    if (!ready_tile_queue_is_configured(q) || out == nullptr) {
        return false;
    }

    ready_tile_clear(out);

    const uint64_t head_ticket = *((volatile const uint64_t*)q->head);
    const uint64_t tail_ticket = *((volatile const uint64_t*)q->tail);

    if (ticket < head_ticket || ticket >= tail_ticket) {
        return false;
    }

    const ReadyTile* rec =
        &q->records[static_cast<size_t>(ticket % static_cast<uint64_t>(q->capacity))];

    const uint32_t flags = *((volatile const uint32_t*)&rec->flags);
    const uint64_t seen_ticket = *((volatile const uint64_t*)&rec->queue_ticket);

    if ((flags & kReadyTileFlagReady) == 0u) {
        return false;
    }
    if (seen_ticket != ticket) {
        return false;
    }

    *out = *rec;
    return ready_tile_is_valid(out);
}

__device__ __forceinline__ bool device_ready_tile_queue_try_peek_head(
    const ReadyTileQueue* q,
    ReadyTile* out) {
    if (!ready_tile_queue_is_configured(q) || out == nullptr) {
        return false;
    }

    const uint64_t head_ticket = *((volatile const uint64_t*)q->head);
    return device_ready_tile_queue_try_peek_ticket(q, head_ticket, out);
}

__device__ __forceinline__ void device_ready_tile_queue_pop_head(
    ReadyTileQueue* q) {
    if (!ready_tile_queue_is_configured(q)) {
        return;
    }

    const uint64_t head_ticket = *q->head;
    ReadyTile* rec =
        &q->records[static_cast<size_t>(head_ticket % static_cast<uint64_t>(q->capacity))];

    ready_tile_clear(rec);
    __threadfence();
    *q->head = head_ticket + 1;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
