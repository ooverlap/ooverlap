#pragma once

#include <cstddef>
#include <cstdint>

#include "comm/transport/buffer.h"

namespace ooverlap {
namespace comm {
namespace collective {

enum class ReduceKind : uint8_t {
    kSum = 0,
};

enum : uint32_t {
    kPublishedTileFlagReady = 1u << 0,
    kPublishedTileFlagConsumed = 1u << 1,
};

struct PublishedTile {
    uint64_t op_id = 0;
    uint64_t publish_ticket = 0;

    uint64_t logical_dst_offset_bytes = 0;
    uint64_t src_ptr = 0;

    uint32_t tile_id = 0;
    uint32_t bytes = 0;

    uint16_t producer_rank = 0;
    uint8_t reduce_kind = static_cast<uint8_t>(ReduceKind::kSum);
    uint8_t reserved0 = 0;

    uint32_t flags = 0;
};

struct PublishedTileQueue {
    transport::CommBuffer records_buffer;
    transport::CommBuffer head_buffer;
    transport::CommBuffer tail_buffer;
    transport::CommBuffer overflow_buffer;

    uint32_t capacity = 0;
    int owner_rank = -1;
};

__host__ __device__ __forceinline__ void published_tile_clear(
    PublishedTile* tile) {
    tile->op_id = 0;
    tile->publish_ticket = 0;
    tile->logical_dst_offset_bytes = 0;
    tile->src_ptr = 0;
    tile->tile_id = 0;
    tile->bytes = 0;
    tile->producer_rank = 0;
    tile->reduce_kind = static_cast<uint8_t>(ReduceKind::kSum);
    tile->reserved0 = 0;
    tile->flags = 0;
}

__host__ __device__ __forceinline__ bool published_tile_is_ready(
    const PublishedTile* tile) {
    return tile != nullptr &&
           (tile->flags & kPublishedTileFlagReady) != 0u;
}

__host__ __device__ __forceinline__ bool published_tile_is_valid(
    const PublishedTile* tile) {
    return tile != nullptr &&
           tile->src_ptr != 0 &&
           tile->bytes > 0 &&
           published_tile_is_ready(tile);
}

__host__ __device__ __forceinline__ const unsigned char* published_tile_src_bytes(
    const PublishedTile* tile) {
    return reinterpret_cast<const unsigned char*>(tile->src_ptr);
}

__host__ __device__ __forceinline__ bool published_tile_queue_is_configured(
    const PublishedTileQueue* q) {
    return q != nullptr &&
           q->records_buffer.ptr != nullptr &&
           q->tail_buffer.ptr != nullptr &&
           q->overflow_buffer.ptr != nullptr &&
           q->capacity > 0 &&
           q->owner_rank >= 0;
}

inline PublishedTile* published_tile_queue_records(
    PublishedTileQueue* q) {
    return reinterpret_cast<PublishedTile*>(q->records_buffer.ptr);
}

inline const PublishedTile* published_tile_queue_records(
    const PublishedTileQueue* q) {
    return reinterpret_cast<const PublishedTile*>(q->records_buffer.ptr);
}

inline uint32_t* published_tile_queue_head(
    PublishedTileQueue* q) {
    return reinterpret_cast<uint32_t*>(q->head_buffer.ptr);
}

inline const uint32_t* published_tile_queue_head(
    const PublishedTileQueue* q) {
    return reinterpret_cast<const uint32_t*>(q->head_buffer.ptr);
}

inline uint32_t* published_tile_queue_tail(
    PublishedTileQueue* q) {
    return reinterpret_cast<uint32_t*>(q->tail_buffer.ptr);
}

inline const uint32_t* published_tile_queue_tail(
    const PublishedTileQueue* q) {
    return reinterpret_cast<const uint32_t*>(q->tail_buffer.ptr);
}

inline uint32_t* published_tile_queue_overflow(
    PublishedTileQueue* q) {
    return reinterpret_cast<uint32_t*>(q->overflow_buffer.ptr);
}

inline const uint32_t* published_tile_queue_overflow(
    const PublishedTileQueue* q) {
    return reinterpret_cast<const uint32_t*>(q->overflow_buffer.ptr);
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
