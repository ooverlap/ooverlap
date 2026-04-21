#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comm/collective/published_tile.h"

namespace ooverlap {
namespace comm {
namespace collective {

struct DeviceSessionHandle {
    uint64_t op_id = 0;
    int rank = -1;
    int world_size = 0;
    ReduceKind reduce_kind = ReduceKind::kSum;

    PublishedTile* records = nullptr;
    uint32_t* head = nullptr;
    uint32_t* tail = nullptr;
    uint32_t* overflow = nullptr;
    uint32_t capacity = 0;
};

__host__ __device__ __forceinline__ bool device_session_handle_is_valid(
    const DeviceSessionHandle* sess) {
    return sess != nullptr &&
           sess->op_id != 0 &&
           sess->rank >= 0 &&
           sess->world_size > 0 &&
           sess->records != nullptr &&
           sess->tail != nullptr &&
           sess->overflow != nullptr &&
           sess->capacity > 0;
}

__device__ __forceinline__ bool device_publish_tile(
    DeviceSessionHandle const& sess,
    uint32_t tile_id,
    const void* src,
    uint32_t bytes,
    uint64_t logical_dst_offset_bytes) {

    if (!device_session_handle_is_valid(&sess) ||
        src == nullptr ||
        bytes == 0) {
        return false;
    }

    const uint32_t ticket = atomicAdd(sess.tail, 1u);
    if (ticket >= sess.capacity) {
        atomicExch(sess.overflow, 1u);
        return false;
    }

    PublishedTile rec{};
    rec.op_id = sess.op_id;
    rec.publish_ticket = static_cast<uint64_t>(ticket);
    rec.logical_dst_offset_bytes = logical_dst_offset_bytes;
    rec.src_ptr = reinterpret_cast<uint64_t>(src);
    rec.tile_id = tile_id;
    rec.bytes = bytes;
    rec.producer_rank = static_cast<uint16_t>(sess.rank);
    rec.reduce_kind = static_cast<uint8_t>(sess.reduce_kind);
    rec.flags = 0;

    sess.records[ticket] = rec;
    __threadfence();
    sess.records[ticket].flags = kPublishedTileFlagReady;
    return true;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
