#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/group.h"

namespace ooverlap {
namespace comm {
namespace collective {

enum class ReduceKind : uint8_t {
    kSum = 0,
};

enum : uint32_t {
    kPublishedTileFlagReady = 1u,
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
    CommBuffer records_buffer;
    CommBuffer tail_buffer;
    CommBuffer overflow_buffer;

    uint32_t capacity = 0;
    int owner_rank = -1;
};

struct DeviceSessionHandle {
    uint64_t op_id = 0;
    int rank = -1;
    int world_size = 0;
    ReduceKind reduce_kind = ReduceKind::kSum;

    PublishedTile* records = nullptr;
    uint32_t* tail = nullptr;
    uint32_t* overflow = nullptr;
    uint32_t capacity = 0;
};

struct AllReduceSession {
    Group* group = nullptr;
    uint64_t op_id = 0;
    ReduceKind reduce_kind = ReduceKind::kSum;

    uint32_t published_tile_capacity = 0;

    // One producer-facing queue per rank.
    std::vector<PublishedTileQueue> published_tile_queues;
};

bool allreduce_session_init(
    AllReduceSession* session,
    Group* group,
    uint64_t op_id,
    uint32_t published_tile_capacity,
    ReduceKind reduce_kind = ReduceKind::kSum);

void allreduce_session_destroy(
    AllReduceSession* session);

void allreduce_session_reset_rank_queue(
    AllReduceSession* session,
    int rank);

void allreduce_session_reset_all_queues(
    AllReduceSession* session);

DeviceSessionHandle allreduce_session_get_device_handle(
    const AllReduceSession* session,
    int rank);

bool allreduce_session_rank_overflowed(
    const AllReduceSession* session,
    int rank);

__device__ __forceinline__ bool device_publish_tile(
    DeviceSessionHandle const& sess,
    uint32_t tile_id,
    const void* src,
    uint32_t bytes,
    uint64_t logical_dst_offset_bytes) {

    if (sess.records == nullptr ||
        sess.tail == nullptr ||
        sess.overflow == nullptr ||
        sess.capacity == 0 ||
        src == nullptr ||
        bytes == 0 ||
        sess.rank < 0) {
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
