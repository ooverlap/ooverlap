#pragma once

#include <cstdint>
#include <vector>

#include "comm/collective/device_session_handle.h"
#include "comm/collective/published_tile.h"
#include "comm/collective/tile_task.h"
#include "comm/group.h"

namespace ooverlap {
namespace comm {
namespace collective {

struct AllReduceSession {
    Group* group = nullptr;
    uint64_t op_id = 0;
    ReduceKind reduce_kind = ReduceKind::kSum;

    uint32_t published_tile_capacity = 0;

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

const PublishedTileQueue* allreduce_session_get_published_tile_queue(
    const AllReduceSession* session,
    int rank);

PublishedTileQueue* allreduce_session_get_published_tile_queue(
    AllReduceSession* session,
    int rank);

DeviceSessionHandle allreduce_session_get_device_handle(
    const AllReduceSession* session,
    int rank);

bool allreduce_session_rank_overflowed(
    const AllReduceSession* session,
    int rank);

} // namespace collective
} // namespace comm
} // namespace ooverlap
