#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/collective/allreduce_mapping.h"
#include "comm/collective/device_session_handle.h"
#include "comm/collective/operation_window.h"
#include "comm/collective/published_tile.h"
#include "comm/group.h"

namespace ooverlap {
namespace comm {
namespace collective {

struct AllReduceSession {
    Group* group = nullptr;
    uint64_t op_id = 0;
    ReduceKind reduce_kind = ReduceKind::kSum;

    uint32_t published_tile_capacity = 0;

    uint32_t operation_window_capacity = 0;
    size_t operation_window_bytes = 0;
    AllReducePhysicalDstKind physical_dst_kind = AllReducePhysicalDstKind::kDirectFinal;

    std::vector<PublishedTileQueue> published_tile_queues;

    TileStateTable tile_state_table;
    TileAccumulatorWindowTable operation_window_table;
    TileCompletionTable completion_table;
};

bool allreduce_session_init(
    AllReduceSession* session,
    Group* group,
    uint64_t op_id,
    uint32_t published_tile_capacity,
    ReduceKind reduce_kind = ReduceKind::kSum);

bool allreduce_session_init_with_window_pool(
    AllReduceSession* session,
    Group* group,
    uint64_t op_id,
    uint32_t published_tile_capacity,
    uint32_t operation_window_capacity,
    size_t operation_window_bytes,
    ReduceKind reduce_kind = ReduceKind::kSum,
    AllReducePhysicalDstKind physical_dst_kind = AllReducePhysicalDstKind::kDirectFinal);

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

TileState* allreduce_session_get_tile_state(
    AllReduceSession* session,
    uint32_t window_idx);

const TileState* allreduce_session_get_tile_state(
    const AllReduceSession* session,
    uint32_t window_idx);

TileState* allreduce_session_bind_tile_state(
    AllReduceSession* session,
    const PublishedTile* tile,
    uint32_t* out_window_idx);

void allreduce_session_release_window(
    AllReduceSession* session,
    uint32_t window_idx);

void allreduce_session_update_window_contributor_count(
    AllReduceSession* session,
    uint32_t window_idx,
    uint32_t contributor_count);

void allreduce_session_mark_window_complete(
    AllReduceSession* session,
    uint32_t window_idx);

bool allreduce_session_window_is_complete(
    const AllReduceSession* session,
    uint32_t window_idx);

bool allreduce_session_resolve_physical_mapping(
    const AllReduceSession* session,
    uint32_t window_idx,
    int dst_rank,
    AllReducePhysicalTileMapping* out);

} // namespace collective
} // namespace comm
} // namespace ooverlap
