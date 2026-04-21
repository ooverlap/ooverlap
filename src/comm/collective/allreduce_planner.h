#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/collective/allreduce_session.h"
#include "comm/transport/dispatch_queue.h"

namespace ooverlap {
namespace comm {
namespace collective {

// Host-side progress/planner layer.
// Reads PublishedTile queues, updates TileState,
// lowers work into per-channel DispatchQueue.
struct AllReducePlanner {
    AllReduceSession* session = nullptr;
    size_t dispatch_chunk_bytes = 0;

    std::vector<TileState> tile_states;
};

bool allreduce_planner_init(
    AllReducePlanner* planner,
    AllReduceSession* session,
    size_t dispatch_chunk_bytes);

void allreduce_planner_destroy(
    AllReducePlanner* planner);

void allreduce_planner_reset(
    AllReducePlanner* planner);

bool allreduce_planner_progress_rank(
    AllReducePlanner* planner,
    int producer_rank);

bool allreduce_planner_progress(
    AllReducePlanner* planner);

} // namespace collective
} // namespace comm
} // namespace ooverlap
