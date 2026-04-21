#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/collective/allreduce_mapping.h"
#include "comm/collective/tile_task.h"
#include "comm/transport/buffer.h"

namespace ooverlap {
namespace comm {
namespace collective {

struct TileStateTable {
    std::vector<TileState> entries;
};

struct TileAccumulatorWindow {
    uint32_t window_idx = kInvalidWindowIndex;
    bool in_use = false;

    uint64_t bound_op_id = 0;
    uint32_t bound_tile_id = 0;
    uint64_t logical_dst_offset_bytes = 0;
    uint32_t bytes = 0;

    size_t capacity_bytes = 0;
    AllReducePhysicalDstKind dst_kind = AllReducePhysicalDstKind::kInvalid;

    int metadata_owner_rank = -1;
    transport::CommBuffer contributor_count_buffer;

    // Only allocated when dst_kind == kIntermediateAccum.
    std::vector<transport::CommBuffer> accum_buffers;
};

struct TileAccumulatorWindowTable {
    std::vector<TileAccumulatorWindow> entries;
};

struct TileCompletionTable {
    int owner_rank = -1;
    std::vector<transport::CommBuffer> flag_buffers;
};

} // namespace collective
} // namespace comm
} // namespace ooverlap
