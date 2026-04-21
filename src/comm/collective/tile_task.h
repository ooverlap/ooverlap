#pragma once

#include <cstddef>
#include <cstdint>

#include "comm/collective/published_tile.h"
#include "comm/exec/chunk.h"
#include "comm/utils.h"

namespace ooverlap {
namespace comm {
namespace collective {

struct TileState {
    uint64_t op_id = 0;
    uint32_t tile_id = 0;

    uint64_t logical_dst_offset_bytes = 0;
    uint64_t final_dst_ptr = 0;
    uint64_t accum_dst_ptr = 0;

    uint32_t bytes = 0;
    uint16_t expected_contributors = 0;
    uint16_t received_contributors = 0;

    uint64_t contributor_mask = 0;
    uint64_t first_publish_ticket = 0;
    uint64_t last_publish_ticket = 0;

    ReduceKind reduce_kind = ReduceKind::kSum;
    bool complete = false;
};

struct ChunkTask {
    uint64_t op_id = 0;
    uint32_t tile_id = 0;

    int src_rank = -1;
    int dst_rank = -1;

    const unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    size_t bytes = 0;

    uint64_t logical_dst_offset_bytes = 0;
    uint64_t publish_ticket = 0;

    int chunk_idx = -1;
    int num_chunks = 0;

    exec::ChunkOpKind op = exec::ChunkOpKind::kInvalid;
};

__host__ __device__ __forceinline__ void tile_state_clear(
    TileState* st) {
    st->op_id = 0;
    st->tile_id = 0;
    st->logical_dst_offset_bytes = 0;
    st->final_dst_ptr = 0;
    st->accum_dst_ptr = 0;
    st->bytes = 0;
    st->expected_contributors = 0;
    st->received_contributors = 0;
    st->contributor_mask = 0;
    st->first_publish_ticket = 0;
    st->last_publish_ticket = 0;
    st->reduce_kind = ReduceKind::kSum;
    st->complete = false;
}

__host__ __device__ __forceinline__ bool tile_state_is_configured(
    const TileState* st) {
    return st != nullptr &&
           st->op_id != 0 &&
           st->bytes > 0 &&
           st->expected_contributors > 0;
}

__host__ __device__ __forceinline__ bool tile_state_is_complete(
    const TileState* st) {
    return tile_state_is_configured(st) &&
           st->complete &&
           st->received_contributors == st->expected_contributors;
}

__host__ __device__ __forceinline__ void tile_state_init_from_published_tile(
    TileState* st,
    const PublishedTile* tile,
    uint16_t expected_contributors,
    unsigned char* final_dst,
    unsigned char* accum_dst) {
    tile_state_clear(st);
    st->op_id = tile->op_id;
    st->tile_id = tile->tile_id;
    st->logical_dst_offset_bytes = tile->logical_dst_offset_bytes;
    st->final_dst_ptr = reinterpret_cast<uint64_t>(final_dst);
    st->accum_dst_ptr = reinterpret_cast<uint64_t>(accum_dst);
    st->bytes = tile->bytes;
    st->expected_contributors = expected_contributors;
    st->received_contributors = 0;
    st->contributor_mask = 0;
    st->first_publish_ticket = tile->publish_ticket;
    st->last_publish_ticket = tile->publish_ticket;
    st->reduce_kind = static_cast<ReduceKind>(tile->reduce_kind);
    st->complete = false;
}

__host__ __device__ __forceinline__ bool tile_state_note_contributor(
    TileState* st,
    const PublishedTile* tile) {
    if (st == nullptr || tile == nullptr) {
        return false;
    }
    if (st->op_id != tile->op_id ||
        st->tile_id != tile->tile_id ||
        st->bytes != tile->bytes) {
        return false;
    }

    const uint16_t rank = tile->producer_rank;
    if (rank >= 64) {
        return false;
    }

    const uint64_t bit = (uint64_t{1} << rank);
    if ((st->contributor_mask & bit) != 0) {
        st->last_publish_ticket = tile->publish_ticket;
        return true;
    }

    st->contributor_mask |= bit;
    st->received_contributors += 1;
    st->last_publish_ticket = tile->publish_ticket;
    st->complete = (st->received_contributors == st->expected_contributors);
    return true;
}

__host__ __device__ __forceinline__ void chunk_task_clear(
    ChunkTask* task) {
    task->op_id = 0;
    task->tile_id = 0;
    task->src_rank = -1;
    task->dst_rank = -1;
    task->src = nullptr;
    task->dst = nullptr;
    task->bytes = 0;
    task->logical_dst_offset_bytes = 0;
    task->publish_ticket = 0;
    task->chunk_idx = -1;
    task->num_chunks = 0;
    task->op = exec::ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool chunk_task_is_valid(
    const ChunkTask* task) {
    return task != nullptr &&
           task->op_id != 0 &&
           task->src_rank >= 0 &&
           task->dst_rank >= 0 &&
           task->src != nullptr &&
           task->dst != nullptr &&
           task->bytes > 0 &&
           task->chunk_idx >= 0 &&
           task->num_chunks > 0 &&
           task->op != exec::ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool chunk_task_make_from_published_tile(
    const PublishedTile* tile,
    int dst_rank,
    unsigned char* dst_base,
    size_t chunk_bytes,
    int chunk_idx,
    ChunkTask* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_task_clear(out);

    if (tile == nullptr ||
        !published_tile_is_valid(tile) ||
        dst_base == nullptr ||
        chunk_bytes == 0 ||
        tile->bytes == 0) {
        return false;
    }

    const int num_chunks =
        static_cast<int>((static_cast<size_t>(tile->bytes) + chunk_bytes - 1) / chunk_bytes);
    if (chunk_idx < 0 || chunk_idx >= num_chunks) {
        return false;
    }

    const size_t offset = static_cast<size_t>(chunk_idx) * chunk_bytes;
    const size_t bytes =
        utils::min_sz(chunk_bytes, static_cast<size_t>(tile->bytes) - offset);

    out->op_id = tile->op_id;
    out->tile_id = tile->tile_id;
    out->src_rank = static_cast<int>(tile->producer_rank);
    out->dst_rank = dst_rank;
    out->src = published_tile_src_bytes(tile) + offset;
    out->dst = dst_base + offset;
    out->bytes = bytes;
    out->logical_dst_offset_bytes = tile->logical_dst_offset_bytes + offset;
    out->publish_ticket = tile->publish_ticket;
    out->chunk_idx = chunk_idx;
    out->num_chunks = num_chunks;
    out->op = exec::ChunkOpKind::kReduceAddNoFtzF16;
    return true;
}

__host__ __device__ __forceinline__ void chunk_task_to_exec_chunk(
    const ChunkTask* task,
    exec::Chunk* out) {
    exec::chunk_clear(out);
    if (!chunk_task_is_valid(task)) {
        return;
    }

    out->src = task->src;
    out->dst = task->dst;
    out->bytes = task->bytes;
    out->span_ticket = task->publish_ticket;
    out->user_tag = (static_cast<uint64_t>(task->tile_id) << 32) |
                    static_cast<uint32_t>(task->chunk_idx);
    out->chunk_idx = task->chunk_idx;
    out->span_offset_bytes = task->logical_dst_offset_bytes;
    out->op = task->op;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
