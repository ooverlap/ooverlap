#include "comm/collective/allreduce_planner.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace collective {
namespace {

TileState* find_tile_state(
    std::vector<TileState>* states,
    uint64_t op_id,
    uint32_t tile_id) {
    for (auto& st : *states) {
        if (st.op_id == op_id && st.tile_id == tile_id) {
            return &st;
        }
    }
    return nullptr;
}

TileState* get_or_create_tile_state(
    AllReducePlanner* planner,
    const PublishedTile* tile) {
    TileState* st = find_tile_state(&planner->tile_states, tile->op_id, tile->tile_id);
    if (st != nullptr) {
        return st;
    }

    TileState fresh{};
    fresh.op_id = tile->op_id;
    fresh.tile_id = tile->tile_id;
    fresh.logical_dst_offset_bytes = tile->logical_dst_offset_bytes;
    fresh.bytes = tile->bytes;
    fresh.expected_contributors =
        static_cast<uint16_t>(planner->session->group->world_size);
    fresh.received_contributors = 0;
    fresh.contributor_mask = 0;
    fresh.first_publish_ticket = tile->publish_ticket;
    fresh.last_publish_ticket = tile->publish_ticket;
    fresh.reduce_kind = static_cast<ReduceKind>(tile->reduce_kind);
    fresh.complete = false;

    planner->tile_states.push_back(fresh);
    return &planner->tile_states.back();
}

PublishedTile read_published_tile_host(
    const std::vector<int>& devices,
    int owner_rank,
    const PublishedTileQueue* q,
    uint64_t ticket) {

    PublishedTile out{};
    const PublishedTile* records = published_tile_queue_records(q);
    const PublishedTile* src =
        &records[static_cast<size_t>(ticket % static_cast<uint64_t>(q->capacity))];

    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(&out, src, sizeof(PublishedTile), cudaMemcpyDeviceToHost),
        "cudaMemcpy(published tile record)");
    return out;
}

uint32_t read_u32_host(
    const std::vector<int>& devices,
    int owner_rank,
    const void* ptr,
    const char* what) {
    uint32_t out = 0;
    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(&out, ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost),
        what);
    return out;
}

void write_u32_host(
    const std::vector<int>& devices,
    int owner_rank,
    void* ptr,
    uint32_t value,
    const char* what) {
    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(ptr, &value, sizeof(uint32_t), cudaMemcpyHostToDevice),
        what);
}

transport::DispatchRecord dispatch_record_from_chunk_task(
    const ChunkTask* task) {
    transport::DispatchRecord rec{};
    rec.op_id = task->op_id;
    rec.publish_ticket = task->publish_ticket;
    rec.logical_dst_offset_bytes = task->logical_dst_offset_bytes;
    rec.src_ptr = reinterpret_cast<uint64_t>(task->src);
    rec.dst_ptr = reinterpret_cast<uint64_t>(task->dst);
    rec.tile_id = task->tile_id;
    rec.bytes = static_cast<uint32_t>(task->bytes);
    rec.src_rank = static_cast<uint16_t>(task->src_rank);
    rec.dst_rank = static_cast<uint16_t>(task->dst_rank);
    rec.chunk_idx = static_cast<uint16_t>(task->chunk_idx);
    rec.num_chunks = static_cast<uint16_t>(task->num_chunks);
    rec.op_kind = static_cast<uint8_t>(task->op);
    rec.flags = 0;
    return rec;
}

} // namespace

bool allreduce_planner_init(
    AllReducePlanner* planner,
    AllReduceSession* session,
    size_t dispatch_chunk_bytes) {
    if (planner == nullptr) {
        throw std::invalid_argument("allreduce_planner_init: planner is null");
    }
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_planner_init: session is null");
    }
    if (session->group == nullptr) {
        throw std::invalid_argument("allreduce_planner_init: session->group is null");
    }
    if (dispatch_chunk_bytes == 0) {
        throw std::invalid_argument("allreduce_planner_init: dispatch_chunk_bytes must be > 0");
    }

    planner->session = session;
    planner->dispatch_chunk_bytes = dispatch_chunk_bytes;
    planner->tile_states.clear();
    return true;
}

void allreduce_planner_destroy(
    AllReducePlanner* planner) {
    if (planner == nullptr) {
        return;
    }
    planner->session = nullptr;
    planner->dispatch_chunk_bytes = 0;
    planner->tile_states.clear();
}

void allreduce_planner_reset(
    AllReducePlanner* planner) {
    if (planner == nullptr) {
        throw std::invalid_argument("allreduce_planner_reset: planner is null");
    }
    planner->tile_states.clear();
}

bool allreduce_planner_progress_rank(
    AllReducePlanner* planner,
    int producer_rank) {

    if (planner == nullptr) {
        throw std::invalid_argument("allreduce_planner_progress_rank: planner is null");
    }
    if (planner->session == nullptr || planner->session->group == nullptr) {
        throw std::invalid_argument("allreduce_planner_progress_rank: planner session/group is null");
    }

    Group* group = planner->session->group;
    validate_rank_or_throw(group->world_size, producer_rank, "allreduce_planner_progress_rank: invalid producer_rank");

    PublishedTileQueue* q =
        allreduce_session_get_published_tile_queue(planner->session, producer_rank);

    uint32_t head = read_u32_host(
        group->devices,
        producer_rank,
        q->head_buffer.ptr,
        "cudaMemcpy(published queue head)");

    const uint32_t tail = read_u32_host(
        group->devices,
        producer_rank,
        q->tail_buffer.ptr,
        "cudaMemcpy(published queue tail)");

    bool changed = false;
    uint32_t consumed_head = head;

    while (consumed_head < tail) {
        const PublishedTile tile =
            read_published_tile_host(group->devices, producer_rank, q, consumed_head);

        if (!published_tile_is_valid(&tile)) {
            break;
        }
        if (tile.publish_ticket != consumed_head) {
            break;
        }

        changed = true;

        TileState* st = get_or_create_tile_state(planner, &tile);
        tile_state_note_contributor(st, &tile);

        // Current baseline lowering:
        // fan out this producer contribution to every peer rank.
        // Physical destination is currently resolved to group local_full_buffers.
        // TODO(keyvand): replace this with session-owned operation window mapping.
        for (int dst_rank = 0; dst_rank < group->world_size; ++dst_rank) {
            if (dst_rank == producer_rank) {
                continue;
            }

            Channel* ch = group_get_channel(group, producer_rank, dst_rank);
            if (ch == nullptr) {
                continue;
            }

            auto* dst_buf = group_get_local_full_buffer(group, dst_rank);
            unsigned char* dst_base =
                reinterpret_cast<unsigned char*>(dst_buf->ptr) +
                tile.logical_dst_offset_bytes;

            const int num_chunks = static_cast<int>(
                (static_cast<size_t>(tile.bytes) + planner->dispatch_chunk_bytes - 1) /
                planner->dispatch_chunk_bytes);

            for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                ChunkTask task{};
                if (!chunk_task_make_from_published_tile(
                        &tile,
                        dst_rank,
                        dst_base,
                        planner->dispatch_chunk_bytes,
                        chunk_idx,
                        &task)) {
                    throw std::runtime_error("allreduce_planner_progress_rank: failed to build ChunkTask");
                }

                const transport::DispatchRecord rec =
                    dispatch_record_from_chunk_task(&task);

                if (!transport::dispatch_queue_push_host_blocking(
                        group->devices,
                        &ch->dispatch_queue,
                        &rec)) {
                    throw std::runtime_error("allreduce_planner_progress_rank: failed to push DispatchRecord");
                }
            }
        }

        consumed_head += 1;
    }

    if (consumed_head != head) {
        write_u32_host(
            group->devices,
            producer_rank,
            q->head_buffer.ptr,
            consumed_head,
            "cudaMemcpy(update published queue head)");
    }

    return changed;
}

bool allreduce_planner_progress(
    AllReducePlanner* planner) {
    if (planner == nullptr) {
        throw std::invalid_argument("allreduce_planner_progress: planner is null");
    }
    if (planner->session == nullptr || planner->session->group == nullptr) {
        throw std::invalid_argument("allreduce_planner_progress: planner session/group is null");
    }

    bool changed = false;
    for (int rank = 0; rank < planner->session->group->world_size; ++rank) {
        changed |= allreduce_planner_progress_rank(planner, rank);
    }
    return changed;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
