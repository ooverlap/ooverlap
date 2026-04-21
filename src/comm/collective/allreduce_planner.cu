#include "comm/collective/allreduce_planner.h"
#include "comm/collective/allreduce_session.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace collective {
namespace {

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

PublishedTile read_published_tile_host(
    const std::vector<int>& devices,
    int owner_rank,
    const PublishedTileQueue* q,
    uint32_t ticket) {

    PublishedTile out{};
    const PublishedTile* records = published_tile_queue_records(q);
    const PublishedTile* src =
        &records[static_cast<size_t>(ticket % q->capacity)];

    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(&out, src, sizeof(PublishedTile), cudaMemcpyDeviceToHost),
        "cudaMemcpy(published tile record)");
    return out;
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
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("allreduce_planner_init: session/group is null");
    }

    planner->session = session;
    planner->dispatch_chunk_bytes =
        (dispatch_chunk_bytes != 0)
            ? dispatch_chunk_bytes
            : session->group->channel_dispatch_chunk_bytes;
    if (planner->dispatch_chunk_bytes == 0) {
        throw std::invalid_argument("allreduce_planner_init: dispatch_chunk_bytes must be > 0");
    }
    return true;
}

void allreduce_planner_destroy(
    AllReducePlanner* planner) {
    if (planner == nullptr) {
        return;
    }
    planner->session = nullptr;
    planner->dispatch_chunk_bytes = 0;
}

bool allreduce_planner_progress_rank(
    AllReducePlanner* planner,
    int producer_rank) {

    if (planner == nullptr || planner->session == nullptr || planner->session->group == nullptr) {
        throw std::invalid_argument("allreduce_planner_progress_rank: planner/session/group is null");
    }

    Group* group = planner->session->group;
    validate_rank_or_throw(group->world_size, producer_rank, "allreduce_planner_progress_rank: invalid producer_rank");

    PublishedTileQueue* q =
        allreduce_session_get_published_tile_queue(planner->session, producer_rank);

    const uint32_t head = read_u32_host(
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

        uint32_t window_idx = kInvalidWindowIndex;
        TileState* st =
            allreduce_session_bind_tile_state(planner->session, &tile, &window_idx);

        // No free operation window right now -> backpressure.
        if (st == nullptr) {
            break;
        }

        changed = true;

        if (!tile_state_note_contributor(st, &tile)) {
            throw std::runtime_error("allreduce_planner_progress_rank: tile_state_note_contributor failed");
        }

        allreduce_session_update_window_contributor_count(
            planner->session,
            window_idx,
            static_cast<uint32_t>(st->received_contributors));

        // TODO(keyvand): add an explicit self/local path.
        // For now we only lower to real peer channels.
        for (int dst_rank = 0; dst_rank < group->world_size; ++dst_rank) {
            Channel* ch = group_get_channel(group, producer_rank, dst_rank);
            if (ch == nullptr) {
                continue;
            }

            AllReducePhysicalTileMapping mapping{};
            if (!allreduce_session_resolve_physical_mapping(
                    planner->session,
                    window_idx,
                    dst_rank,
                    &mapping)) {
                throw std::runtime_error("allreduce_planner_progress_rank: failed to resolve physical mapping");
            }

            const int num_chunks = static_cast<int>(
                (mapping.bytes + planner->dispatch_chunk_bytes - 1) /
                planner->dispatch_chunk_bytes);

            for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                ChunkTask task{};
                if (!chunk_task_make_from_published_tile_and_mapping(
                        &tile,
                        &mapping,
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

        if (tile_state_is_complete(st)) {
            allreduce_session_mark_window_complete(planner->session, window_idx);
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
    if (planner == nullptr || planner->session == nullptr || planner->session->group == nullptr) {
        throw std::invalid_argument("allreduce_planner_progress: planner/session/group is null");
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
