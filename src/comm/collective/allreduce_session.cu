#include "comm/collective/allreduce_session.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace collective {
namespace {

int metadata_owner_rank(const AllReduceSession* session) {
    return (session != nullptr && session->group != nullptr && session->group->world_size > 0) ? 0 : -1;
}

void memset_on_rank(
    const std::vector<int>& devices,
    int rank,
    void* ptr,
    size_t bytes,
    const char* what) {
    system::runtime::set_device(devices[static_cast<size_t>(rank)]);
    system::runtime::check_cuda(cudaMemset(ptr, 0, bytes), what);
}

void write_u32_on_rank(
    const std::vector<int>& devices,
    int rank,
    void* ptr,
    uint32_t value,
    const char* what) {
    system::runtime::set_device(devices[static_cast<size_t>(rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(ptr, &value, sizeof(uint32_t), cudaMemcpyHostToDevice),
        what);
}

uint32_t read_u32_on_rank(
    const std::vector<int>& devices,
    int rank,
    const void* ptr,
    const char* what) {
    uint32_t out = 0;
    system::runtime::set_device(devices[static_cast<size_t>(rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(&out, ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost),
        what);
    return out;
}

void reset_rank_queue_impl(
    AllReduceSession* session,
    int rank) {
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("reset_rank_queue_impl: session/group is null");
    }
    validate_rank_or_throw(
        session->group->world_size,
        rank,
        "reset_rank_queue_impl: invalid rank");

    PublishedTileQueue& q =
        session->published_tile_queues[static_cast<size_t>(rank)];

    memset_on_rank(session->group->devices, rank, q.records_buffer.ptr, q.records_buffer.bytes,
                   "cudaMemset(allreduce session records)");
    memset_on_rank(session->group->devices, rank, q.head_buffer.ptr, q.head_buffer.bytes,
                   "cudaMemset(allreduce session head)");
    memset_on_rank(session->group->devices, rank, q.tail_buffer.ptr, q.tail_buffer.bytes,
                   "cudaMemset(allreduce session tail)");
    memset_on_rank(session->group->devices, rank, q.overflow_buffer.ptr, q.overflow_buffer.bytes,
                   "cudaMemset(allreduce session overflow)");
}

void reset_window_impl(
    AllReduceSession* session,
    uint32_t window_idx) {
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("reset_window_impl: session/group is null");
    }
    if (window_idx >= session->operation_window_table.entries.size()) {
        throw std::invalid_argument("reset_window_impl: invalid window_idx");
    }

    TileAccumulatorWindow& win =
        session->operation_window_table.entries[window_idx];

    win.in_use = false;
    win.bound_op_id = 0;
    win.bound_tile_id = 0;
    win.logical_dst_offset_bytes = 0;
    win.bytes = 0;
    win.dst_kind = session->physical_dst_kind;

    tile_state_clear(&session->tile_state_table.entries[window_idx]);

    if (win.contributor_count_buffer.ptr != nullptr) {
        write_u32_on_rank(
            session->group->devices,
            win.metadata_owner_rank,
            win.contributor_count_buffer.ptr,
            0u,
            "cudaMemcpy(reset contributor_count)");
    }

    if (window_idx < session->completion_table.flag_buffers.size() &&
        session->completion_table.flag_buffers[window_idx].ptr != nullptr) {
        write_u32_on_rank(
            session->group->devices,
            session->completion_table.owner_rank,
            session->completion_table.flag_buffers[window_idx].ptr,
            0u,
            "cudaMemcpy(reset completion_flag)");
    }

    if (session->physical_dst_kind == AllReducePhysicalDstKind::kIntermediateAccum) {
        for (int rank = 0; rank < session->group->world_size; ++rank) {
            if (rank < static_cast<int>(win.accum_buffers.size()) &&
                win.accum_buffers[static_cast<size_t>(rank)].ptr != nullptr &&
                win.accum_buffers[static_cast<size_t>(rank)].bytes > 0) {
                memset_on_rank(
                    session->group->devices,
                    rank,
                    win.accum_buffers[static_cast<size_t>(rank)].ptr,
                    win.accum_buffers[static_cast<size_t>(rank)].bytes,
                    "cudaMemset(reset accum_buffer)");
            }
        }
    }
}

} // namespace

bool allreduce_session_init(
    AllReduceSession* session,
    Group* group,
    uint64_t op_id,
    uint32_t published_tile_capacity,
    ReduceKind reduce_kind) {
    return allreduce_session_init_with_window_pool(
        session,
        group,
        op_id,
        published_tile_capacity,
        published_tile_capacity,
        0,
        reduce_kind,
        AllReducePhysicalDstKind::kDirectFinal);
}

bool allreduce_session_init_with_window_pool(
    AllReduceSession* session,
    Group* group,
    uint64_t op_id,
    uint32_t published_tile_capacity,
    uint32_t operation_window_capacity,
    size_t operation_window_bytes,
    ReduceKind reduce_kind,
    AllReducePhysicalDstKind physical_dst_kind) {

    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_init_with_window_pool: session is null");
    }
    if (group == nullptr) {
        throw std::invalid_argument("allreduce_session_init_with_window_pool: group is null");
    }
    if (group->world_size <= 0) {
        throw std::invalid_argument("allreduce_session_init_with_window_pool: group is not initialized");
    }
    if (published_tile_capacity == 0) {
        throw std::invalid_argument("allreduce_session_init_with_window_pool: published_tile_capacity must be > 0");
    }
    if (operation_window_capacity == 0) {
        operation_window_capacity = published_tile_capacity;
    }
    if (physical_dst_kind == AllReducePhysicalDstKind::kIntermediateAccum &&
        operation_window_bytes == 0) {
        throw std::invalid_argument("allreduce_session_init_with_window_pool: operation_window_bytes must be > 0 for intermediate accumulation");
    }

    allreduce_session_destroy(session);

    session->group = group;
    session->op_id = op_id;
    session->reduce_kind = reduce_kind;
    session->published_tile_capacity = published_tile_capacity;
    session->operation_window_capacity = operation_window_capacity;
    session->operation_window_bytes = operation_window_bytes;
    session->physical_dst_kind = physical_dst_kind;

    session->published_tile_queues.resize(static_cast<size_t>(group->world_size));

    const size_t records_bytes =
        static_cast<size_t>(published_tile_capacity) * sizeof(PublishedTile);
    const size_t scalar_bytes = sizeof(uint32_t);

    for (int rank = 0; rank < group->world_size; ++rank) {
        PublishedTileQueue& q =
            session->published_tile_queues[static_cast<size_t>(rank)];
        q.capacity = published_tile_capacity;
        q.owner_rank = rank;
        q.records_buffer =
            transport::alloc_local_buffer_for_rank(group->devices, rank, records_bytes);
        q.head_buffer =
            transport::alloc_local_buffer_for_rank(group->devices, rank, scalar_bytes);
        q.tail_buffer =
            transport::alloc_local_buffer_for_rank(group->devices, rank, scalar_bytes);
        q.overflow_buffer =
            transport::alloc_local_buffer_for_rank(group->devices, rank, scalar_bytes);
    }

    session->tile_state_table.entries.resize(static_cast<size_t>(operation_window_capacity));
    session->operation_window_table.entries.resize(static_cast<size_t>(operation_window_capacity));
    session->completion_table.owner_rank = metadata_owner_rank(session);
    session->completion_table.flag_buffers.resize(static_cast<size_t>(operation_window_capacity));

    for (uint32_t i = 0; i < operation_window_capacity; ++i) {
        TileAccumulatorWindow& win = session->operation_window_table.entries[i];
        win.window_idx = i;
        win.in_use = false;
        win.bound_op_id = 0;
        win.bound_tile_id = 0;
        win.logical_dst_offset_bytes = 0;
        win.bytes = 0;
        win.capacity_bytes = operation_window_bytes;
        win.dst_kind = physical_dst_kind;
        win.metadata_owner_rank = metadata_owner_rank(session);
        win.contributor_count_buffer =
            transport::alloc_local_buffer_for_rank(
                group->devices,
                win.metadata_owner_rank,
                scalar_bytes);

        session->completion_table.flag_buffers[i] =
            transport::alloc_local_buffer_for_rank(
                group->devices,
                session->completion_table.owner_rank,
                scalar_bytes);

        if (physical_dst_kind == AllReducePhysicalDstKind::kIntermediateAccum) {
            win.accum_buffers.resize(static_cast<size_t>(group->world_size));
            for (int rank = 0; rank < group->world_size; ++rank) {
                win.accum_buffers[static_cast<size_t>(rank)] =
                    transport::alloc_local_buffer_for_rank(
                        group->devices,
                        rank,
                        operation_window_bytes);
            }
        }

        tile_state_clear(&session->tile_state_table.entries[i]);
    }

    allreduce_session_reset_all_queues(session);
    return true;
}

void allreduce_session_destroy(
    AllReduceSession* session) {
    if (session == nullptr) {
        return;
    }

    if (session->group != nullptr) {
        for (auto& q : session->published_tile_queues) {
            transport::free_comm_buffer(session->group->devices, q.records_buffer);
            transport::free_comm_buffer(session->group->devices, q.head_buffer);
            transport::free_comm_buffer(session->group->devices, q.tail_buffer);
            transport::free_comm_buffer(session->group->devices, q.overflow_buffer);
            q.capacity = 0;
            q.owner_rank = -1;
        }

        for (auto& win : session->operation_window_table.entries) {
            transport::free_comm_buffer(session->group->devices, win.contributor_count_buffer);
            for (auto& buf : win.accum_buffers) {
                transport::free_comm_buffer(session->group->devices, buf);
            }
            win.accum_buffers.clear();
            win.window_idx = kInvalidWindowIndex;
            win.in_use = false;
            win.bound_op_id = 0;
            win.bound_tile_id = 0;
            win.logical_dst_offset_bytes = 0;
            win.bytes = 0;
            win.capacity_bytes = 0;
            win.dst_kind = AllReducePhysicalDstKind::kInvalid;
            win.metadata_owner_rank = -1;
        }

        for (auto& buf : session->completion_table.flag_buffers) {
            transport::free_comm_buffer(session->group->devices, buf);
        }
    }

    session->group = nullptr;
    session->op_id = 0;
    session->reduce_kind = ReduceKind::kSum;
    session->published_tile_capacity = 0;
    session->operation_window_capacity = 0;
    session->operation_window_bytes = 0;
    session->physical_dst_kind = AllReducePhysicalDstKind::kDirectFinal;
    session->published_tile_queues.clear();
    session->tile_state_table.entries.clear();
    session->operation_window_table.entries.clear();
    session->completion_table.owner_rank = -1;
    session->completion_table.flag_buffers.clear();
}

void allreduce_session_reset_rank_queue(
    AllReduceSession* session,
    int rank) {
    reset_rank_queue_impl(session, rank);
}

void allreduce_session_reset_all_queues(
    AllReduceSession* session) {
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_reset_all_queues: session/group is null");
    }

    for (int rank = 0; rank < session->group->world_size; ++rank) {
        reset_rank_queue_impl(session, rank);
    }

    for (uint32_t i = 0; i < session->operation_window_capacity; ++i) {
        reset_window_impl(session, i);
    }
}

const PublishedTileQueue* allreduce_session_get_published_tile_queue(
    const AllReduceSession* session,
    int rank) {
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_get_published_tile_queue: session/group is null");
    }
    validate_rank_or_throw(
        session->group->world_size,
        rank,
        "allreduce_session_get_published_tile_queue: invalid rank");
    return &session->published_tile_queues[static_cast<size_t>(rank)];
}

PublishedTileQueue* allreduce_session_get_published_tile_queue(
    AllReduceSession* session,
    int rank) {
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_get_published_tile_queue: session/group is null");
    }
    validate_rank_or_throw(
        session->group->world_size,
        rank,
        "allreduce_session_get_published_tile_queue: invalid rank");
    return &session->published_tile_queues[static_cast<size_t>(rank)];
}

DeviceSessionHandle allreduce_session_get_device_handle(
    const AllReduceSession* session,
    int rank) {
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_get_device_handle: session/group is null");
    }
    validate_rank_or_throw(
        session->group->world_size,
        rank,
        "allreduce_session_get_device_handle: invalid rank");

    const PublishedTileQueue& q =
        session->published_tile_queues[static_cast<size_t>(rank)];

    DeviceSessionHandle out{};
    out.op_id = session->op_id;
    out.rank = rank;
    out.world_size = session->group->world_size;
    out.reduce_kind = session->reduce_kind;
    out.records = reinterpret_cast<PublishedTile*>(q.records_buffer.ptr);
    out.head = reinterpret_cast<uint32_t*>(q.head_buffer.ptr);
    out.tail = reinterpret_cast<uint32_t*>(q.tail_buffer.ptr);
    out.overflow = reinterpret_cast<uint32_t*>(q.overflow_buffer.ptr);
    out.capacity = q.capacity;
    return out;
}

bool allreduce_session_rank_overflowed(
    const AllReduceSession* session,
    int rank) {
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_rank_overflowed: session/group is null");
    }
    validate_rank_or_throw(
        session->group->world_size,
        rank,
        "allreduce_session_rank_overflowed: invalid rank");

    const PublishedTileQueue& q =
        session->published_tile_queues[static_cast<size_t>(rank)];

    const uint32_t overflow = read_u32_on_rank(
        session->group->devices,
        rank,
        q.overflow_buffer.ptr,
        "cudaMemcpy(allreduce session overflow)");
    return overflow != 0;
}

TileState* allreduce_session_get_tile_state(
    AllReduceSession* session,
    uint32_t window_idx) {
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_get_tile_state: session is null");
    }
    if (window_idx >= session->tile_state_table.entries.size()) {
        throw std::invalid_argument("allreduce_session_get_tile_state: invalid window_idx");
    }
    return &session->tile_state_table.entries[window_idx];
}

const TileState* allreduce_session_get_tile_state(
    const AllReduceSession* session,
    uint32_t window_idx) {
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_get_tile_state: session is null");
    }
    if (window_idx >= session->tile_state_table.entries.size()) {
        throw std::invalid_argument("allreduce_session_get_tile_state: invalid window_idx");
    }
    return &session->tile_state_table.entries[window_idx];
}

TileState* allreduce_session_bind_tile_state(
    AllReduceSession* session,
    const PublishedTile* tile,
    uint32_t* out_window_idx) {
    if (session == nullptr || session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_bind_tile_state: session/group is null");
    }
    if (tile == nullptr || !published_tile_is_valid(tile)) {
        return nullptr;
    }

    for (uint32_t i = 0; i < session->operation_window_table.entries.size(); ++i) {
        TileAccumulatorWindow& win = session->operation_window_table.entries[i];
        TileState& st = session->tile_state_table.entries[i];

        if (win.in_use &&
            st.op_id == tile->op_id &&
            st.tile_id == tile->tile_id) {
            if (out_window_idx != nullptr) {
                *out_window_idx = i;
            }
            return &st;
        }
    }

    for (uint32_t i = 0; i < session->operation_window_table.entries.size(); ++i) {
        TileAccumulatorWindow& win = session->operation_window_table.entries[i];
        if (win.in_use) {
            continue;
        }

        win.in_use = true;
        win.bound_op_id = tile->op_id;
        win.bound_tile_id = tile->tile_id;
        win.logical_dst_offset_bytes = tile->logical_dst_offset_bytes;
        win.bytes = tile->bytes;
        win.dst_kind = session->physical_dst_kind;

        reset_window_impl(session, i);
        win.in_use = true;
        win.bound_op_id = tile->op_id;
        win.bound_tile_id = tile->tile_id;
        win.logical_dst_offset_bytes = tile->logical_dst_offset_bytes;
        win.bytes = tile->bytes;
        win.dst_kind = session->physical_dst_kind;

        TileState* st = &session->tile_state_table.entries[i];
        tile_state_init_from_published_tile(
            st,
            tile,
            static_cast<uint16_t>(session->group->world_size),
            i,
            session->physical_dst_kind);

        if (out_window_idx != nullptr) {
            *out_window_idx = i;
        }
        return st;
    }

    return nullptr;
}

void allreduce_session_release_window(
    AllReduceSession* session,
    uint32_t window_idx) {
    reset_window_impl(session, window_idx);
}

void allreduce_session_update_window_contributor_count(
    AllReduceSession* session,
    uint32_t window_idx,
    uint32_t contributor_count) {
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_update_window_contributor_count: session is null");
    }
    if (window_idx >= session->operation_window_table.entries.size()) {
        throw std::invalid_argument("allreduce_session_update_window_contributor_count: invalid window_idx");
    }

    TileAccumulatorWindow& win = session->operation_window_table.entries[window_idx];
    if (win.contributor_count_buffer.ptr == nullptr) {
        return;
    }

    write_u32_on_rank(
        session->group->devices,
        win.metadata_owner_rank,
        win.contributor_count_buffer.ptr,
        contributor_count,
        "cudaMemcpy(update contributor_count)");
}

void allreduce_session_mark_window_complete(
    AllReduceSession* session,
    uint32_t window_idx) {
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_mark_window_complete: session is null");
    }
    if (window_idx >= session->tile_state_table.entries.size() ||
        window_idx >= session->completion_table.flag_buffers.size()) {
        throw std::invalid_argument("allreduce_session_mark_window_complete: invalid window_idx");
    }

    session->tile_state_table.entries[window_idx].complete = true;

    write_u32_on_rank(
        session->group->devices,
        session->completion_table.owner_rank,
        session->completion_table.flag_buffers[window_idx].ptr,
        1u,
        "cudaMemcpy(mark completion_flag)");
}

bool allreduce_session_window_is_complete(
    const AllReduceSession* session,
    uint32_t window_idx) {
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_window_is_complete: session is null");
    }
    if (window_idx >= session->completion_table.flag_buffers.size()) {
        throw std::invalid_argument("allreduce_session_window_is_complete: invalid window_idx");
    }

    const uint32_t flag = read_u32_on_rank(
        session->group->devices,
        session->completion_table.owner_rank,
        session->completion_table.flag_buffers[window_idx].ptr,
        "cudaMemcpy(read completion_flag)");
    return flag != 0;
}

bool allreduce_session_resolve_physical_mapping(
    const AllReduceSession* session,
    uint32_t window_idx,
    int dst_rank,
    AllReducePhysicalTileMapping* out) {
    if (out == nullptr) {
        return false;
    }
    *out = AllReducePhysicalTileMapping{};

    if (session == nullptr || session->group == nullptr) {
        return false;
    }
    if (window_idx >= session->operation_window_table.entries.size() ||
        window_idx >= session->tile_state_table.entries.size()) {
        return false;
    }
    validate_rank_or_throw(
        session->group->world_size,
        dst_rank,
        "allreduce_session_resolve_physical_mapping: invalid dst_rank");

    const TileAccumulatorWindow& win =
        session->operation_window_table.entries[window_idx];
    const TileState& st =
        session->tile_state_table.entries[window_idx];

    if (!win.in_use || !tile_state_is_configured(&st)) {
        return false;
    }

    out->window_idx = window_idx;
    out->dst_kind = win.dst_kind;
    out->dst_rank = dst_rank;
    out->bytes = st.bytes;
    out->logical_dst_offset_bytes = st.logical_dst_offset_bytes;

    if (win.dst_kind == AllReducePhysicalDstKind::kDirectFinal) {
        const auto* dst_buf =
            group_get_local_full_buffer(session->group, dst_rank);
        out->dst_base =
            reinterpret_cast<unsigned char*>(dst_buf->ptr) +
            st.logical_dst_offset_bytes;
        return true;
    }

    if (win.dst_kind == AllReducePhysicalDstKind::kIntermediateAccum) {
        if (dst_rank >= static_cast<int>(win.accum_buffers.size())) {
            return false;
        }
        if (st.bytes > win.capacity_bytes) {
            return false;
        }
        out->dst_base =
            reinterpret_cast<unsigned char*>(win.accum_buffers[static_cast<size_t>(dst_rank)].ptr);
        return true;
    }

    return false;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
