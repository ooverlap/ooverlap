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

void reset_rank_queue_impl(
    AllReduceSession* session,
    int rank) {
    if (session == nullptr) {
        throw std::invalid_argument("reset_rank_queue_impl: session is null");
    }
    if (session->group == nullptr) {
        throw std::invalid_argument("reset_rank_queue_impl: session->group is null");
    }
    validate_rank_or_throw(
        session->group->world_size,
        rank,
        "reset_rank_queue_impl: invalid rank");

    PublishedTileQueue& q =
        session->published_tile_queues[static_cast<size_t>(rank)];

    system::runtime::set_device(session->group->devices[rank]);
    system::runtime::check_cuda(
        cudaMemset(q.records_buffer.ptr, 0, q.records_buffer.bytes),
        "cudaMemset(allreduce session records)");
    system::runtime::check_cuda(
        cudaMemset(q.head_buffer.ptr, 0, q.head_buffer.bytes),
        "cudaMemset(allreduce session head)");
    system::runtime::check_cuda(
        cudaMemset(q.tail_buffer.ptr, 0, q.tail_buffer.bytes),
        "cudaMemset(allreduce session tail)");
    system::runtime::check_cuda(
        cudaMemset(q.overflow_buffer.ptr, 0, q.overflow_buffer.bytes),
        "cudaMemset(allreduce session overflow)");
}

} // namespace

bool allreduce_session_init(
    AllReduceSession* session,
    Group* group,
    uint64_t op_id,
    uint32_t published_tile_capacity,
    ReduceKind reduce_kind) {

    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_init: session is null");
    }
    if (group == nullptr) {
        throw std::invalid_argument("allreduce_session_init: group is null");
    }
    if (group->world_size <= 0) {
        throw std::invalid_argument("allreduce_session_init: group is not initialized");
    }
    if (published_tile_capacity == 0) {
        throw std::invalid_argument("allreduce_session_init: published_tile_capacity must be > 0");
    }

    allreduce_session_destroy(session);

    session->group = group;
    session->op_id = op_id;
    session->reduce_kind = reduce_kind;
    session->published_tile_capacity = published_tile_capacity;
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
            alloc_local_buffer_for_rank(group->devices, rank, records_bytes);
        q.head_buffer =
            alloc_local_buffer_for_rank(group->devices, rank, scalar_bytes);
        q.tail_buffer =
            alloc_local_buffer_for_rank(group->devices, rank, scalar_bytes);
        q.overflow_buffer =
            alloc_local_buffer_for_rank(group->devices, rank, scalar_bytes);
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
            free_comm_buffer(session->group->devices, q.records_buffer);
            free_comm_buffer(session->group->devices, q.head_buffer);
            free_comm_buffer(session->group->devices, q.tail_buffer);
            free_comm_buffer(session->group->devices, q.overflow_buffer);
            q.capacity = 0;
            q.owner_rank = -1;
        }
    }

    session->group = nullptr;
    session->op_id = 0;
    session->reduce_kind = ReduceKind::kSum;
    session->published_tile_capacity = 0;
    session->published_tile_queues.clear();
}

void allreduce_session_reset_rank_queue(
    AllReduceSession* session,
    int rank) {
    reset_rank_queue_impl(session, rank);
}

void allreduce_session_reset_all_queues(
    AllReduceSession* session) {
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_reset_all_queues: session is null");
    }
    if (session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_reset_all_queues: session->group is null");
    }

    for (int rank = 0; rank < session->group->world_size; ++rank) {
        reset_rank_queue_impl(session, rank);
    }
}

const PublishedTileQueue* allreduce_session_get_published_tile_queue(
    const AllReduceSession* session,
    int rank) {
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_get_published_tile_queue: session is null");
    }
    if (session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_get_published_tile_queue: session->group is null");
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
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_get_published_tile_queue: session is null");
    }
    if (session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_get_published_tile_queue: session->group is null");
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
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_get_device_handle: session is null");
    }
    if (session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_get_device_handle: session->group is null");
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
    if (session == nullptr) {
        throw std::invalid_argument("allreduce_session_rank_overflowed: session is null");
    }
    if (session->group == nullptr) {
        throw std::invalid_argument("allreduce_session_rank_overflowed: session->group is null");
    }
    validate_rank_or_throw(
        session->group->world_size,
        rank,
        "allreduce_session_rank_overflowed: invalid rank");

    const PublishedTileQueue& q =
        session->published_tile_queues[static_cast<size_t>(rank)];

    uint32_t overflow = 0;
    system::runtime::set_device(session->group->devices[rank]);
    system::runtime::check_cuda(
        cudaMemcpy(&overflow, q.overflow_buffer.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy(allreduce session overflow)");
    return overflow != 0;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
