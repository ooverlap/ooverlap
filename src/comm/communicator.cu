#include "comm/communicator.h"
#include "comm/channel_worker.h"
#include "comm/transport/control_plane.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace comm {
    
bool group_init(
    Group* group,
    const std::vector<int>& devices,
    size_t max_full_numel,
    int num_channel_slots,
    ChannelMode channel_mode,
    uint32_t channel_dispatch_capacity,
    size_t channel_dispatch_chunk_bytes) {
    if (group == nullptr) {
        throw std::invalid_argument("group_init: group is null");
    }
    if (devices.size() < 2) {
        throw std::invalid_argument("group_init: need at least 2 devices");
    }
    if (max_full_numel == 0) {
        throw std::invalid_argument("group_init: max_full_numel must be > 0");
    }
    if (max_full_numel % devices.size() != 0) {
        throw std::invalid_argument("group_init: max_full_numel must be divisible by world_size");
    }
    if (num_channel_slots <= 0) {
        throw std::invalid_argument("group_init: num_channel_slots must be > 0");
    }

    group_destroy(group);

    group->world_size = static_cast<int>(devices.size());
    group->devices = devices;
    group->streams.resize(devices.size(), nullptr);
    group->endpoints.resize(devices.size());
    group->max_full_numel = max_full_numel;
    group->max_shard_numel = max_full_numel / devices.size();
    group->num_channel_slots = num_channel_slots;
    group->channel_dispatch_capacity = channel_dispatch_capacity;
    group->channel_dispatch_chunk_bytes = channel_dispatch_chunk_bytes;

    group->channels.resize(static_cast<size_t>(group->world_size * group->world_size));
    group->channel_workers.resize(static_cast<size_t>(group->world_size * group->world_size));
    group->local_shard_buffers.resize(static_cast<size_t>(group->world_size));
    group->local_full_buffers.resize(static_cast<size_t>(group->world_size));

    const size_t full_bytes = group->max_full_numel * sizeof(half);
    const size_t shard_bytes = group->max_shard_numel * sizeof(half);
    const size_t signal_bytes = sizeof(uint64_t);

    for (int rank = 0; rank < group->world_size; ++rank) {
        system::runtime::ensure_context_on_device(group->devices[rank]);
        group->streams[rank] =
            system::runtime::create_stream_on_device(group->devices[rank]);

        Endpoint ep{};
        ep.rank = rank;
        ep.device = group->devices[rank];
        ep.stream = group->streams[rank];
        group->endpoints[static_cast<size_t>(rank)] = ep;
    }

    for (int rank = 0; rank < group->world_size; ++rank) {
        group->local_shard_buffers[rank] =
            transport::alloc_local_buffer_for_rank(group->devices, rank, shard_bytes);
        group->local_full_buffers[rank] =
            transport::alloc_local_buffer_for_rank(group->devices, rank, full_bytes);
    }

    for (int src = 0; src < group->world_size; ++src) {
        for (int dst = 0; dst < group->world_size; ++dst) {
            Channel& ch =
                group->channels[static_cast<size_t>(channel_index(group->world_size, src, dst))];
            ch.src_rank = src;
            ch.dst_rank = dst;
            ch.src_device = group->devices[src];
            ch.dst_device = group->devices[dst];
            ch.mode = channel_mode;
            ch.dispatch_queue_capacity = channel_dispatch_capacity;
            ch.dispatch_chunk_bytes = channel_dispatch_chunk_bytes;

            if (src == dst) {
                ch.slot_bytes = 0;
                ch.num_slots = 0;
                ch.dispatch_queue_capacity = 0;
                ch.dispatch_chunk_bytes = 0;
                ch.slots.clear();
                channel_worker_reset(
                    &group->channel_workers[static_cast<size_t>(
                        channel_index(group->world_size, src, dst))]);
                continue;
            }

            if (channel_mode == ChannelMode::kDirectReduce) {
                ch.slot_bytes = 0;
                ch.num_slots = 0;
                ch.slots.clear();
                transport::dispatch_queue_init(
                    group->devices,
                    &ch.dispatch_queue,
                    src,
                    src,
                    dst,
                    channel_dispatch_capacity);
                transport::direct_reduce_control_init(
                    group->devices,
                    &ch.direct_control,
                    dst,
                    src,
                    dst);
                continue;
            }

            ch.slot_bytes = shard_bytes;
            ch.num_slots = num_channel_slots;
            ch.slots.resize(static_cast<size_t>(num_channel_slots));

            for (int slot = 0; slot < num_channel_slots; ++slot) {
                ChannelSlot& slot_ref = ch.slots[static_cast<size_t>(slot)];
                slot_ref.slot_id = static_cast<uint32_t>(slot);
                slot_ref.buffer =
                    transport::alloc_peer_visible_buffer_for_rank(group->devices, dst, shard_bytes);
                slot_ref.signal_buffer =
                    transport::alloc_peer_visible_buffer_for_rank(group->devices, dst, signal_bytes);
                slot_ref.seq = 0;

                system::runtime::set_device(group->devices[dst]);
                system::runtime::check_cuda(
                    cudaMemset(slot_ref.signal_buffer.ptr, 0, slot_ref.signal_buffer.bytes),
                    "cudaMemset(channel slot signal)");
                transport::channel_slot_control_init(
                    group->devices,
                    &slot_ref.control,
                    dst,
                    src,
                    slot_ref.slot_id);
            }

            transport::dispatch_queue_init(
                group->devices,
                &ch.dispatch_queue,
                src,
                src,
                dst,
                channel_dispatch_capacity);
        }

        for (int src = 0; src < group->world_size; ++src) {
        for (int dst = 0; dst < group->world_size; ++dst) {
            if (src == dst) {
                continue;
            }

            Channel* ch = group_get_channel(group, src, dst);
            Endpoint* ep = group_get_endpoint(group, src);
            ChannelWorker* worker =
                &group->channel_workers[static_cast<size_t>(
                    channel_index(group->world_size, src, dst))];

            channel_worker_bind(worker, ep, ch);
        }
    }
    }

    return true;
}

void group_destroy(
    Group* group) {
    if (group == nullptr) {
        return;
    }

    for (auto& ch : group->channels) {
        for (auto& slot : ch.slots) {
            free_comm_buffer(group->devices, slot.buffer);
            free_comm_buffer(group->devices, slot.signal_buffer);
            transport::channel_slot_control_destroy(group->devices, &slot.control);
            slot.seq = 0;
            slot.slot_id = 0;
        }

        transport::direct_reduce_control_destroy(group->devices, &ch.direct_control);
        ch.slots.clear();
        transport::dispatch_queue_destroy(group->devices, &ch.dispatch_queue);
        ch.dispatch_queue_capacity = 0;
        ch.dispatch_chunk_bytes = 0;
        ch.src_rank = -1;
        ch.dst_rank = -1;
        ch.src_device = -1;
        ch.dst_device = -1;
        ch.mode = ChannelMode::kSlotQueue;
        ch.slot_bytes = 0;
        ch.num_slots = 0;
    }

    for (auto& buf : group->local_shard_buffers) {
        free_comm_buffer(group->devices, buf);
    }
    for (auto& buf : group->local_full_buffers) {
        free_comm_buffer(group->devices, buf);
    }

    for (size_t i = 0; i < group->streams.size(); ++i) {
        if (group->streams[i] != nullptr) {
            system::runtime::destroy_stream_on_device(group->devices[i], group->streams[i]);
        }
    }

    for (auto& worker : group->channel_workers) {
        channel_worker_reset(&worker);
    }

    group->world_size = 0;
    group->devices.clear();
    group->streams.clear();
    group->endpoints.clear();
    group->max_full_numel = 0;
    group->max_shard_numel = 0;
    group->num_channel_slots = 0;
    group->channel_dispatch_capacity = 0;
    group->channel_dispatch_chunk_bytes = 0;
    group->channels.clear();
    group->local_shard_buffers.clear();
    group->local_full_buffers.clear();
    group->channel_workers.clear();
}

Endpoint* group_get_endpoint(
    Group* group,
    int rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_endpoint: group is null");
    }
    validate_rank_or_throw(group->world_size, rank, "group_get_endpoint: invalid rank");
    return &group->endpoints[static_cast<size_t>(rank)];
}

const Endpoint* group_get_endpoint(
    const Group* group,
    int rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_endpoint: group is null");
    }
    validate_rank_or_throw(group->world_size, rank, "group_get_endpoint: invalid rank");
    return &group->endpoints[static_cast<size_t>(rank)];
}

Channel* group_get_channel(
    Group* group,
    int src_rank,
    int dst_rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_channel: group is null");
    }
    validate_rank_or_throw(group->world_size, src_rank, "group_get_channel: invalid src_rank");
    validate_rank_or_throw(group->world_size, dst_rank, "group_get_channel: invalid dst_rank");
    if (src_rank == dst_rank) {
        return nullptr;
    }
    return &group->channels[static_cast<size_t>(channel_index(group->world_size, src_rank, dst_rank))];
}

const Channel* group_get_channel(
    const Group* group,
    int src_rank,
    int dst_rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_channel: group is null");
    }
    validate_rank_or_throw(group->world_size, src_rank, "group_get_channel: invalid src_rank");
    validate_rank_or_throw(group->world_size, dst_rank, "group_get_channel: invalid dst_rank");
    if (src_rank == dst_rank) {
        return nullptr;
    }
    return &group->channels[static_cast<size_t>(channel_index(group->world_size, src_rank, dst_rank))];
}

ChannelWorker* group_get_channel_worker(
    Group* group,
    int src_rank,
    int dst_rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_channel_worker: group is null");
    }
    validate_rank_or_throw(group->world_size, src_rank, "group_get_channel_worker: invalid src_rank");
    validate_rank_or_throw(group->world_size, dst_rank, "group_get_channel_worker: invalid dst_rank");
    if (src_rank == dst_rank) {
        return nullptr;
    }
    return &group->channel_workers[static_cast<size_t>(
        channel_index(group->world_size, src_rank, dst_rank))];
}

const ChannelWorker* group_get_channel_worker(
    const Group* group,
    int src_rank,
    int dst_rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_channel_worker: group is null");
    }
    validate_rank_or_throw(group->world_size, src_rank, "group_get_channel_worker: invalid src_rank");
    validate_rank_or_throw(group->world_size, dst_rank, "group_get_channel_worker: invalid dst_rank");
    if (src_rank == dst_rank) {
        return nullptr;
    }
    return &group->channel_workers[static_cast<size_t>(
        channel_index(group->world_size, src_rank, dst_rank))];
}

transport::CommBuffer* group_get_local_shard_buffer(
    Group* group,
    int rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_local_shard_buffer: group is null");
    }
    validate_rank_or_throw(group->world_size, rank, "group_get_local_shard_buffer: invalid rank");
    return &group->local_shard_buffers[static_cast<size_t>(rank)];
}

const transport::CommBuffer* group_get_local_shard_buffer(
    const Group* group,
    int rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_local_shard_buffer: group is null");
    }
    validate_rank_or_throw(group->world_size, rank, "group_get_local_shard_buffer: invalid rank");
    return &group->local_shard_buffers[static_cast<size_t>(rank)];
}

transport::CommBuffer* group_get_local_full_buffer(
    Group* group,
    int rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_local_full_buffer: group is null");
    }
    validate_rank_or_throw(group->world_size, rank, "group_get_local_full_buffer: invalid rank");
    return &group->local_full_buffers[static_cast<size_t>(rank)];
}

const transport::CommBuffer* group_get_local_full_buffer(
    const Group* group,
    int rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_local_full_buffer: group is null");
    }
    validate_rank_or_throw(group->world_size, rank, "group_get_local_full_buffer: invalid rank");
    return &group->local_full_buffers[static_cast<size_t>(rank)];
}

// -------------------------
// Compatibility wrappers
// -------------------------

bool communicator_init(
    Communicator* comm,
    const std::vector<int>& devices,
    size_t max_full_numel,
    int num_channel_slots) {
    return group_init(
        comm,
        devices,
        max_full_numel,
        num_channel_slots,
        ChannelMode::kSlotQueue);
}

void communicator_destroy(
    Communicator* comm) {
    group_destroy(comm);
}

CommChannel* communicator_get_channel(
    Communicator* comm,
    int src_rank,
    int dst_rank) {
    return group_get_channel(comm, src_rank, dst_rank);
}

const CommChannel* communicator_get_channel(
    const Communicator* comm,
    int src_rank,
    int dst_rank) {
    return group_get_channel(comm, src_rank, dst_rank);
}

transport::CommBuffer* channel_get_slot_buffer(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx) {
    CommChannel* ch = communicator_get_channel(comm, src_rank, dst_rank);
    return ooverlap::comm::channel_get_slot_buffer(ch, slot_idx);
}

const transport::CommBuffer* channel_get_slot_buffer(
    const Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx) {
    const CommChannel* ch = communicator_get_channel(comm, src_rank, dst_rank);
    return ooverlap::comm::channel_get_slot_buffer(ch, slot_idx);
}

transport::CommBuffer* channel_get_slot_signal_buffer(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx) {
    CommChannel* ch = communicator_get_channel(comm, src_rank, dst_rank);
    return ooverlap::comm::channel_get_slot_signal_buffer(ch, slot_idx);
}

const transport::CommBuffer* channel_get_slot_signal_buffer(
    const Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx) {
    const CommChannel* ch = communicator_get_channel(comm, src_rank, dst_rank);
    return ooverlap::comm::channel_get_slot_signal_buffer(ch, slot_idx);
}

transport::CommBuffer* communicator_get_local_shard_buffer(
    Communicator* comm,
    int rank) {
    return group_get_local_shard_buffer(comm, rank);
}

const transport::CommBuffer* communicator_get_local_shard_buffer(
    const Communicator* comm,
    int rank) {
    return group_get_local_shard_buffer(comm, rank);
}

transport::CommBuffer* communicator_get_local_full_buffer(
    Communicator* comm,
    int rank) {
    return group_get_local_full_buffer(comm, rank);
}

const transport::CommBuffer* communicator_get_local_full_buffer(
    const Communicator* comm,
    int rank) {
    return group_get_local_full_buffer(comm, rank);
}

} // namespace comm
} // namespace ooverlap
