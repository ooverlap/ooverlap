#include "overlap/comm/tma_communicator.h"

#include "overlap/bulk_tma_copy_sm90.cuh"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/system/peer_buffer.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace comm {

namespace {
inline int channel_index(const TmaCommunicator* comm, int src_rank, int dst_rank) {
    return src_rank * comm->world_size + dst_rank;
}

inline void validate_rank(const TmaCommunicator* comm, int rank, const char* what) {
    if (comm == nullptr) {
        throw std::invalid_argument("communicator is null");
    }
    if (rank < 0 || rank >= comm->world_size) {
        throw std::invalid_argument(what);
    }
}

inline Buffer alloc_peer_visible_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes) {

    system::mapped_peer_buffer mapped =
        system::alloc_peer_visible_buffer(bytes, devices[owner_rank], devices);

    Buffer out{};
    out.ptr = mapped.ptr;
    out.bytes = mapped.mapped_size;
    out.owner_rank = owner_rank;
    out.peer_visible = true;
    return out;
}

inline Buffer alloc_local_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes) {

    Buffer out{};
    out.bytes = bytes;
    out.owner_rank = owner_rank;
    out.peer_visible = false;

    system::runtime::set_device(devices[owner_rank]);
    system::runtime::check_cuda(cudaMalloc(&out.ptr, bytes), "cudaMalloc(local buffer)");
    return out;
}

inline void free_buffer(
    const std::vector<int>& devices,
    Buffer& buf) {
    if (buf.ptr == nullptr) {
        return;
    }

    if (buf.peer_visible) {
        system::mapped_peer_buffer mapped{};
        mapped.ptr = buf.ptr;
        mapped.mapped_size = buf.bytes;
        system::free_peer_visible_buffer(mapped);
    } else {
        system::runtime::set_device(devices[buf.owner_rank]);
        system::runtime::check_cuda(cudaFree(buf.ptr), "cudaFree(local buffer)");
    }

    buf.ptr = nullptr;
    buf.bytes = 0;
    buf.owner_rank = -1;
    buf.peer_visible = false;
}

} // namespace

bool communicator_init(
    TmaCommunicator* comm,
    const std::vector<int>& devices,
    size_t max_full_numel,
    int num_channel_slots) {

    if (comm == nullptr) {
        throw std::invalid_argument("communicator_init: comm is null");
    }
    if (devices.size() < 2) {
        throw std::invalid_argument("communicator_init: need at least 2 devices");
    }
    if (max_full_numel == 0) {
        throw std::invalid_argument("communicator_init: max_full_numel must be > 0");
    }
    if (max_full_numel % devices.size() != 0) {
        throw std::invalid_argument("communicator_init: max_full_numel must be divisible by world_size");
    }
    if (num_channel_slots <= 0) {
        throw std::invalid_argument("communicator_init: num_channel_slots must be > 0");
    }

    communicator_destroy(comm);

    comm->world_size = static_cast<int>(devices.size());
    comm->devices = devices;
    comm->streams.resize(devices.size(), nullptr);
    comm->max_full_numel = max_full_numel;
    comm->max_shard_numel = max_full_numel / devices.size();
    comm->num_channel_slots = num_channel_slots;

    comm->channels.resize(static_cast<size_t>(comm->world_size * comm->world_size));
    comm->local_shard_buffers.resize(static_cast<size_t>(comm->world_size));
    comm->local_full_buffers.resize(static_cast<size_t>(comm->world_size));

    const size_t full_bytes = comm->max_full_numel * sizeof(half);
    const size_t shard_bytes = comm->max_shard_numel * sizeof(half);
    const size_t signal_bytes = sizeof(uint64_t);

    for (int rank = 0; rank < comm->world_size; ++rank) {
        system::runtime::ensure_context_on_device(comm->devices[rank]);
        comm->streams[rank] = system::runtime::create_stream_on_device(comm->devices[rank]);
    }

    for (int rank = 0; rank < comm->world_size; ++rank) {
        comm->local_shard_buffers[rank] =
            alloc_local_buffer_for_rank(comm->devices, rank, shard_bytes);
        comm->local_full_buffers[rank] =
            alloc_local_buffer_for_rank(comm->devices, rank, full_bytes);
    }

    for (int src = 0; src < comm->world_size; ++src) {
        for (int dst = 0; dst < comm->world_size; ++dst) {
            Channel& ch = comm->channels[static_cast<size_t>(channel_index(comm, src, dst))];
            ch.src_rank = src;
            ch.dst_rank = dst;

            if (src == dst) {
                ch.slot_bytes = 0;
                ch.num_slots = 0;
                ch.slots.clear();
                continue;
            }

            ch.slot_bytes = shard_bytes;
            ch.num_slots = num_channel_slots;
            ch.slots.resize(static_cast<size_t>(num_channel_slots));

            for (int slot = 0; slot < num_channel_slots; ++slot) {
                ChannelSlot& slot_ref = ch.slots[static_cast<size_t>(slot)];
                slot_ref.buffer =
                    alloc_peer_visible_buffer_for_rank(comm->devices, dst, shard_bytes);
                slot_ref.signal_buffer =
                    alloc_peer_visible_buffer_for_rank(comm->devices, dst, signal_bytes);
                slot_ref.seq = 0;

                system::runtime::set_device(comm->devices[dst]);
                system::runtime::check_cuda(
                    cudaMemset(slot_ref.signal_buffer.ptr, 0, slot_ref.signal_buffer.bytes),
                    "cudaMemset(channel slot signal)");
            }
        }
    }

    return true;
}

void communicator_destroy(TmaCommunicator* comm) {
    if (comm == nullptr) {
        return;
    }

    for (auto& ch : comm->channels) {
        for (auto& slot : ch.slots) {
            free_buffer(comm->devices, slot.buffer);
            free_buffer(comm->devices, slot.signal_buffer);
            slot.seq = 0;
        }
        ch.slots.clear();
        ch.src_rank = -1;
        ch.dst_rank = -1;
        ch.slot_bytes = 0;
        ch.num_slots = 0;
    }

    for (auto& buf : comm->local_shard_buffers) {
        free_buffer(comm->devices, buf);
    }
    for (auto& buf : comm->local_full_buffers) {
        free_buffer(comm->devices, buf);
    }

    for (size_t i = 0; i < comm->streams.size(); ++i) {
        if (comm->streams[i] != nullptr) {
            system::runtime::destroy_stream_on_device(comm->devices[i], comm->streams[i]);
        }
    }

    comm->world_size = 0;
    comm->devices.clear();
    comm->streams.clear();
    comm->max_full_numel = 0;
    comm->max_shard_numel = 0;
    comm->num_channel_slots = 0;
    comm->channels.clear();
    comm->local_shard_buffers.clear();
    comm->local_full_buffers.clear();
}

Channel* communicator_get_channel(
    TmaCommunicator* comm,
    int src_rank,
    int dst_rank) {
    validate_rank(comm, src_rank, "communicator_get_channel: invalid src_rank");
    validate_rank(comm, dst_rank, "communicator_get_channel: invalid dst_rank");
    if (src_rank == dst_rank) {
        return nullptr;
    }
    return &comm->channels[static_cast<size_t>(channel_index(comm, src_rank, dst_rank))];
}

const Channel* communicator_get_channel(
    const TmaCommunicator* comm,
    int src_rank,
    int dst_rank) {
    validate_rank(comm, src_rank, "communicator_get_channel: invalid src_rank");
    validate_rank(comm, dst_rank, "communicator_get_channel: invalid dst_rank");
    if (src_rank == dst_rank) {
        return nullptr;
    }
    return &comm->channels[static_cast<size_t>(channel_index(comm, src_rank, dst_rank))];
}

Buffer* channel_get_slot_buffer(
    TmaCommunicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx) {
    Channel* ch = communicator_get_channel(comm, src_rank, dst_rank);
    if (ch == nullptr) {
        return nullptr;
    }
    if (slot_idx < 0 || slot_idx >= ch->num_slots) {
        throw std::invalid_argument("channel_get_slot_buffer: invalid slot_idx");
    }
    return &ch->slots[static_cast<size_t>(slot_idx)].buffer;
}

const Buffer* channel_get_slot_buffer(
    const TmaCommunicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx) {
    const Channel* ch = communicator_get_channel(comm, src_rank, dst_rank);
    if (ch == nullptr) {
        return nullptr;
    }
    if (slot_idx < 0 || slot_idx >= ch->num_slots) {
        throw std::invalid_argument("channel_get_slot_buffer: invalid slot_idx");
    }
    return &ch->slots[static_cast<size_t>(slot_idx)].buffer;
}

Buffer* channel_get_slot_signal_buffer(
    TmaCommunicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx) {
    Channel* ch = communicator_get_channel(comm, src_rank, dst_rank);
    if (ch == nullptr) {
        return nullptr;
    }
    if (slot_idx < 0 || slot_idx >= ch->num_slots) {
        throw std::invalid_argument("channel_get_slot_signal_buffer: invalid slot_idx");
    }
    return &ch->slots[static_cast<size_t>(slot_idx)].signal_buffer;
}

const Buffer* channel_get_slot_signal_buffer(
    const TmaCommunicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx) {
    const Channel* ch = communicator_get_channel(comm, src_rank, dst_rank);
    if (ch == nullptr) {
        return nullptr;
    }
    if (slot_idx < 0 || slot_idx >= ch->num_slots) {
        throw std::invalid_argument("channel_get_slot_signal_buffer: invalid slot_idx");
    }
    return &ch->slots[static_cast<size_t>(slot_idx)].signal_buffer;
}

Buffer* communicator_get_local_shard_buffer(
    TmaCommunicator* comm,
    int rank) {
    validate_rank(comm, rank, "communicator_get_local_shard_buffer: invalid rank");
    return &comm->local_shard_buffers[static_cast<size_t>(rank)];
}

const Buffer* communicator_get_local_shard_buffer(
    const TmaCommunicator* comm,
    int rank) {
    validate_rank(comm, rank, "communicator_get_local_shard_buffer: invalid rank");
    return &comm->local_shard_buffers[static_cast<size_t>(rank)];
}

Buffer* communicator_get_local_full_buffer(
    TmaCommunicator* comm,
    int rank) {
    validate_rank(comm, rank, "communicator_get_local_full_buffer: invalid rank");
    return &comm->local_full_buffers[static_cast<size_t>(rank)];
}

const Buffer* communicator_get_local_full_buffer(
    const TmaCommunicator* comm,
    int rank) {
    validate_rank(comm, rank, "communicator_get_local_full_buffer: invalid rank");
    return &comm->local_full_buffers[static_cast<size_t>(rank)];
}

cudaError_t channel_send_bulk_tma(
    TmaCommunicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx,
    const half* src,
    size_t numel,
    cudaStream_t stream) {

    if (src == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    Buffer* dst_buf = channel_get_slot_buffer(comm, src_rank, dst_rank, slot_idx);
    if (dst_buf == nullptr) {
        return cudaErrorInvalidValue;
    }

    const size_t bytes = numel * sizeof(half);
    if (bytes > dst_buf->bytes) {
        return cudaErrorInvalidValue;
    }

    return enqueue_bulk_tma_copy_sm90(src, buffer_as_half(dst_buf), numel, stream);
}

} // namespace comm
} // namespace ooverlap
