#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ooverlap {

struct Buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    int owner_rank = -1;
    bool peer_visible = false;
};

struct ChannelSlot {
    Buffer buffer;
    uint64_t seq = 0;
};

struct Channel {
    int src_rank = -1;
    int dst_rank = -1;
    size_t slot_bytes = 0;
    int num_slots = 0;
    std::vector<ChannelSlot> slots;
};

struct TmaCommunicator {
    int world_size = 0;
    std::vector<int> devices;
    std::vector<cudaStream_t> streams;

    size_t max_full_numel = 0;
    size_t max_shard_numel = 0;
    int num_channel_slots = 0;

    // Dense matrix layout: channels[src_rank * world_size + dst_rank]
    std::vector<Channel> channels;

    // Local, device-owned scratch buffers.
    std::vector<Buffer> local_shard_buffers;
    std::vector<Buffer> local_full_buffers;
};

bool communicator_init(
    TmaCommunicator* comm,
    const std::vector<int>& devices,
    size_t max_full_numel,
    int num_channel_slots = 1);

void communicator_destroy(TmaCommunicator* comm);

Channel* communicator_get_channel(
    TmaCommunicator* comm,
    int src_rank,
    int dst_rank);

const Channel* communicator_get_channel(
    const TmaCommunicator* comm,
    int src_rank,
    int dst_rank);

Buffer* channel_get_slot_buffer(
    TmaCommunicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

const Buffer* channel_get_slot_buffer(
    const TmaCommunicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

Buffer* communicator_get_local_shard_buffer(
    TmaCommunicator* comm,
    int rank);

const Buffer* communicator_get_local_shard_buffer(
    const TmaCommunicator* comm,
    int rank);

Buffer* communicator_get_local_full_buffer(
    TmaCommunicator* comm,
    int rank);

const Buffer* communicator_get_local_full_buffer(
    const TmaCommunicator* comm,
    int rank);

inline half* buffer_as_half(Buffer* buf) {
    return reinterpret_cast<half*>(buf->ptr);
}

inline const half* buffer_as_half(const Buffer* buf) {
    return reinterpret_cast<const half*>(buf->ptr);
}

cudaError_t channel_send_bulk_tma(
    TmaCommunicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx,
    const half* src,
    size_t numel,
    cudaStream_t stream);

} // namespace ooverlap
