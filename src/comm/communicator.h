#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/buffer.h"
#include "comm/channel.h"

namespace ooverlap {
namespace comm {

struct Communicator {
    int world_size = 0;
    std::vector<int> devices;
    std::vector<cudaStream_t> streams;

    size_t max_full_numel = 0;
    size_t max_shard_numel = 0;
    int num_channel_slots = 0;

    // Dense matrix layout: channels[src_rank * world_size + dst_rank]
    std::vector<CommChannel> channels;

    // Local, device-owned scratch buffers.
    std::vector<CommBuffer> local_shard_buffers;
    std::vector<CommBuffer> local_full_buffers;
};

bool communicator_init(
    Communicator* comm,
    const std::vector<int>& devices,
    size_t max_full_numel,
    int num_channel_slots = 1);

void communicator_destroy(Communicator* comm);

CommChannel* communicator_get_channel(
    Communicator* comm,
    int src_rank,
    int dst_rank);

const CommChannel* communicator_get_channel(
    const Communicator* comm,
    int src_rank,
    int dst_rank);

CommBuffer* channel_get_slot_buffer(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

const CommBuffer* channel_get_slot_buffer(
    const Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

CommBuffer* channel_get_slot_signal_buffer(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

const CommBuffer* channel_get_slot_signal_buffer(
    const Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

CommBuffer* communicator_get_local_shard_buffer(
    Communicator* comm,
    int rank);

const CommBuffer* communicator_get_local_shard_buffer(
    const Communicator* comm,
    int rank);

CommBuffer* communicator_get_local_full_buffer(
    Communicator* comm,
    int rank);

const CommBuffer* communicator_get_local_full_buffer(
    const Communicator* comm,
    int rank);

cudaError_t channel_send_bulk_tma(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx,
    const half* src,
    size_t numel,
    cudaStream_t stream);

// Compatibility aliases for current code.
using TmaCommunicator = Communicator;

} // namespace comm
} // namespace ooverlap
