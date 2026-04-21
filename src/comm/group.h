#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/transport/buffer.h"
#include "comm/channel.h"
#include "comm/endpoint.h"

namespace ooverlap {
namespace comm {

struct Group {
    int world_size = 0;

    // Kept for compatibility with current code.
    std::vector<int> devices;
    std::vector<cudaStream_t> streams;

    // New top-level per-GPU primitive.
    std::vector<Endpoint> endpoints;

    size_t max_full_numel = 0;
    size_t max_shard_numel = 0;
    int num_channel_slots = 0;

    // Dense matrix layout: channels[src_rank * world_size + dst_rank]
    std::vector<Channel> channels;

    // Local, device-owned scratch buffers.
    std::vector<transport::CommBuffer> local_shard_buffers;
    std::vector<transport::CommBuffer> local_full_buffers;
};

bool group_init(
    Group* group,
    const std::vector<int>& devices,
    size_t max_full_numel,
    int num_channel_slots = 1,
    ChannelMode channel_mode = ChannelMode::kSlotQueue);

void group_destroy(
    Group* group);

Endpoint* group_get_endpoint(
    Group* group,
    int rank);

const Endpoint* group_get_endpoint(
    const Group* group,
    int rank);

Channel* group_get_channel(
    Group* group,
    int src_rank,
    int dst_rank);

const Channel* group_get_channel(
    const Group* group,
    int src_rank,
    int dst_rank);

transport::CommBuffer* group_get_local_shard_buffer(
    Group* group,
    int rank);

const transport::CommBuffer* group_get_local_shard_buffer(
    const Group* group,
    int rank);

transport::CommBuffer* group_get_local_full_buffer(
    Group* group,
    int rank);

const transport::CommBuffer* group_get_local_full_buffer(
    const Group* group,
    int rank);

} // namespace comm
} // namespace ooverlap
