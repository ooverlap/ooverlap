#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/transport/buffer.h"
#include "comm/channel.h"
#include "comm/channel_worker.h"
#include "comm/endpoint.h"

namespace ooverlap {
namespace comm {

struct Group {
    int world_size = 0;

    std::vector<int> devices;
    std::vector<cudaStream_t> streams;
    std::vector<Endpoint> endpoints;

    size_t max_full_numel = 0;
    size_t max_shard_numel = 0;
    int num_channel_slots = 0;

    uint32_t channel_dispatch_capacity = 0;
    size_t channel_dispatch_chunk_bytes = 0;

    std::vector<Channel> channels;
    std::vector<ChannelWorker> channel_workers;

    std::vector<transport::CommBuffer> local_shard_buffers;
    std::vector<transport::CommBuffer> local_full_buffers;
};

bool group_init(
    Group* group,
    const std::vector<int>& devices,
    size_t max_full_numel,
    int num_channel_slots = 1,
    ChannelMode channel_mode = ChannelMode::kSlotQueue,
    uint32_t channel_dispatch_capacity = 1024,
    size_t channel_dispatch_chunk_bytes = 16 * 1024);

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

ChannelWorker* group_get_channel_worker(
    Group* group,
    int src_rank,
    int dst_rank);

const ChannelWorker* group_get_channel_worker(
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
