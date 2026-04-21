#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "comm/channel.h"
#include "comm/collective/published_tile.h"
#include "comm/endpoint.h"
#include "comm/exec/chunk_scheduler.h"
#include "comm/group.h"

namespace ooverlap {
namespace comm {

struct DeviceEndpointRuntime {
    int rank = -1;
    int device = -1;
    cudaStream_t stream = nullptr;

    collective::ReadyTileQueue* input_queues = nullptr;
    int num_input_queues = 0;

    exec::SchedulerQueueBinding* scheduler_bindings = nullptr;
    int num_scheduler_bindings = 0;

    Channel* outgoing_channels = nullptr;
    int num_outgoing_channels = 0;
};

struct EndpointRuntime {
    Endpoint endpoint{};

    // Host-owned mirrors.
    std::vector<collective::ReadyTileQueue> input_queues_host;
    std::vector<exec::SchedulerQueueBinding> scheduler_bindings_host;
    std::vector<Channel> outgoing_channels_host;

    // Device arrays containing copies of the host mirrors above.
    collective::ReadyTileQueue* input_queues_device = nullptr;
    exec::SchedulerQueueBinding* scheduler_bindings_device = nullptr;
    Channel* outgoing_channels_device = nullptr;

    DeviceEndpointRuntime device{};
};

__host__ __device__ __forceinline__ bool device_endpoint_runtime_is_valid(
    const DeviceEndpointRuntime* rt) {
    return rt != nullptr &&
           rt->rank >= 0 &&
           rt->device >= 0 &&
           rt->stream != nullptr &&
           rt->input_queues != nullptr &&
           rt->num_input_queues > 0 &&
           rt->scheduler_bindings != nullptr &&
           rt->num_scheduler_bindings > 0;
}

bool endpoint_runtime_init(
    EndpointRuntime* rt,
    const Group* group,
    int rank,
    int num_input_queues,
    uint32_t input_queue_capacity);

void endpoint_runtime_destroy(
    EndpointRuntime* rt);

void endpoint_runtime_refresh_device(
    EndpointRuntime* rt);

int endpoint_runtime_find_outgoing_channel_index(
    const EndpointRuntime* rt,
    int dst_rank);

bool endpoint_runtime_bind_queue_to_dst(
    EndpointRuntime* rt,
    int queue_idx,
    int dst_rank,
    exec::ChunkOpKind op);

bool endpoint_runtime_unbind_queue(
    EndpointRuntime* rt,
    int queue_idx);

inline const DeviceEndpointRuntime* endpoint_runtime_device_handle(
    const EndpointRuntime* rt) {
    return (rt != nullptr) ? &rt->device : nullptr;
}

} // namespace comm
} // namespace ooverlap
