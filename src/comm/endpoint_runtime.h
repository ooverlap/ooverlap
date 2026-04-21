#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "comm/collective/operation.h"
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

    collective::OperationDesc* operations = nullptr;
    int num_operations = 0;

    exec::SchedulerQueueBinding* scheduler_bindings = nullptr;
    int num_scheduler_bindings = 0;
};

struct EndpointRuntime {
    Endpoint endpoint{};

    // Host mirrors.
    std::vector<collective::ReadyTileQueue> input_queues_host;
    std::vector<collective::OperationDesc> operations_host;
    std::vector<exec::SchedulerQueueBinding> scheduler_bindings_host;

    // Device storage / tables.
    collective::ReadyTileQueue* input_queues_device = nullptr;
    collective::OperationTable operation_table{};
    exec::SchedulerQueueBinding* scheduler_bindings_device = nullptr;

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
           rt->operations != nullptr &&
           rt->num_operations > 0 &&
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

bool endpoint_runtime_configure_queue_operation(
    EndpointRuntime* rt,
    int queue_idx,
    uint32_t op_id,
    int dst_rank,
    void* dst_ptr,
    size_t dst_bytes,
    size_t tile_stride_bytes,
    uint32_t expected_contributions,
    exec::ChunkOpKind op,
    uint32_t epoch = 1,
    bool enabled = true);

bool endpoint_runtime_clear_queue_operation(
    EndpointRuntime* rt,
    int queue_idx);

inline const DeviceEndpointRuntime* endpoint_runtime_device_handle(
    const EndpointRuntime* rt) {
    return (rt != nullptr) ? &rt->device : nullptr;
}

} // namespace comm
} // namespace ooverlap
