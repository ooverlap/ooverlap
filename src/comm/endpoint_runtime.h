#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "comm/collective/operation.h"
#include "comm/endpoint.h"
#include "comm/exec/chunk.h"
#include "comm/group.h"

namespace ooverlap {
namespace comm {

struct DeviceEndpointRuntime {
    int rank = -1;
    int world_size = 0;
    int device = -1;
    cudaStream_t stream = nullptr;
};

struct EndpointRuntime {
    Endpoint endpoint{};
    int world_size = 0;

    DeviceEndpointRuntime device{};
};

__host__ __device__ __forceinline__ bool device_endpoint_runtime_is_valid(
    const DeviceEndpointRuntime* rt) {
    return rt != nullptr &&
           rt->rank >= 0 &&
           rt->world_size > 0 &&
           rt->device >= 0 &&
           rt->stream != nullptr;
}

bool endpoint_runtime_init(
    EndpointRuntime* rt,
    const Group* group,
    int rank);

void endpoint_runtime_destroy(
    EndpointRuntime* rt);

void endpoint_runtime_refresh_device(
    EndpointRuntime* rt);

bool endpoint_runtime_configure_submission(
    EndpointRuntime* rt,
    const void* src_ptr,
    void* dst_ptr,
    size_t bytes,
    exec::ChunkOpKind op,
    uint32_t op_id,
    int dst_rank,
    uint64_t user_tag = 0);

bool endpoint_runtime_clear_submission(
    EndpointRuntime* rt);

bool endpoint_runtime_build_ring_allreduce_operation(
    const EndpointRuntime* rt,
    collective::OperationDesc* out,
    void* accum_ptr,
    void* next_accum_ptr,
    size_t total_bytes,
    size_t chunk_bytes,
    void* inbound_steps_ptr,
    void* next_inbound_steps_ptr,
    void* done_ptr,
    void* next_done_ptr,
    collective::ChunkState* chunk_states_ptr,
    uint32_t op_id,
    uint32_t epoch = 1,
    bool enabled = true,
    uint64_t user_tag = 0);

inline const DeviceEndpointRuntime* endpoint_runtime_device_handle(
    const EndpointRuntime* rt) {
    return (rt != nullptr) ? &rt->device : nullptr;
}

} // namespace comm
} // namespace ooverlap
