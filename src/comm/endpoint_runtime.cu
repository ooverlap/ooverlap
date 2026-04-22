#include "comm/endpoint_runtime.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace {

void endpoint_runtime_reset_device_handle(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        return;
    }

    rt->device.rank = -1;
    rt->device.world_size = 0;
    rt->device.device = -1;
    rt->device.stream = nullptr;
}

void endpoint_runtime_reset_host_state(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        return;
    }

    rt->endpoint = Endpoint{};
    rt->world_size = 0;
    endpoint_runtime_reset_device_handle(rt);
}

} // namespace

bool endpoint_runtime_init(
    EndpointRuntime* rt,
    const Group* group,
    int rank) {
    if (rt == nullptr) {
        throw std::invalid_argument("endpoint_runtime_init: rt is null");
    }
    if (group == nullptr) {
        throw std::invalid_argument("endpoint_runtime_init: group is null");
    }
    if (rank < 0 || rank >= group->world_size) {
        throw std::invalid_argument("endpoint_runtime_init: invalid rank");
    }

    endpoint_runtime_destroy(rt);

    rt->endpoint = *group_get_endpoint(group, rank);
    rt->world_size = group->world_size;

    endpoint_runtime_refresh_device(rt);
    return true;
}

void endpoint_runtime_destroy(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        return;
    }

    endpoint_runtime_reset_host_state(rt);
}

void endpoint_runtime_refresh_device(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: rt is null");
    }
    if (!endpoint_is_valid(rt->endpoint)) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: endpoint is invalid");
    }
    if (rt->world_size <= 0) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: world_size is invalid");
    }

    rt->device.rank = rt->endpoint.rank;
    rt->device.world_size = rt->world_size;
    rt->device.device = rt->endpoint.device;
    rt->device.stream = rt->endpoint.stream;
}

bool endpoint_runtime_configure_submission(
    EndpointRuntime* rt,
    const void* src_ptr,
    void* dst_ptr,
    size_t bytes,
    exec::ChunkOpKind op,
    uint32_t op_id,
    int dst_rank,
    uint64_t user_tag) {
    if (rt == nullptr) {
        throw std::invalid_argument("endpoint_runtime_configure_submission: rt is null");
    }
    if (!endpoint_is_valid(rt->endpoint)) {
        throw std::invalid_argument("endpoint_runtime_configure_submission: endpoint is invalid");
    }
    if (src_ptr == nullptr) {
        throw std::invalid_argument("endpoint_runtime_configure_submission: src_ptr is null");
    }
    if (dst_ptr == nullptr) {
        throw std::invalid_argument("endpoint_runtime_configure_submission: dst_ptr is null");
    }
    if (bytes == 0) {
        throw std::invalid_argument("endpoint_runtime_configure_submission: bytes must be > 0");
    }
    if (op == exec::ChunkOpKind::kInvalid) {
        throw std::invalid_argument("endpoint_runtime_configure_submission: invalid op");
    }
    if (op_id == 0) {
        throw std::invalid_argument("endpoint_runtime_configure_submission: op_id must be > 0");
    }
    if (dst_rank < 0) {
        throw std::invalid_argument("endpoint_runtime_configure_submission: invalid dst_rank");
    }

    (void)src_ptr;
    (void)dst_ptr;
    (void)bytes;
    (void)op;
    (void)op_id;
    (void)dst_rank;
    (void)user_tag;
    return true;
}

bool endpoint_runtime_clear_submission(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        throw std::invalid_argument("endpoint_runtime_clear_submission: rt is null");
    }
    return true;
}

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
    uint32_t epoch,
    bool enabled,
    uint64_t user_tag) {
    if (rt == nullptr) {
        throw std::invalid_argument("endpoint_runtime_build_ring_allreduce_operation: rt is null");
    }
    if (!endpoint_is_valid(rt->endpoint)) {
        throw std::invalid_argument("endpoint_runtime_build_ring_allreduce_operation: endpoint is invalid");
    }
    if (rt->world_size <= 0) {
        throw std::invalid_argument("endpoint_runtime_build_ring_allreduce_operation: world_size is invalid");
    }

    return collective::operation_desc_build_ring_allreduce(
        out,
        rt->endpoint.rank,
        rt->world_size,
        total_bytes,
        chunk_bytes,
        exec::ChunkOpKind::kReduceAddNoFtzF16,
        accum_ptr,
        next_accum_ptr,
        inbound_steps_ptr,
        next_inbound_steps_ptr,
        done_ptr,
        next_done_ptr,
        chunk_states_ptr,
        op_id,
        epoch,
        enabled,
        user_tag);
}

} // namespace comm
} // namespace ooverlap
