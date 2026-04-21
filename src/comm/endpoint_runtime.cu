#include "comm/endpoint_runtime.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace comm {
namespace {

void endpoint_runtime_reset_device_handle(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        return;
    }

    rt->device.rank = -1;
    rt->device.device = -1;
    rt->device.stream = nullptr;

    rt->device.input_queues = nullptr;
    rt->device.num_input_queues = 0;

    rt->device.scheduler_bindings = nullptr;
    rt->device.num_scheduler_bindings = 0;

    rt->device.outgoing_channels = nullptr;
    rt->device.num_outgoing_channels = 0;
}

void endpoint_runtime_reset_host_state(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        return;
    }

    rt->endpoint = Endpoint{};
    rt->input_queues_host.clear();
    rt->scheduler_bindings_host.clear();
    rt->outgoing_channels_host.clear();

    rt->input_queues_device = nullptr;
    rt->scheduler_bindings_device = nullptr;
    rt->outgoing_channels_device = nullptr;

    endpoint_runtime_reset_device_handle(rt);
}

void endpoint_runtime_validate_queue_index(
    const EndpointRuntime* rt,
    int queue_idx,
    const char* what) {
    if (rt == nullptr) {
        throw std::invalid_argument(what);
    }
    if (queue_idx < 0 || queue_idx >= static_cast<int>(rt->input_queues_host.size())) {
        throw std::invalid_argument(what);
    }
}

} // namespace

bool endpoint_runtime_init(
    EndpointRuntime* rt,
    const Group* group,
    int rank,
    int num_input_queues,
    uint32_t input_queue_capacity) {
    if (rt == nullptr) {
        throw std::invalid_argument("endpoint_runtime_init: rt is null");
    }
    if (group == nullptr) {
        throw std::invalid_argument("endpoint_runtime_init: group is null");
    }
    if (rank < 0 || rank >= group->world_size) {
        throw std::invalid_argument("endpoint_runtime_init: invalid rank");
    }
    if (num_input_queues <= 0) {
        throw std::invalid_argument("endpoint_runtime_init: num_input_queues must be > 0");
    }
    if (input_queue_capacity == 0) {
        throw std::invalid_argument("endpoint_runtime_init: input_queue_capacity must be > 0");
    }

    endpoint_runtime_destroy(rt);

    rt->endpoint = *group_get_endpoint(group, rank);

    // Build the local outgoing channel list for this endpoint.
    rt->outgoing_channels_host.reserve(static_cast<size_t>(group->world_size - 1));
    for (int dst_rank = 0; dst_rank < group->world_size; ++dst_rank) {
        if (dst_rank == rank) {
            continue;
        }

        const Channel* ch = group_get_channel(group, rank, dst_rank);
        if (ch != nullptr) {
            rt->outgoing_channels_host.push_back(*ch);
        }
    }

    // Allocate local ready queues for GEMM -> persistent-kernel handoff.
    rt->input_queues_host.resize(static_cast<size_t>(num_input_queues));
    for (int i = 0; i < num_input_queues; ++i) {
        collective::ready_tile_queue_init(
            &rt->input_queues_host[static_cast<size_t>(i)],
            rt->endpoint.device,
            input_queue_capacity);
    }

    // Default scheduler bindings are created but left unbound until the caller
    // explicitly maps each queue to a destination/op.
    rt->scheduler_bindings_host.resize(static_cast<size_t>(num_input_queues));
    for (int i = 0; i < num_input_queues; ++i) {
        exec::SchedulerQueueBinding binding{};
        binding.queue_id = static_cast<uint32_t>(i);
        binding.dst_rank = -1;
        binding.queue = rt->input_queues_host[static_cast<size_t>(i)];
        binding.channel_buffer_base = nullptr;
        binding.channel_buffer_bytes = 0;
        binding.op = exec::ChunkOpKind::kInvalid;
        rt->scheduler_bindings_host[static_cast<size_t>(i)] = binding;
    }

    system::runtime::set_device(rt->endpoint.device);

    system::runtime::check_cuda(
        cudaMalloc(
            &rt->input_queues_device,
            rt->input_queues_host.size() * sizeof(collective::ReadyTileQueue)),
        "cudaMalloc(endpoint runtime input_queues_device)");

    system::runtime::check_cuda(
        cudaMalloc(
            &rt->scheduler_bindings_device,
            rt->scheduler_bindings_host.size() * sizeof(exec::SchedulerQueueBinding)),
        "cudaMalloc(endpoint runtime scheduler_bindings_device)");

    if (!rt->outgoing_channels_host.empty()) {
        system::runtime::check_cuda(
            cudaMalloc(
                &rt->outgoing_channels_device,
                rt->outgoing_channels_host.size() * sizeof(Channel)),
            "cudaMalloc(endpoint runtime outgoing_channels_device)");
    }

    endpoint_runtime_refresh_device(rt);
    return true;
}

void endpoint_runtime_destroy(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        return;
    }

    const int device =
        (rt->endpoint.device >= 0) ? rt->endpoint.device :
        (rt->device.device >= 0 ? rt->device.device : -1);

    if (device >= 0) {
        system::runtime::set_device(device);
    }

    if (rt->outgoing_channels_device != nullptr) {
        system::runtime::check_cuda(
            cudaFree(rt->outgoing_channels_device),
            "cudaFree(endpoint runtime outgoing_channels_device)");
        rt->outgoing_channels_device = nullptr;
    }

    if (rt->scheduler_bindings_device != nullptr) {
        system::runtime::check_cuda(
            cudaFree(rt->scheduler_bindings_device),
            "cudaFree(endpoint runtime scheduler_bindings_device)");
        rt->scheduler_bindings_device = nullptr;
    }

    if (rt->input_queues_device != nullptr) {
        system::runtime::check_cuda(
            cudaFree(rt->input_queues_device),
            "cudaFree(endpoint runtime input_queues_device)");
        rt->input_queues_device = nullptr;
    }

    for (auto& q : rt->input_queues_host) {
        collective::ready_tile_queue_destroy(&q);
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
    if (rt->input_queues_host.empty()) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: no input queues");
    }
    if (rt->scheduler_bindings_host.empty()) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: no scheduler bindings");
    }
    if (rt->input_queues_device == nullptr) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: input_queues_device is null");
    }
    if (rt->scheduler_bindings_device == nullptr) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: scheduler_bindings_device is null");
    }

    system::runtime::set_device(rt->endpoint.device);

    system::runtime::check_cuda(
        cudaMemcpy(
            rt->input_queues_device,
            rt->input_queues_host.data(),
            rt->input_queues_host.size() * sizeof(collective::ReadyTileQueue),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(endpoint runtime input queues)");

    system::runtime::check_cuda(
        cudaMemcpy(
            rt->scheduler_bindings_device,
            rt->scheduler_bindings_host.data(),
            rt->scheduler_bindings_host.size() * sizeof(exec::SchedulerQueueBinding),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(endpoint runtime scheduler bindings)");

    if (!rt->outgoing_channels_host.empty()) {
        if (rt->outgoing_channels_device == nullptr) {
            throw std::invalid_argument("endpoint_runtime_refresh_device: outgoing_channels_device is null");
        }

        system::runtime::check_cuda(
            cudaMemcpy(
                rt->outgoing_channels_device,
                rt->outgoing_channels_host.data(),
                rt->outgoing_channels_host.size() * sizeof(Channel),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(endpoint runtime outgoing channels)");
    }

    rt->device.rank = rt->endpoint.rank;
    rt->device.device = rt->endpoint.device;
    rt->device.stream = rt->endpoint.stream;

    rt->device.input_queues = rt->input_queues_device;
    rt->device.num_input_queues = static_cast<int>(rt->input_queues_host.size());

    rt->device.scheduler_bindings = rt->scheduler_bindings_device;
    rt->device.num_scheduler_bindings = static_cast<int>(rt->scheduler_bindings_host.size());

    rt->device.outgoing_channels = rt->outgoing_channels_device;
    rt->device.num_outgoing_channels = static_cast<int>(rt->outgoing_channels_host.size());
}

int endpoint_runtime_find_outgoing_channel_index(
    const EndpointRuntime* rt,
    int dst_rank) {
    if (rt == nullptr) {
        throw std::invalid_argument("endpoint_runtime_find_outgoing_channel_index: rt is null");
    }

    for (int i = 0; i < static_cast<int>(rt->outgoing_channels_host.size()); ++i) {
        const Channel& ch = rt->outgoing_channels_host[static_cast<size_t>(i)];
        if (ch.dst_rank == dst_rank) {
            return i;
        }
    }

    return -1;
}

bool endpoint_runtime_bind_queue_to_dst(
    EndpointRuntime* rt,
    int queue_idx,
    int dst_rank,
    exec::ChunkOpKind op) {
    endpoint_runtime_validate_queue_index(
        rt,
        queue_idx,
        "endpoint_runtime_bind_queue_to_dst: invalid queue_idx");

    if (dst_rank < 0) {
        throw std::invalid_argument("endpoint_runtime_bind_queue_to_dst: invalid dst_rank");
    }
    if (op == exec::ChunkOpKind::kInvalid) {
        throw std::invalid_argument("endpoint_runtime_bind_queue_to_dst: invalid op");
    }

    const int channel_idx = endpoint_runtime_find_outgoing_channel_index(rt, dst_rank);
    if (channel_idx < 0) {
        return false;
    }

    const Channel& ch = rt->outgoing_channels_host[static_cast<size_t>(channel_idx)];
    const transport::CommBuffer* buf = channel_get_buffer(&ch);
    if (buf == nullptr || buf->ptr == nullptr || buf->bytes == 0) {
        return false;
    }

    exec::SchedulerQueueBinding& binding =
        rt->scheduler_bindings_host[static_cast<size_t>(queue_idx)];

    binding.queue_id = static_cast<uint32_t>(queue_idx);
    binding.dst_rank = dst_rank;
    binding.queue = rt->input_queues_host[static_cast<size_t>(queue_idx)];
    binding.channel_buffer_base = reinterpret_cast<unsigned char*>(buf->ptr);
    binding.channel_buffer_bytes = buf->bytes;
    binding.op = op;

    endpoint_runtime_refresh_device(rt);
    return true;
}

bool endpoint_runtime_unbind_queue(
    EndpointRuntime* rt,
    int queue_idx) {
    endpoint_runtime_validate_queue_index(
        rt,
        queue_idx,
        "endpoint_runtime_unbind_queue: invalid queue_idx");

    exec::SchedulerQueueBinding& binding =
        rt->scheduler_bindings_host[static_cast<size_t>(queue_idx)];

    binding.queue_id = static_cast<uint32_t>(queue_idx);
    binding.dst_rank = -1;
    binding.queue = rt->input_queues_host[static_cast<size_t>(queue_idx)];
    binding.channel_buffer_base = nullptr;
    binding.channel_buffer_bytes = 0;
    binding.op = exec::ChunkOpKind::kInvalid;

    endpoint_runtime_refresh_device(rt);
    return true;
}

} // namespace comm
} // namespace ooverlap
