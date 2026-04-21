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

    rt->device.operations = nullptr;
    rt->device.num_operations = 0;

    rt->device.scheduler_bindings = nullptr;
    rt->device.num_scheduler_bindings = 0;
}

void endpoint_runtime_reset_host_state(
    EndpointRuntime* rt) {
    if (rt == nullptr) {
        return;
    }

    rt->endpoint = Endpoint{};
    rt->input_queues_host.clear();
    rt->operations_host.clear();
    rt->scheduler_bindings_host.clear();

    rt->input_queues_device = nullptr;
    rt->scheduler_bindings_device = nullptr;
    rt->operation_table = collective::OperationTable{};

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

    // One queue and one operation slot per input queue for now.
    rt->input_queues_host.resize(static_cast<size_t>(num_input_queues));
    rt->operations_host.resize(static_cast<size_t>(num_input_queues));
    rt->scheduler_bindings_host.resize(static_cast<size_t>(num_input_queues));

    for (int i = 0; i < num_input_queues; ++i) {
        collective::ready_tile_queue_init(
            &rt->input_queues_host[static_cast<size_t>(i)],
            rt->endpoint.device,
            input_queue_capacity);

        collective::operation_desc_clear(
            &rt->operations_host[static_cast<size_t>(i)]);

        exec::SchedulerQueueBinding binding{};
        binding.queue = nullptr;
        binding.operation = nullptr;
        rt->scheduler_bindings_host[static_cast<size_t>(i)] = binding;
    }

    collective::operation_table_init(
        &rt->operation_table,
        rt->endpoint.device,
        static_cast<uint32_t>(num_input_queues));

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

    collective::operation_table_destroy(&rt->operation_table);

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
    if (rt->operations_host.empty()) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: no operations");
    }
    if (rt->scheduler_bindings_host.empty()) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: no scheduler bindings");
    }
    if (rt->input_queues_device == nullptr) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: input_queues_device is null");
    }
    if (!collective::operation_table_is_configured(&rt->operation_table)) {
        throw std::invalid_argument("endpoint_runtime_refresh_device: operation_table not configured");
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
            rt->operation_table.records,
            rt->operations_host.data(),
            rt->operations_host.size() * sizeof(collective::OperationDesc),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(endpoint runtime operation table)");

    // Rebuild binding pointers so they point at device-resident queue/op records.
    for (int i = 0; i < static_cast<int>(rt->scheduler_bindings_host.size()); ++i) {
        exec::SchedulerQueueBinding binding{};
        binding.queue = &rt->input_queues_device[static_cast<size_t>(i)];

        if (collective::operation_desc_is_active(
                &rt->operations_host[static_cast<size_t>(i)])) {
            binding.operation =
                &rt->operation_table.records[static_cast<size_t>(i)];
        } else {
            binding.operation = nullptr;
        }

        rt->scheduler_bindings_host[static_cast<size_t>(i)] = binding;
    }

    system::runtime::check_cuda(
        cudaMemcpy(
            rt->scheduler_bindings_device,
            rt->scheduler_bindings_host.data(),
            rt->scheduler_bindings_host.size() * sizeof(exec::SchedulerQueueBinding),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(endpoint runtime scheduler bindings)");

    rt->device.rank = rt->endpoint.rank;
    rt->device.device = rt->endpoint.device;
    rt->device.stream = rt->endpoint.stream;

    rt->device.input_queues = rt->input_queues_device;
    rt->device.num_input_queues = static_cast<int>(rt->input_queues_host.size());

    rt->device.operations = rt->operation_table.records;
    rt->device.num_operations = static_cast<int>(rt->operations_host.size());

    rt->device.scheduler_bindings = rt->scheduler_bindings_device;
    rt->device.num_scheduler_bindings = static_cast<int>(rt->scheduler_bindings_host.size());
}

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
    uint32_t epoch,
    bool enabled) {
    endpoint_runtime_validate_queue_index(
        rt,
        queue_idx,
        "endpoint_runtime_configure_queue_operation: invalid queue_idx");

    if (op_id == 0) {
        throw std::invalid_argument("endpoint_runtime_configure_queue_operation: op_id must be > 0");
    }
    if (dst_rank < 0) {
        throw std::invalid_argument("endpoint_runtime_configure_queue_operation: invalid dst_rank");
    }
    if (dst_ptr == nullptr) {
        throw std::invalid_argument("endpoint_runtime_configure_queue_operation: dst_ptr is null");
    }
    if (dst_bytes == 0) {
        throw std::invalid_argument("endpoint_runtime_configure_queue_operation: dst_bytes must be > 0");
    }
    if (tile_stride_bytes == 0) {
        throw std::invalid_argument("endpoint_runtime_configure_queue_operation: tile_stride_bytes must be > 0");
    }
    if (expected_contributions == 0) {
        throw std::invalid_argument("endpoint_runtime_configure_queue_operation: expected_contributions must be > 0");
    }
    if (op == exec::ChunkOpKind::kInvalid) {
        throw std::invalid_argument("endpoint_runtime_configure_queue_operation: invalid op");
    }

    collective::OperationDesc desc{};
    desc.op_id = op_id;
    desc.queue_id = static_cast<uint32_t>(queue_idx);
    desc.epoch = epoch;
    desc.flags = enabled ? collective::kOperationFlagEnabled : 0u;
    desc.src_rank = rt->endpoint.rank;
    desc.dst_rank = dst_rank;
    desc.dst_ptr = reinterpret_cast<uint64_t>(dst_ptr);
    desc.dst_bytes = dst_bytes;
    desc.tile_stride_bytes = tile_stride_bytes;
    desc.expected_contributions = expected_contributions;
    desc.op = op;

    if (!collective::operation_desc_is_valid(&desc)) {
        throw std::invalid_argument("endpoint_runtime_configure_queue_operation: produced invalid descriptor");
    }

    rt->operations_host[static_cast<size_t>(queue_idx)] = desc;
    endpoint_runtime_refresh_device(rt);
    return true;
}

bool endpoint_runtime_clear_queue_operation(
    EndpointRuntime* rt,
    int queue_idx) {
    endpoint_runtime_validate_queue_index(
        rt,
        queue_idx,
        "endpoint_runtime_clear_queue_operation: invalid queue_idx");

    collective::operation_desc_clear(
        &rt->operations_host[static_cast<size_t>(queue_idx)]);

    endpoint_runtime_refresh_device(rt);
    return true;
}

} // namespace comm
} // namespace ooverlap
