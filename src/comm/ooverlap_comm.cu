#include "comm/ooverlap_comm_private.h"

#include "comm/tuning/tuning_policy.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool valid_group_size(int num_devices) {
    return num_devices > 0 && num_devices <= kOoMaxLocalDevices;
}

std::vector<int> devices_vector(const int* devices, int num_devices) {
    std::vector<int> out;
    out.reserve(static_cast<size_t>(num_devices));

    for (int i = 0; i < num_devices; ++i) {
        out.push_back(devices[i]);
    }

    return out;
}

void clear_ready_signal(oo_ready_signal& slot) {
    if (slot.kind == oo_ready_signal_kind::owned_vmm) {
        ooverlap::system::free_peer_visible_buffer(slot.owned_vmm);
    } else if (slot.kind == oo_ready_signal_kind::owned_legacy) {
        if (slot.owned_legacy_ptr != nullptr) {
            if (slot.owner_device >= 0) {
                ooverlap::system::runtime::set_device(slot.owner_device);
            }
            cudaFree(slot.owned_legacy_ptr);
        }
    } else if (slot.kind == oo_ready_signal_kind::imported_legacy ||
               slot.kind == oo_ready_signal_kind::imported_vmm) {
        slot.imported.reset();
    }

    slot.ptr = nullptr;
    slot.bytes = 0;
    slot.mapped_bytes = 0;
    slot.owner_rank = -1;
    slot.owner_device = -1;
    slot.kind = oo_ready_signal_kind::empty;
    slot.owned_legacy_ptr = nullptr;
}

void destroy_group_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    for (int i = 0; i < kOoMaxLocalDevices; ++i) {
        clear_ready_signal(group->ready_signal_slots[i]);
        group->ready_signals[i] = {};
    }
}

oo_status_t allocate_same_process_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const std::vector<int> access_devices =
        devices_vector(group->devices, group->num_devices);

    for (int rank = 0; rank < group->num_devices; ++rank) {
        const int device = group->devices[rank];

        auto mapped =
            ooverlap::system::alloc_peer_visible_buffer(
                sizeof(int),
                device,
                access_devices);

        ooverlap::system::runtime::set_device(device);
        ooverlap::system::runtime::check_cuda(
            cudaMemset(mapped.ptr, 0, sizeof(int)),
            "cudaMemset(ready signal)");

        oo_ready_signal& slot = group->ready_signal_slots[rank];
        slot.ptr = mapped.ptr;
        slot.bytes = sizeof(int);
        slot.mapped_bytes = mapped.mapped_size;
        slot.owner_rank = rank;
        slot.owner_device = device;
        slot.kind = oo_ready_signal_kind::owned_vmm;
        slot.owned_vmm = mapped;

        group->ready_signals[rank] = mapped;
    }

    return OO_SUCCESS;
}

oo_status_t allocate_ipc_ready_signals(oo_group_t* group) {
    if (group == nullptr || group->broker == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int rank = group->local_rank;
    const int world_size = group->local_world_size;

    if (rank < 0 || rank >= world_size || world_size != group->num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int local_device = group->devices[rank];

    void* local_signal = nullptr;

    ooverlap::system::runtime::set_device(local_device);
    ooverlap::system::runtime::check_cuda(
        cudaMalloc(&local_signal, sizeof(int)),
        "cudaMalloc(ipc ready signal)");
    ooverlap::system::runtime::check_cuda(
        cudaMemset(local_signal, 0, sizeof(int)),
        "cudaMemset(ipc ready signal)");

    ooverlap::system::legacy_peer_buffer_descriptor local_desc =
        ooverlap::system::export_legacy_peer_buffer(
            local_signal,
            sizeof(int),
            local_device,
            sizeof(int));

    std::vector<ooverlap::system::legacy_peer_buffer_descriptor> descs(
        static_cast<size_t>(world_size));

    group->broker->exchange_data(
        descs.data(),
        &local_desc,
        sizeof(local_desc));

    group->broker->sync();

    for (int r = 0; r < world_size; ++r) {
        oo_ready_signal& slot = group->ready_signal_slots[r];

        if (r == rank) {
            slot.ptr = local_signal;
            slot.bytes = sizeof(int);
            slot.mapped_bytes = sizeof(int);
            slot.owner_rank = r;
            slot.owner_device = local_device;
            slot.kind = oo_ready_signal_kind::owned_legacy;
            slot.owned_legacy_ptr = local_signal;
            continue;
        }

        auto imported =
            ooverlap::system::import_legacy_peer_buffer(
                descs[static_cast<size_t>(r)],
                std::vector<int>{local_device});

        slot.ptr = imported.ptr;
        slot.bytes = imported.bytes;
        slot.mapped_bytes = imported.mapped_size;
        slot.owner_rank = r;
        slot.owner_device = imported.owner_device;
        slot.kind = oo_ready_signal_kind::imported_legacy;
        slot.imported = std::move(imported);
    }

    group->broker->sync();

    return OO_SUCCESS;
}

void destroy_buffer_storage(oo_buffer_t* buffer) {
    if (buffer == nullptr) {
        return;
    }

    if (buffer->system_kind == ooverlap::system::peer_buffer_kind::owned_vmm) {
        ooverlap::system::free_peer_visible_buffer(buffer->mapped);
    } else if (
        buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_legacy ||
        buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_vmm) {
        buffer->imported.reset();
    }

    buffer->ptr = nullptr;
    buffer->bytes = 0;
    buffer->mapped_bytes = 0;
    buffer->group = nullptr;
    buffer->owner_rank = -1;
    buffer->owner_device = -1;
    buffer->system_kind = ooverlap::system::peer_buffer_kind::empty;
}

} // namespace

namespace ooverlap {
namespace comm {
namespace api {

oo_status_t exception_to_status() {
    try {
        throw;
    } catch (const std::invalid_argument&) {
        return OO_ERROR_INVALID_ARGUMENT;
    } catch (const std::out_of_range&) {
        return OO_ERROR_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        return OO_ERROR_INTERNAL;
    } catch (const std::exception&) {
        return OO_ERROR_INTERNAL;
    } catch (...) {
        return OO_ERROR_INTERNAL;
    }
}

oo_status_t cuda_to_status(cudaError_t error) {
    if (error == cudaSuccess) {
        return OO_SUCCESS;
    }

    if (error == cudaErrorInvalidDevice) {
        return OO_ERROR_INVALID_DEVICE;
    }

    if (error == cudaErrorInvalidValue) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    return OO_ERROR_CUDA;
}

oo_status_t checked_element_bytes(
    size_t count,
    oo_dtype_t dtype,
    size_t* out_bytes) {
    if (out_bytes == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const size_t dtype_size = oo_dtype_size(dtype);

    if (dtype_size == 0) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (count > static_cast<size_t>(-1) / dtype_size) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_bytes = count * dtype_size;
    return OO_SUCCESS;
}

oo_status_t checked_element_offset_bytes(
    size_t element_offset,
    oo_dtype_t dtype,
    size_t* out_offset_bytes) {
    if (out_offset_bytes == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const size_t dtype_size = oo_dtype_size(dtype);

    if (dtype_size == 0) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (element_offset > static_cast<size_t>(-1) / dtype_size) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_offset_bytes = element_offset * dtype_size;
    return OO_SUCCESS;
}

oo_status_t fill_rank_partition(
    int rank,
    int world_size,
    size_t count,
    size_t* out_element_offset,
    size_t* out_count) {
    if (out_element_offset == nullptr || out_count == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_element_offset = 0;
    *out_count = 0;

    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);

    const size_t base = count / world;
    const size_t rem = count % world;

    *out_element_offset = r * base + ((r < rem) ? r : rem);
    *out_count = base + ((r < rem) ? 1 : 0);

    return OO_SUCCESS;
}

oo_status_t fill_tensor_slice(
    oo_buffer_t* local,
    int rank,
    int world_size,
    size_t base_element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_tensor_slice_t* out_slice) {
    if (out_slice == nullptr) {
        return OO_SUCCESS;
    }

    if (local == nullptr || local->ptr == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    size_t local_offset = 0;
    size_t local_count = 0;

    oo_status_t status =
        fill_rank_partition(
            rank,
            world_size,
            count,
            &local_offset,
            &local_count);

    if (status != OO_SUCCESS) {
        return status;
    }

    size_t absolute_offset = base_element_offset + local_offset;

    if (absolute_offset < base_element_offset) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    size_t absolute_offset_bytes = 0;

    status =
        checked_element_offset_bytes(
            absolute_offset,
            dtype,
            &absolute_offset_bytes);

    if (status != OO_SUCCESS) {
        return status;
    }

    const size_t dtype_size = oo_dtype_size(dtype);

    if (dtype_size == 0) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (local_count > static_cast<size_t>(-1) / dtype_size) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const size_t local_bytes = local_count * dtype_size;

    if (absolute_offset_bytes > local->bytes ||
        local_bytes > local->bytes - absolute_offset_bytes) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    out_slice->element_offset = absolute_offset;
    out_slice->count = local_count;
    out_slice->ptr =
        reinterpret_cast<void*>(
            reinterpret_cast<std::uint8_t*>(local->ptr) +
            absolute_offset_bytes);

    return OO_SUCCESS;
}

LaunchConfig select_public_launch_config(
    size_t bytes,
    oo_tuning_mode_t tuning_mode) {
    return select_launch_config_for_allreduce(
        bytes,
        tuning_preference_from_public(tuning_mode));
}

oo_status_t prepare_collective_launch(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    CollectiveLaunchState* out) {
    if (out == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out = CollectiveLaunchState{};

    if (node == nullptr ||
        node->group == nullptr ||
        local == nullptr ||
        local->ptr == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_group_t* group = node->group;

    if (!valid_group_size(group->num_devices)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (node->rank < 0 || node->rank >= group->num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (node->device != group->devices[node->rank]) {
        return OO_ERROR_INVALID_DEVICE;
    }

    if (peer_count != group->num_devices - 1 ||
        peer_count < 0 ||
        peer_count > kMaxPublicPeers) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (peer_count > 0 && peers == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (local->group != group ||
        local->owner_rank != node->rank ||
        local->owner_device != node->device) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    size_t bytes = 0;
    oo_status_t status = checked_element_bytes(count, dtype, &bytes);

    if (status != OO_SUCCESS) {
        return status;
    }

    size_t offset_bytes = 0;
    status = checked_element_offset_bytes(element_offset, dtype, &offset_bytes);

    if (status != OO_SUCCESS) {
        return status;
    }

    if (offset_bytes > local->bytes || bytes > local->bytes - offset_bytes) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    out->local_ptr =
        reinterpret_cast<void*>(
            reinterpret_cast<std::uint8_t*>(local->ptr) + offset_bytes);

    bool seen_rank[kOoMaxLocalDevices] = {};

    for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
        oo_buffer_t* peer = peers[peer_idx];

        if (peer == nullptr || peer->ptr == nullptr || peer->group != group) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        if (peer->owner_rank == node->rank ||
            peer->owner_rank < 0 ||
            peer->owner_rank >= group->num_devices) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        if (seen_rank[peer->owner_rank]) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        seen_rank[peer->owner_rank] = true;

        if (offset_bytes > peer->bytes || bytes > peer->bytes - offset_bytes) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        out->peer_ptrs[peer_idx] =
            reinterpret_cast<void*>(
                reinterpret_cast<std::uint8_t*>(peer->ptr) + offset_bytes);

        const oo_ready_signal& ready =
            group->ready_signal_slots[peer->owner_rank];

        out->peer_ready_signals[peer_idx] =
            reinterpret_cast<const int*>(ready.ptr);
    }

    out->peer_count = peer_count;
    out->local_ready_signal =
        reinterpret_cast<int*>(
            group->ready_signal_slots[node->rank].ptr);
    out->rank = node->rank;
    out->world_size = group->num_devices;
    out->local_device = node->device;
    out->collective_epoch = ++node->collective_epoch;
    out->dtype_size = oo_dtype_size(dtype);
    out->bytes = bytes;

    return OO_SUCCESS;
}

} // namespace api
} // namespace comm
} // namespace ooverlap

extern "C" size_t oo_dtype_size(
    oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
            return 2;
        case OO_DTYPE_BFLOAT16:
            return 2;
        case OO_DTYPE_FLOAT32:
            return 4;
        default:
            return 0;
    }
}

extern "C" oo_status_t oo_rank_partition(
    int rank,
    int world_size,
    size_t count,
    size_t* out_element_offset,
    size_t* out_count) {
    return ooverlap::comm::api::fill_rank_partition(
        rank,
        world_size,
        count,
        out_element_offset,
        out_count);
}

extern "C" int oo_allreduce_supported(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16 ||
               dtype == OO_DTYPE_FLOAT32;
    }

    if (op == OO_REDUCE_MIN || op == OO_REDUCE_MAX) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16;
    }

    return 0;
}

extern "C" int oo_reduce_scatter_supported(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    return oo_allreduce_supported(dtype, op);
}

extern "C" int oo_all_gather_supported(
    oo_dtype_t dtype) {
    return dtype == OO_DTYPE_FLOAT16 ||
           dtype == OO_DTYPE_BFLOAT16 ||
           dtype == OO_DTYPE_FLOAT32;
}

extern "C" oo_status_t oo_group_create(
    const int* devices,
    int num_devices,
    oo_group_t** out_group) {
    if (out_group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_group = nullptr;

    if (devices == nullptr || !valid_group_size(num_devices)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto group = std::make_unique<oo_group_t>();

        group->num_devices = num_devices;
        group->bootstrap_kind = oo_group_bootstrap_kind::same_process;
        group->local_rank = -1;
        group->local_world_size = num_devices;

        for (int i = 0; i < num_devices; ++i) {
            if (devices[i] < 0) {
                return OO_ERROR_INVALID_DEVICE;
            }

            for (int j = 0; j < i; ++j) {
                if (devices[i] == devices[j]) {
                    return OO_ERROR_INVALID_ARGUMENT;
                }
            }

            group->devices[i] = devices[i];
        }

        oo_status_t status = allocate_same_process_ready_signals(group.get());

        if (status != OO_SUCCESS) {
            destroy_group_ready_signals(group.get());
            return status;
        }

        *out_group = group.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

extern "C" oo_status_t oo_group_create_ipc(
    const int* devices,
    int num_devices,
    int local_rank,
    const char* broker_key,
    oo_group_t** out_group) {
    if (out_group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_group = nullptr;

    if (devices == nullptr ||
        broker_key == nullptr ||
        !valid_group_size(num_devices) ||
        local_rank < 0 ||
        local_rank >= num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto group = std::make_unique<oo_group_t>();

        group->num_devices = num_devices;
        group->bootstrap_kind = oo_group_bootstrap_kind::multiprocess_ipc;
        group->local_rank = local_rank;
        group->local_world_size = num_devices;

        for (int i = 0; i < num_devices; ++i) {
            if (devices[i] < 0) {
                return OO_ERROR_INVALID_DEVICE;
            }

            group->devices[i] = devices[i];
        }

        group->broker =
            std::make_unique<ooverlap::system::Broker>(
                local_rank,
                num_devices,
                broker_key);

        oo_status_t status = allocate_ipc_ready_signals(group.get());

        if (status != OO_SUCCESS) {
            destroy_group_ready_signals(group.get());
            return status;
        }

        *out_group = group.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

extern "C" void oo_group_destroy(
    oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    destroy_group_ready_signals(group);

    if (group->broker) {
        group->broker->destroy();
        group->broker.reset();
    }

    delete group;
}

extern "C" int oo_group_size(
    const oo_group_t* group) {
    return group != nullptr ? group->num_devices : 0;
}

extern "C" int oo_group_device(
    const oo_group_t* group,
    int rank) {
    if (group == nullptr || rank < 0 || rank >= group->num_devices) {
        return -1;
    }

    return group->devices[rank];
}

extern "C" oo_status_t oo_node_create(
    oo_group_t* group,
    int rank,
    oo_node_t** out_node) {
    if (out_node == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_node = nullptr;

    if (group == nullptr || rank < 0 || rank >= group->num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc &&
        rank != group->local_rank) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    auto* node = new (std::nothrow) oo_node_t();

    if (node == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    node->group = group;
    node->rank = rank;
    node->device = group->devices[rank];
    node->collective_epoch = 0;

    *out_node = node;
    return OO_SUCCESS;
}

extern "C" void oo_node_destroy(
    oo_node_t* node) {
    delete node;
}

extern "C" oo_group_t* oo_node_group(
    const oo_node_t* node) {
    return node != nullptr ? node->group : nullptr;
}

extern "C" int oo_node_rank(
    const oo_node_t* node) {
    return node != nullptr ? node->rank : -1;
}

extern "C" int oo_node_device(
    const oo_node_t* node) {
    return node != nullptr ? node->device : -1;
}

extern "C" oo_status_t oo_buffer_alloc(
    oo_node_t* node,
    size_t bytes,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr || bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        oo_group_t* group = node->group;

        auto mapped =
            ooverlap::system::alloc_peer_visible_buffer(
                bytes,
                node->device,
                devices_vector(group->devices, group->num_devices));

        auto* buffer = new oo_buffer_t();

        buffer->ptr = mapped.ptr;
        buffer->bytes = bytes;
        buffer->mapped_bytes = mapped.mapped_size;
        buffer->kind = OO_BUFFER_KIND_VMM;
        buffer->group = group;
        buffer->owner_rank = node->rank;
        buffer->owner_device = node->device;
        buffer->system_kind = ooverlap::system::peer_buffer_kind::owned_vmm;
        buffer->mapped = mapped;

        *out_buffer = buffer;
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

extern "C" oo_status_t oo_buffer_wrap(
    oo_node_t* node,
    void* ptr,
    size_t bytes,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_buffer = nullptr;

    if (node == nullptr ||
        node->group == nullptr ||
        ptr == nullptr ||
        bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    auto* buffer = new (std::nothrow) oo_buffer_t();

    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    buffer->ptr = ptr;
    buffer->bytes = bytes;
    buffer->mapped_bytes = bytes;
    buffer->kind = OO_BUFFER_KIND_WRAPPED;
    buffer->group = node->group;
    buffer->owner_rank = node->rank;
    buffer->owner_device = node->device;
    buffer->system_kind = ooverlap::system::peer_buffer_kind::wrapped;

    *out_buffer = buffer;
    return OO_SUCCESS;
}

extern "C" void oo_buffer_destroy(
    oo_buffer_t* buffer) {
    if (buffer == nullptr) {
        return;
    }

    destroy_buffer_storage(buffer);
    delete buffer;
}

extern "C" void* oo_buffer_ptr(
    const oo_buffer_t* buffer) {
    return buffer != nullptr ? buffer->ptr : nullptr;
}

extern "C" size_t oo_buffer_bytes(
    const oo_buffer_t* buffer) {
    return buffer != nullptr ? buffer->bytes : 0;
}

extern "C" size_t oo_buffer_mapped_bytes(
    const oo_buffer_t* buffer) {
    return buffer != nullptr ? buffer->mapped_bytes : 0;
}

extern "C" oo_buffer_kind_t oo_buffer_kind(
    const oo_buffer_t* buffer) {
    return buffer != nullptr ? buffer->kind : OO_BUFFER_KIND_WRAPPED;
}

extern "C" oo_status_t oo_group_sync(
    oo_group_t* group) {
    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        if (group->broker) {
            group->broker->sync();
            return OO_SUCCESS;
        }

        for (int i = 0; i < group->num_devices; ++i) {
            ooverlap::system::runtime::set_device(group->devices[i]);
            ooverlap::system::runtime::check_cuda(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize(group sync)");
        }

        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

oo_status_t oo_buffer_export_legacy_descriptor(
    oo_buffer_t* buffer,
    ooverlap::system::legacy_peer_buffer_descriptor* out_desc) {
    if (buffer == nullptr || buffer->ptr == nullptr || out_desc == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        *out_desc =
            ooverlap::system::export_legacy_peer_buffer(
                buffer->ptr,
                buffer->bytes,
                buffer->owner_device,
                buffer->mapped_bytes != 0 ? buffer->mapped_bytes : buffer->bytes);

        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

oo_status_t oo_buffer_adopt_imported_peer_buffer(
    oo_node_t* node,
    ooverlap::system::imported_peer_buffer&& imported,
    oo_buffer_t** out_buffer) {
    if (node == nullptr ||
        node->group == nullptr ||
        out_buffer == nullptr ||
        !imported.valid()) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_buffer = nullptr;

    auto* buffer = new (std::nothrow) oo_buffer_t();

    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    buffer->ptr = imported.ptr;
    buffer->bytes = imported.bytes;
    buffer->mapped_bytes = imported.mapped_size;
    buffer->kind = OO_BUFFER_KIND_WRAPPED;
    buffer->group = node->group;
    buffer->owner_rank = -1;
    buffer->owner_device = imported.owner_device;
    buffer->system_kind = imported.kind;
    buffer->imported = std::move(imported);

    *out_buffer = buffer;
    return OO_SUCCESS;
}

oo_status_t oo_buffer_import_legacy_descriptor(
    oo_node_t* node,
    const ooverlap::system::legacy_peer_buffer_descriptor& desc,
    oo_buffer_t** out_buffer) {
    if (node == nullptr || node->group == nullptr || out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto imported =
            ooverlap::system::import_legacy_peer_buffer(
                desc,
                std::vector<int>{node->device});

        return oo_buffer_adopt_imported_peer_buffer(
            node,
            std::move(imported),
            out_buffer);
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

extern "C" oo_status_t oo_buffer_exchange_ipc_peers(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t** out_peers,
    int* out_peer_count) {
    if (out_peer_count != nullptr) {
        *out_peer_count = 0;
    }

    if (node == nullptr ||
        node->group == nullptr ||
        local == nullptr ||
        local->ptr == nullptr ||
        out_peers == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_group_t* group = node->group;

    if (group->bootstrap_kind != oo_group_bootstrap_kind::multiprocess_ipc ||
        group->broker == nullptr ||
        node->rank != group->local_rank) {
        return OO_ERROR_UNSUPPORTED;
    }

    try {
        ooverlap::system::legacy_peer_buffer_descriptor local_desc{};

        oo_status_t status =
            oo_buffer_export_legacy_descriptor(
                local,
                &local_desc);

        if (status != OO_SUCCESS) {
            return status;
        }

        std::vector<ooverlap::system::legacy_peer_buffer_descriptor> descs(
            static_cast<size_t>(group->num_devices));

        group->broker->exchange_data(
            descs.data(),
            &local_desc,
            sizeof(local_desc));

        group->broker->sync();

        int written = 0;

        for (int rank = 0; rank < group->num_devices; ++rank) {
            if (rank == node->rank) {
                continue;
            }

            oo_buffer_t* peer = nullptr;

            status =
                oo_buffer_import_legacy_descriptor(
                    node,
                    descs[static_cast<size_t>(rank)],
                    &peer);

            if (status != OO_SUCCESS) {
                for (int i = 0; i < written; ++i) {
                    oo_buffer_destroy(out_peers[i]);
                    out_peers[i] = nullptr;
                }
                return status;
            }

            peer->owner_rank = rank;
            peer->group = group;
            out_peers[written++] = peer;
        }

        if (out_peer_count != nullptr) {
            *out_peer_count = written;
        }

        group->broker->sync();

        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}
