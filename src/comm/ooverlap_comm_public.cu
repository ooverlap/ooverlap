#include "comm/ooverlap_comm_private.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/utils/collective_utils.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <new>
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

    slot = oo_ready_signal{};
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

size_t oo_dtype_size(oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
        case OO_DTYPE_BFLOAT16:
            return 2;

        case OO_DTYPE_FLOAT32:
            return 4;

        default:
            return 0;
    }
}

const char* oo_status_string(oo_status_t status) {
    switch (status) {
        case OO_SUCCESS:
            return "OO_SUCCESS";
        case OO_ERROR_INVALID_ARGUMENT:
            return "OO_ERROR_INVALID_ARGUMENT";
        case OO_ERROR_INVALID_DEVICE:
            return "OO_ERROR_INVALID_DEVICE";
        case OO_ERROR_UNSUPPORTED:
            return "OO_ERROR_UNSUPPORTED";
        case OO_ERROR_CUDA:
            return "OO_ERROR_CUDA";
        case OO_ERROR_INTERNAL:
            return "OO_ERROR_INTERNAL";
        default:
            return "OO_ERROR_UNKNOWN";
    }
}

oo_status_t oo_rank_partition(
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

int oo_allreduce_supported(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    return ooverlap::comm::utils::reduce_op_supported_for_dtype(dtype, op) ? 1 : 0;
}

int oo_reduce_scatter_supported(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    return ooverlap::comm::utils::reduce_op_supported_for_dtype(dtype, op) ? 1 : 0;
}

int oo_all_gather_supported(oo_dtype_t dtype) {
    return ooverlap::comm::utils::dtype_supported_for_copy_collective(dtype) ? 1 : 0;
}

oo_status_t oo_group_create(
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
        std::unique_ptr<oo_group_t> group(new oo_group_t{});

        group->num_devices = num_devices;
        group->bootstrap_kind = oo_group_bootstrap_kind::same_process;

        for (int i = 0; i < num_devices; ++i) {
            if (devices[i] < 0) {
                return OO_ERROR_INVALID_DEVICE;
            }

            group->devices[i] = devices[i];
        }

        const oo_status_t status =
            allocate_same_process_ready_signals(group.get());

        if (status != OO_SUCCESS) {
            return status;
        }

        *out_group = group.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

oo_status_t oo_group_create_ipc(
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
        std::unique_ptr<oo_group_t> group(new oo_group_t{});

        group->num_devices = num_devices;
        group->bootstrap_kind = oo_group_bootstrap_kind::multiprocess_ipc;
        group->local_rank = local_rank;
        group->local_world_size = num_devices;
        group->broker.reset(
            new ooverlap::system::Broker(
                local_rank,
                num_devices,
                broker_key));

        for (int i = 0; i < num_devices; ++i) {
            if (devices[i] < 0) {
                return OO_ERROR_INVALID_DEVICE;
            }

            group->devices[i] = devices[i];
        }

        const oo_status_t status =
            allocate_ipc_ready_signals(group.get());

        if (status != OO_SUCCESS) {
            return status;
        }

        *out_group = group.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

void oo_group_destroy(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    destroy_group_ready_signals(group);
    group->broker.reset();
    delete group;
}

int oo_group_size(const oo_group_t* group) {
    return group != nullptr ? group->num_devices : 0;
}

int oo_group_device(
    const oo_group_t* group,
    int rank) {
    if (group == nullptr || rank < 0 || rank >= group->num_devices) {
        return -1;
    }

    return group->devices[rank];
}

oo_status_t oo_node_create(
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

    try {
        std::unique_ptr<oo_node_t> node(new oo_node_t{});
        node->group = group;
        node->rank = rank;
        node->device = group->devices[rank];
        node->collective_epoch = 0;

        *out_node = node.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

void oo_node_destroy(oo_node_t* node) {
    delete node;
}

oo_group_t* oo_node_group(const oo_node_t* node) {
    return node != nullptr ? node->group : nullptr;
}

int oo_node_rank(const oo_node_t* node) {
    return node != nullptr ? node->rank : -1;
}

int oo_node_device(const oo_node_t* node) {
    return node != nullptr ? node->device : -1;
}

oo_status_t oo_buffer_alloc(
    oo_node_t* node,
    size_t bytes,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_buffer = nullptr;

    if (node == nullptr ||
        node->group == nullptr ||
        node->rank < 0 ||
        node->rank >= node->group->num_devices ||
        bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        const std::vector<int> access_devices =
            devices_vector(
                node->group->devices,
                node->group->num_devices);

        auto mapped =
            ooverlap::system::alloc_peer_visible_buffer(
                bytes,
                node->device,
                access_devices);

        std::unique_ptr<oo_buffer_t> buffer(new oo_buffer_t{});
        buffer->ptr = mapped.ptr;
        buffer->bytes = bytes;
        buffer->mapped_bytes = mapped.mapped_size;
        buffer->kind = OO_BUFFER_KIND_VMM;
        buffer->group = node->group;
        buffer->owner_rank = node->rank;
        buffer->owner_device = node->device;
        buffer->system_kind = ooverlap::system::peer_buffer_kind::owned_vmm;
        buffer->mapped = mapped;

        *out_buffer = buffer.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

oo_status_t oo_buffer_wrap(
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

    try {
        std::unique_ptr<oo_buffer_t> buffer(new oo_buffer_t{});
        buffer->ptr = ptr;
        buffer->bytes = bytes;
        buffer->mapped_bytes = bytes;
        buffer->kind = OO_BUFFER_KIND_WRAPPED;
        buffer->group = node->group;
        buffer->owner_rank = node->rank;
        buffer->owner_device = node->device;
        buffer->system_kind = ooverlap::system::peer_buffer_kind::wrapped;

        *out_buffer = buffer.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

void oo_buffer_destroy(oo_buffer_t* buffer) {
    if (buffer == nullptr) {
        return;
    }

    destroy_buffer_storage(buffer);
    delete buffer;
}

void* oo_buffer_ptr(const oo_buffer_t* buffer) {
    return buffer != nullptr ? buffer->ptr : nullptr;
}

size_t oo_buffer_bytes(const oo_buffer_t* buffer) {
    return buffer != nullptr ? buffer->bytes : 0;
}

size_t oo_buffer_mapped_bytes(const oo_buffer_t* buffer) {
    return buffer != nullptr ? buffer->mapped_bytes : 0;
}

oo_buffer_kind_t oo_buffer_kind(const oo_buffer_t* buffer) {
    return buffer != nullptr ? buffer->kind : OO_BUFFER_KIND_WRAPPED;
}

oo_status_t oo_group_sync(oo_group_t* group) {
    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        if (group->broker != nullptr) {
            group->broker->sync();
            return OO_SUCCESS;
        }

        for (int i = 0; i < group->num_devices; ++i) {
            ooverlap::system::runtime::set_device(group->devices[i]);
            ooverlap::system::runtime::check_cuda(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize");
        }

        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

oo_status_t oo_buffer_exchange_ipc_peers(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t** out_peers,
    int* out_peer_count) {
    if (out_peer_count != nullptr) {
        *out_peer_count = 0;
    }

    if (node == nullptr ||
        node->group == nullptr ||
        node->group->broker == nullptr ||
        local == nullptr ||
        out_peers == nullptr ||
        out_peer_count == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_group_t* group = node->group;

    if (node->rank < 0 ||
        node->rank >= group->num_devices ||
        group->local_rank != node->rank ||
        group->local_world_size != group->num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
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

        int peer_count = 0;

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
                for (int i = 0; i < peer_count; ++i) {
                    oo_buffer_destroy(out_peers[i]);
                    out_peers[i] = nullptr;
                }

                return status;
            }

            out_peers[peer_count++] = peer;
        }

        group->broker->sync();

        *out_peer_count = peer_count;
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}
