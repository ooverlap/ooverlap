#include "comm/ooverlap_comm_private.h"

#include "comm/tuning/tuning_policy.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool valid_group_size(int num_devices) {
    return num_devices > 0 && num_devices <= kOoMaxLocalDevices;
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

oo_status_t resolve_host_mapped_ready_ptr_for_current_device(
    oo_ready_signal& signal,
    int** out) {
    if (out == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out = nullptr;

    if (signal.kind != oo_ready_signal_kind::owned_host_mapped) {
        *out = reinterpret_cast<int*>(signal.ptr);
        return OO_SUCCESS;
    }

    if (signal.owned_host_ptr == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    void* device_ptr = nullptr;

    const cudaError_t err =
        cudaHostGetDevicePointer(
            &device_ptr,
            signal.owned_host_ptr,
            0);

    if (err != cudaSuccess) {
        return cuda_to_status(err);
    }

    *out = reinterpret_cast<int*>(device_ptr);
    return OO_SUCCESS;
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

    const size_t absolute_offset =
        base_element_offset + local_offset;

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

    const size_t local_bytes =
        local_count * dtype_size;

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
    CollectivePlanFor collective,
    size_t bytes,
    oo_tuning_mode_t tuning_mode) {
    return select_launch_config_for_collective(
        collective,
        bytes,
        tuning_preference_from_public(tuning_mode));
}


/*
 * OOVERLAP_IPC_LEGACY_BUFFER_REGISTRATION_HELPER_PATCH:
 *
 * Multiprocess p2p/legacy CUDA IPC buffer registration for public collectives.
 *
 * This path intentionally ignores VMM buffers.  The expected public IPC use is:
 *   - user / framework owns a cudaMalloc-like allocation
 *   - caller wraps it with oo_buffer_wrap(...)
 *   - before each collective, every rank exports its wrapped pointer as a
 *     cudaIpcMemHandle_t descriptor through Broker::exchange_data
 *   - every rank imports all peer descriptors and stores the imported mappings in
 *     group-owned ipc_imported_collective_buffers[]
 *
 * Same-process groups never call this helper.
 */
oo_status_t reset_ipc_imported_collective_buffer(
    oo_group_t* group,
    int rank) {
    if (group == nullptr ||
        rank < 0 ||
        rank >= kOoMaxLocalDevices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* old_buffer =
        group->ipc_imported_collective_buffers[rank].get();

    if (old_buffer != nullptr &&
        group->collective_buffers[rank] == old_buffer) {
        group->collective_buffers[rank] = nullptr;
    }

    group->ipc_imported_collective_buffers[rank].reset();
    return OO_SUCCESS;
}

oo_status_t ensure_ipc_legacy_collective_buffers_registered(
    oo_node_t* node,
    oo_buffer_t* local) {
    if (node == nullptr ||
        node->group == nullptr ||
        local == nullptr ||
        local->ptr == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_group_t* group = node->group;

    if (group->bootstrap_kind != oo_group_bootstrap_kind::multiprocess_ipc) {
        return OO_SUCCESS;
    }

    if (group->broker == nullptr ||
        group->memory_kind != oo_group_memory_kind::multiprocess_legacy_ipc ||
        group->num_devices <= 0 ||
        group->num_devices > kOoMaxLocalDevices ||
        group->local_world_size != group->num_devices ||
        group->local_rank != node->rank ||
        node->rank < 0 ||
        node->rank >= group->num_devices ||
        node->device != group->devices[node->rank] ||
        local->group != group ||
        local->owner_rank != node->rank ||
        local->owner_device != node->device ||
        local->bytes == 0 ||
        local->mapped_bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    /*
     * First p2p IPC implementation only supports externally owned cudaMalloc /
     * framework pointers registered via oo_buffer_wrap().  VMM-owned buffers use
     * a different FD-based import path and are intentionally excluded here.
     */
    if (local->system_kind != ooverlap::system::peer_buffer_kind::wrapped) {
        return OO_ERROR_UNSUPPORTED;
    }

    try {
        
        cudaSetDevice(node->device);
        const ooverlap::system::legacy_peer_buffer_descriptor local_desc =
            ooverlap::system::export_legacy_peer_buffer(
                local->ptr,
                local->bytes,
                local->owner_device,
                local->mapped_bytes != 0 ? local->mapped_bytes : local->bytes);

        std::vector<ooverlap::system::legacy_peer_buffer_descriptor> descs(
            static_cast<std::size_t>(group->num_devices));

        group->broker->exchange_data(
            descs.data(),
            &local_desc,
            sizeof(local_desc));

        group->collective_buffers[node->rank] = local;

        for (int rank = 0; rank < group->num_devices; ++rank) {
            if (rank == node->rank) {
                continue;
            }

            oo_status_t status =
                reset_ipc_imported_collective_buffer(
                    group,
                    rank);

            if (status != OO_SUCCESS) {
                return status;
            }

            const ooverlap::system::legacy_peer_buffer_descriptor& desc =
                descs[static_cast<std::size_t>(rank)];

            if (desc.bytes == 0 ||
                desc.mapped_size == 0 ||
                desc.owner_device != group->devices[rank]) {
                return OO_ERROR_INVALID_ARGUMENT;
            }

            ooverlap::system::imported_peer_buffer imported =
                ooverlap::system::import_legacy_peer_buffer(
                    desc,
                    std::vector<int>{node->device});

            std::unique_ptr<oo_buffer_t> imported_buffer(new oo_buffer_t{});

            imported_buffer->ptr = imported.ptr;
            imported_buffer->bytes = imported.bytes;
            imported_buffer->mapped_bytes = imported.mapped_size;
            imported_buffer->kind = OO_BUFFER_KIND_WRAPPED;
            imported_buffer->group = group;
            imported_buffer->owner_rank = rank;
            imported_buffer->owner_device = desc.owner_device;
            imported_buffer->system_kind =
                ooverlap::system::peer_buffer_kind::imported_legacy;
            imported_buffer->imported = std::move(imported);

            group->ipc_imported_collective_buffers[rank] =
                std::move(imported_buffer);

            group->collective_buffers[rank] =
                group->ipc_imported_collective_buffers[rank].get();
        }

        return OO_SUCCESS;
    } catch (...) {
        return exception_to_status();
    }
}


oo_status_t prepare_collective_launch(
    oo_node_t* node,
    oo_buffer_t* local,
    CollectivePlanFor collective,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    CollectiveLaunchState* out) {
    (void)collective;

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

    if (!valid_group_size(group->num_devices) ||
        group->num_devices > kOoMaxLocalDevices ||
        node->rank < 0 ||
        node->rank >= group->num_devices ||
        node->device != group->devices[node->rank] ||
        local->group != group ||
        local->owner_rank != node->rank ||
        local->owner_device != node->device) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    size_t offset_bytes = 0;
    size_t bytes = 0;

    oo_status_t status =
        checked_element_offset_bytes(
            element_offset,
            dtype,
            &offset_bytes);

    if (status != OO_SUCCESS) {
        return status;
    }

    status =
        checked_element_bytes(
            count,
            dtype,
            &bytes);

    if (status != OO_SUCCESS) {
        return status;
    }

    if (offset_bytes > local->bytes ||
        bytes > local->bytes - offset_bytes) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    /*
     * OOVERLAP_IPC_LEGACY_BUFFER_PREPARE_PATCH:
     *
     * Same-process behavior is unchanged: refresh this rank's local pointer and
     * immediately consume existing group->collective_buffers[].
     *
     * Multiprocess IPC behavior: exchange legacy CUDA IPC descriptors every
     * collective and refresh imported peer pointers before reading peers below.
     */
    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc) {
        status =
            ensure_ipc_legacy_collective_buffers_registered(
                node,
                local);

        if (status != OO_SUCCESS) {
            return status;
        }
    } else {
        group->collective_buffers[node->rank] = local;
    }

    const int world_size = group->num_devices;
    const int peer_count = world_size - 1;

    if (peer_count > kMaxPublicPeers) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_ready_signal& local_signal =
        group->ready_signal_slots[node->rank];
    oo_ready_signal& local_host_signal =
        group->host_ready_signal_slots[node->rank];

    {
        const cudaError_t err =
            cudaSetDevice(node->device);

        if (err != cudaSuccess) {
            return cuda_to_status(err);
        }
    }

    int* local_host_ready_signal = nullptr;

    status =
        resolve_host_mapped_ready_ptr_for_current_device(
            local_host_signal,
            &local_host_ready_signal);

    if (status != OO_SUCCESS) {
        return status;
    }

    out->local_ptr =
        reinterpret_cast<void*>(
            reinterpret_cast<std::uint8_t*>(local->ptr) + offset_bytes);
    out->peer_count = peer_count;
    out->rank = node->rank;
    out->world_size = world_size;
    out->local_device = node->device;
    out->dtype_size = oo_dtype_size(dtype);
    out->bytes = bytes;
    out->collective_epoch = ++node->collective_epoch;
    out->local_ready_signal =
        reinterpret_cast<int*>(local_signal.ptr);
    out->local_ready_signal_by_channel[kOoReadySignalChannelDeviceMemory] =
        reinterpret_cast<int*>(local_signal.ptr);
    out->local_ready_signal_by_channel[kOoReadySignalChannelHostMapped] =
        local_host_ready_signal;
    out->ready_signal_protocol_by_channel[kOoReadySignalChannelDeviceMemory] = 0;
    out->ready_signal_protocol_by_channel[kOoReadySignalChannelHostMapped] = 2;
    out->ready_signal_poll_sleep_cycles_by_channel
        [kOoReadySignalChannelDeviceMemory] = 64;
    out->ready_signal_poll_sleep_cycles_by_channel
        [kOoReadySignalChannelHostMapped] = 256;

    out->staging_slot_count = 0;

    for (int slot = 0; slot < kOoMaxStagingSlots; ++slot) {
        const oo_staging_buffer& staging =
            group->staging_slots[slot];

        out->staging_ptrs[slot] = staging.device_ptr;
        out->staging_bytes[slot] = staging.bytes;
        out->staging_numa_nodes[slot] = staging.numa_node;

        if (staging.device_ptr != nullptr && staging.bytes != 0) {
            out->staging_slot_count = slot + 1;
        }
    }

    int peer_idx = 0;

    for (int rank = 0; rank < world_size; ++rank) {
        if (rank == node->rank) {
            continue;
        }

        oo_buffer_t* peer = group->collective_buffers[rank];

        if (peer == nullptr ||
            peer->ptr == nullptr ||
            peer->group != group ||
            peer->owner_rank != rank) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        if (offset_bytes > peer->bytes ||
            bytes > peer->bytes - offset_bytes) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        out->peer_ptrs[peer_idx] =
            reinterpret_cast<void*>(
                reinterpret_cast<std::uint8_t*>(peer->ptr) + offset_bytes);
        out->peer_ranks[peer_idx] = rank;
        out->peer_devices[peer_idx] = peer->owner_device;

        oo_ready_signal& peer_signal =
            group->ready_signal_slots[rank];
        oo_ready_signal& peer_host_signal =
            group->host_ready_signal_slots[rank];

        out->peer_ready_signals[peer_idx] =
            reinterpret_cast<const int*>(peer_signal.ptr);
        out->peer_ready_signals_by_channel
            [peer_idx][kOoReadySignalChannelDeviceMemory] =
                reinterpret_cast<const int*>(peer_signal.ptr);
        int* peer_host_ready_signal = nullptr;

        status =
            resolve_host_mapped_ready_ptr_for_current_device(
                peer_host_signal,
                &peer_host_ready_signal);

        if (status != OO_SUCCESS) {
            return status;
        }

        out->peer_ready_signals_by_channel
            [peer_idx][kOoReadySignalChannelHostMapped] =
                reinterpret_cast<const int*>(peer_host_ready_signal);

        peer_idx += 1;
    }

    return OO_SUCCESS;
}

} // namespace api
} // namespace comm
} // namespace ooverlap

oo_status_t oo_buffer_export_legacy_descriptor(
    oo_buffer_t* buffer,
    ooverlap::system::legacy_peer_buffer_descriptor* out_desc) {
    if (buffer == nullptr ||
        buffer->ptr == nullptr ||
        out_desc == nullptr ||
        buffer->owner_device < 0 ||
        buffer->bytes == 0) {
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
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_buffer = nullptr;

    if (node == nullptr ||
        node->group == nullptr ||
        imported.ptr == nullptr ||
        imported.bytes == 0 ||
        imported.mapped_size == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::unique_ptr<oo_buffer_t> buffer(new oo_buffer_t{});

        buffer->ptr = imported.ptr;
        buffer->bytes = imported.bytes;
        buffer->mapped_bytes = imported.mapped_size;
        buffer->kind = OO_BUFFER_KIND_WRAPPED;
        buffer->group = node->group;
        buffer->owner_rank = -1;
        buffer->owner_device = imported.owner_device;
        buffer->system_kind = imported.kind;
        buffer->imported = std::move(imported);

        *out_buffer = buffer.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

oo_status_t oo_buffer_import_legacy_descriptor(
    oo_node_t* node,
    const ooverlap::system::legacy_peer_buffer_descriptor& desc,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_buffer = nullptr;

    if (node == nullptr ||
        node->group == nullptr ||
        node->device < 0 ||
        desc.bytes == 0 ||
        desc.mapped_size == 0 ||
        desc.owner_device < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto imported =
            ooverlap::system::import_legacy_peer_buffer(
                desc,
                std::vector<int>{node->device});

        oo_status_t status =
            oo_buffer_adopt_imported_peer_buffer(
                node,
                std::move(imported),
                out_buffer);

        if (status != OO_SUCCESS) {
            return status;
        }

        (*out_buffer)->owner_device = desc.owner_device;

        for (int rank = 0; rank < node->group->num_devices; ++rank) {
            if (node->group->devices[rank] == desc.owner_device) {
                (*out_buffer)->owner_rank = rank;
                break;
            }
        }

        if ((*out_buffer)->owner_rank >= 0 &&
            (*out_buffer)->owner_rank < kOoMaxLocalDevices) {
            node->group->collective_buffers[(*out_buffer)->owner_rank] =
                *out_buffer;
        }

        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}
