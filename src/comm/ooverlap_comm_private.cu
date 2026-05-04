#include "comm/ooverlap_comm_private.h"

#include "comm/tuning/tuning_policy.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

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

    if (node->rank < 0 ||
        node->rank >= group->num_devices ||
        node->device != group->devices[node->rank]) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (local->group != group ||
        local->owner_rank != node->rank ||
        local->owner_device != node->device) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (peer_count != group->num_devices - 1) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (peer_count > 0 && peers == nullptr) {
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

    if (peer_count > kMaxPublicPeers) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    out->local_ptr =
        reinterpret_cast<void*>(
            reinterpret_cast<std::uint8_t*>(local->ptr) + offset_bytes);

    out->peer_count = peer_count;
    out->rank = node->rank;
    out->world_size = group->num_devices;
    out->local_device = node->device;
    out->dtype_size = oo_dtype_size(dtype);
    out->bytes = bytes;
    out->collective_epoch = ++node->collective_epoch;

    oo_ready_signal& local_signal =
        group->ready_signal_slots[node->rank];

    out->local_ready_signal =
        reinterpret_cast<int*>(local_signal.ptr);

    for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
        oo_buffer_t* peer = peers[peer_idx];

        if (peer == nullptr ||
            peer->ptr == nullptr ||
            peer->group != group ||
            peer->owner_rank < 0 ||
            peer->owner_rank >= group->num_devices ||
            peer->owner_rank == node->rank) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        if (offset_bytes > peer->bytes ||
            bytes > peer->bytes - offset_bytes) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        out->peer_ptrs[peer_idx] =
            reinterpret_cast<void*>(
                reinterpret_cast<std::uint8_t*>(peer->ptr) + offset_bytes);

        oo_ready_signal& peer_signal =
            group->ready_signal_slots[peer->owner_rank];

        out->peer_ready_signals[peer_idx] =
            reinterpret_cast<const int*>(peer_signal.ptr);
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

        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}
