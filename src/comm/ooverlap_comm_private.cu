#include "comm/ooverlap_comm_private.h"

#include "comm/tuning/tuning_policy.h"
#include "ooverlap/comm.h"
#include "ooverlap/system/runtime_utils.cuh"

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
 * OOVERLAP_IPC_MULTI_ENTRY_IMPORT_CACHE_PATCH:
 *
 * Simple YAGNI IPC import cache:
 *   - oo_buffer_wrap() remains local and cheap.
 *   - oo_buffer_register_ipc() pre-fills this cache.
 *   - public collectives call the same helper and import on cache miss.
 *   - cache key is an owner-rank token exchanged through Broker, never the
 *     process-local imported pointer value.
 */
oo_group::ipc_import_key make_ipc_import_key(
    const oo_buffer_t* buffer) {
    oo_group::ipc_import_key key{};

    if (buffer == nullptr) {
        return key;
    }

    key.owner_ptr_value =
        reinterpret_cast<std::uintptr_t>(buffer->ptr);
    key.bytes =
        static_cast<std::uint64_t>(buffer->bytes);
    key.mapped_bytes =
        static_cast<std::uint64_t>(buffer->mapped_bytes);
    key.cache_token =
        static_cast<std::uint64_t>(buffer->ipc_cache_token);
    key.owner_rank = buffer->owner_rank;
    key.owner_device = buffer->owner_device;
    return key;
}

bool same_ipc_import_key(
    const oo_group::ipc_import_key& a,
    const oo_group::ipc_import_key& b) {
    return a.owner_ptr_value == b.owner_ptr_value &&
           a.bytes == b.bytes &&
           a.mapped_bytes == b.mapped_bytes &&
           a.cache_token == b.cache_token &&
           a.owner_rank == b.owner_rank &&
           a.owner_device == b.owner_device;
}

bool valid_ipc_import_key_for_rank(
    const oo_group_t* group,
    const oo_group::ipc_import_key& key,
    int rank) {
    return group != nullptr &&
           rank >= 0 &&
           rank < group->num_devices &&
           key.owner_ptr_value != 0 &&
           key.bytes != 0 &&
           key.mapped_bytes != 0 &&
           key.cache_token != 0 &&
           key.owner_rank == rank &&
           key.owner_device == group->devices[rank];
}

oo_group::ipc_import_cache_entry* find_ipc_import_cache_entry(
    oo_group_t* group,
    int owner_rank,
    const oo_group::ipc_import_key& key) {
    if (group == nullptr ||
        owner_rank < 0 ||
        owner_rank >= group->num_devices) {
        return nullptr;
    }

    for (int slot = 0; slot < kOoIpcImportCacheEntriesPerRank; ++slot) {
        oo_group::ipc_import_cache_entry& entry =
            group->ipc_import_cache[owner_rank][slot];

        if (!entry.valid ||
            entry.buffer == nullptr ||
            entry.buffer->ptr == nullptr) {
            continue;
        }

        if (same_ipc_import_key(entry.key, key)) {
            entry.last_used = ++group->ipc_import_cache_clock;
            return &entry;
        }
    }

    return nullptr;
}

void clear_ipc_import_cache_entry(
    oo_group_t* group,
    int owner_rank,
    oo_group::ipc_import_cache_entry& entry) {
    if (group != nullptr &&
        owner_rank >= 0 &&
        owner_rank < kOoMaxLocalDevices &&
        entry.buffer != nullptr &&
        group->collective_buffers[owner_rank] == entry.buffer.get()) {
        group->collective_buffers[owner_rank] = nullptr;
    }

    entry.buffer.reset();
    entry.key = {};
    entry.valid = false;
    entry.last_used = 0;
}

oo_group::ipc_import_cache_entry* select_ipc_import_cache_slot(
    oo_group_t* group,
    int owner_rank) {
    if (group == nullptr ||
        owner_rank < 0 ||
        owner_rank >= group->num_devices) {
        return nullptr;
    }

    oo_group::ipc_import_cache_entry* best =
        &group->ipc_import_cache[owner_rank][0];

    for (int slot = 0; slot < kOoIpcImportCacheEntriesPerRank; ++slot) {
        oo_group::ipc_import_cache_entry& entry =
            group->ipc_import_cache[owner_rank][slot];

        if (!entry.valid || entry.buffer == nullptr) {
            return &entry;
        }

        if (entry.last_used < best->last_used) {
            best = &entry;
        }
    }

    clear_ipc_import_cache_entry(
        group,
        owner_rank,
        *best);

    return best;
}

oo_status_t import_peer_into_cache(
    oo_group_t* group,
    int owner_rank,
    const oo_group::ipc_import_key& key,
    const ooverlap::system::legacy_peer_buffer_descriptor& desc,
    oo_group::ipc_import_cache_entry** out_entry) {
    if (out_entry == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_entry = nullptr;

    if (group == nullptr ||
        !valid_ipc_import_key_for_rank(group, key, owner_rank) ||
        desc.bytes == 0 ||
        desc.mapped_size == 0 ||
        desc.owner_device != key.owner_device) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_group::ipc_import_cache_entry* entry =
        select_ipc_import_cache_slot(
            group,
            owner_rank);

    if (entry == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    ooverlap::system::imported_peer_buffer imported =
        ooverlap::system::import_legacy_peer_buffer(
            desc,
            1);

    std::unique_ptr<oo_buffer_t> imported_buffer(new oo_buffer_t{});

    imported_buffer->ptr = imported.ptr;
    imported_buffer->bytes = imported.bytes;
    imported_buffer->mapped_bytes = imported.mapped_size;
    imported_buffer->kind = OO_BUFFER_KIND_WRAPPED;
    imported_buffer->group = group;
    imported_buffer->owner_rank = owner_rank;
    imported_buffer->owner_device = desc.owner_device;
    imported_buffer->ipc_cache_token = key.cache_token;
    imported_buffer->system_kind =
        ooverlap::system::peer_buffer_kind::imported_legacy;
    imported_buffer->imported = std::move(imported);

    entry->buffer =
        std::move(imported_buffer);
    entry->key = key;
    entry->valid = true;
    entry->last_used = ++group->ipc_import_cache_clock;

    *out_entry = entry;
    return OO_SUCCESS;
}

oo_status_t ensure_ipc_legacy_collective_buffers_registered(
    oo_node_t* node,
    oo_buffer_t* local) {
    
    oo_group_t* group = node->group;

    if (local->system_kind != ooverlap::system::peer_buffer_kind::wrapped) {
        return OO_ERROR_UNSUPPORTED;
    }

    try {
        cudaSetDevice(node->device);

        const oo_group::ipc_import_key local_key =
            make_ipc_import_key(local);

        std::vector<oo_group::ipc_import_key> keys(
            static_cast<std::size_t>(group->num_devices));

        group->broker->exchange_data(
            keys.data(),
            &local_key,
            sizeof(local_key));

        group->collective_buffers[node->rank] = local;

        oo_group::ipc_import_cache_entry* hits[kOoMaxLocalDevices] = {};
        std::uint32_t local_miss_mask = 0;

        for (int rank = 0; rank < group->num_devices; ++rank) {
            const oo_group::ipc_import_key& key =
                keys[static_cast<std::size_t>(rank)];
            
            if (rank == node->rank) {
                continue;
            }

            hits[rank] =
                find_ipc_import_cache_entry(
                    group,
                    rank,
                    key);

            if (hits[rank] == nullptr) {
                local_miss_mask |= (std::uint32_t{1} << rank);
            }
        }


        std::vector<std::uint32_t> miss_masks(
            static_cast<std::size_t>(group->num_devices),
            0);

        group->broker->exchange_data(
            miss_masks.data(),
            &local_miss_mask,
            sizeof(local_miss_mask));

        std::uint32_t global_miss_mask = 0;
        for (int rank = 0; rank < group->num_devices; ++rank) {
            global_miss_mask |=
                miss_masks[static_cast<std::size_t>(rank)];
        }

        std::vector<ooverlap::system::legacy_peer_buffer_descriptor> descs;

        if (global_miss_mask != 0) {
            const ooverlap::system::legacy_peer_buffer_descriptor local_desc =
                ooverlap::system::export_legacy_peer_buffer(
                    local->ptr,
                    local->bytes,
                    local->owner_device,
                    local->mapped_bytes != 0 ? local->mapped_bytes : local->bytes);

            descs.resize(
                static_cast<std::size_t>(group->num_devices));

            group->broker->exchange_data(
                descs.data(),
                &local_desc,
                sizeof(local_desc));
        }

        for (int rank = 0; rank < group->num_devices; ++rank) {
            if (rank == node->rank) {
                continue;
            }

            oo_group::ipc_import_cache_entry* entry = hits[rank];

            if (entry == nullptr) {
                if (descs.empty()) {
                    return OO_ERROR_INTERNAL;
                }

                oo_status_t status =
                    import_peer_into_cache(
                        group,
                        rank,
                        keys[static_cast<std::size_t>(rank)],
                        descs[static_cast<std::size_t>(rank)],
                        &entry);

                if (status != OO_SUCCESS) {
                    return status;
                }
            }

            if (entry == nullptr ||
                entry->buffer == nullptr ||
                entry->buffer->ptr == nullptr) {
                return OO_ERROR_INTERNAL;
            }

            group->collective_buffers[rank] =
                entry->buffer.get();
        }

        return OO_SUCCESS;
    } catch (...) {
        return exception_to_status();
    }
}

oo_status_t register_ipc_collective_buffers(
    oo_node_t* node,
    oo_buffer_t* local) {
    return ensure_ipc_legacy_collective_buffers_registered(
        node,
        local);
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

    // We don't need this
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

    // This one either
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

        // HERE
        return OO_SUCCESS;

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
