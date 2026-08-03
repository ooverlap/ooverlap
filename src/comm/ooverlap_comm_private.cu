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

    void* base_ptr =
        buffer->ipc_base_ptr != nullptr ? buffer->ipc_base_ptr : buffer->ptr;
    const size_t base_bytes =
        buffer->ipc_base_bytes != 0
            ? buffer->ipc_base_bytes
            : (buffer->mapped_bytes != 0 ? buffer->mapped_bytes : buffer->bytes);

    key.owner_ptr_value =
        reinterpret_cast<std::uintptr_t>(base_ptr);
    key.bytes =
        static_cast<std::uint64_t>(buffer->bytes);
    key.mapped_bytes =
        static_cast<std::uint64_t>(base_bytes);
    key.logical_offset_bytes =
        static_cast<std::uint64_t>(buffer->ipc_logical_offset_bytes);
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
           a.logical_offset_bytes == b.logical_offset_bytes &&
           a.cache_token == b.cache_token &&
           a.owner_rank == b.owner_rank &&
           a.owner_device == b.owner_device;
}


/*
 * OOVERLAP_IPC_SYMMETRIC_LOCAL_KEY_FAST_PATH_PATCH:
 *
 * Weak symmetric-collective fast path.
 *
 * In the benchmark / expected symmetric usage, all ranks call the same
 * collective sequence with the same local buffer sequence.  Therefore, if this
 * rank's local buffer wrapper/key did not change since the previous collective,
 * we assume peer ranks did not change their buffers either.
 *
 * This intentionally avoids Broker::exchange_data on the hot path.  It is not a
 * general correctness contract for arbitrary asymmetric buffer changes.  If one
 * rank changes buffers while another rank does not, ranks can take different
 * host paths and deadlock.  Replace this later with explicit epochs/versioning
 * or re-enable the all-rank key exchange for fully general usage.
 */
bool ipc_symmetric_local_key_fast_path_ready(
    oo_group_t* group,
    int local_rank,
    const oo_group::ipc_import_key& local_key) {
    if (group == nullptr ||
        local_rank < 0 ||
        local_rank >= group->num_devices) {
        return false;
    }

    oo_buffer_t* previous_local =
        group->collective_buffers[local_rank];

    if (previous_local == nullptr ||
        previous_local->ptr == nullptr ||
        !same_ipc_import_key(
            make_ipc_import_key(previous_local),
            local_key)) {
        return false;
    }

    for (int rank = 0; rank < group->num_devices; ++rank) {
        if (rank == local_rank) {
            continue;
        }

        oo_buffer_t* peer =
            group->collective_buffers[rank];

        if (peer == nullptr ||
            peer->ptr == nullptr ||
            peer->owner_rank != rank ||
            peer->owner_device != group->devices[rank] ||
            peer->system_kind != ooverlap::system::peer_buffer_kind::imported_legacy) {
            return false;
        }
    }

    return true;
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
           key.logical_offset_bytes <= key.mapped_bytes &&
           key.bytes <= key.mapped_bytes - key.logical_offset_bytes &&
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
    imported_buffer->ipc_base_ptr =
        imported.mapping_base_ptr != nullptr ? imported.mapping_base_ptr : imported.ptr;
    imported_buffer->ipc_base_bytes = imported.mapped_size;
    imported_buffer->ipc_logical_offset_bytes = imported.logical_offset;
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

        if (ipc_symmetric_local_key_fast_path_ready(
                group,
                node->rank,
                local_key)) {
            group->collective_buffers[node->rank] = local;
            return OO_SUCCESS;
        }

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
            void* local_base_ptr =
                local->ipc_base_ptr != nullptr ? local->ipc_base_ptr : local->ptr;
            const size_t local_base_bytes =
                local->ipc_base_bytes != 0
                    ? local->ipc_base_bytes
                    : (local->mapped_bytes != 0 ? local->mapped_bytes : local->bytes);

            const ooverlap::system::legacy_peer_buffer_descriptor local_desc =
                ooverlap::system::export_legacy_peer_buffer_range(
                    local_base_ptr,
                    local_base_bytes,
                    local->ipc_logical_offset_bytes,
                    local->bytes,
                    local->owner_device);

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


oo_status_t prepare_fast_allreduce_launch(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* prebound_rank_buffers,
    int prebound_rank_buffer_count,
    size_t offset_bytes,
    size_t bytes,
    FastAllreduceLaunchState* out) {
    if (out == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out = FastAllreduceLaunchState{};

    if (node == nullptr ||
        node->group == nullptr ||
        local == nullptr ||
        local->ptr == nullptr ||
        bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_group_t* group = node->group;

    if (node->rank < 0 ||
        node->rank >= group->num_devices ||
        group->num_devices < 2 ||
        group->num_devices - 1 > kMaxPublicPeers) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (prebound_rank_buffers != nullptr) {
        if (prebound_rank_buffer_count != group->num_devices ||
            prebound_rank_buffers[node->rank] != local) {
            return OO_ERROR_INVALID_ARGUMENT;
        }
    } else if (
        group->bootstrap_kind ==
            oo_group_bootstrap_kind::multiprocess_ipc) {
        const oo_status_t status =
            ensure_ipc_legacy_collective_buffers_registered(
                node,
                local);

        if (status != OO_SUCCESS) {
            return status;
        }
    } else {
        group->collective_buffers[node->rank] = local;
    }

    const cudaError_t set_device_error =
        cudaSetDevice(node->device);

    if (set_device_error != cudaSuccess) {
        return cuda_to_status(set_device_error);
    }

    oo_ready_signal& local_ready_slot =
        group->ready_signal_slots[node->rank];

    if (local_ready_slot.ptr == nullptr ||
        local_ready_slot.bytes < kOoReadySignalBytes) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    int* local_ready_base =
        reinterpret_cast<int*>(local_ready_slot.ptr);

    int peer_idx = 0;

    for (int rank = 0; rank < group->num_devices; ++rank) {
        oo_buffer_t* buffer =
            prebound_rank_buffers != nullptr
                ? prebound_rank_buffers[rank]
                : group->collective_buffers[rank];

        if (buffer == nullptr ||
            buffer->ptr == nullptr ||
            buffer->group != group ||
            buffer->owner_rank != rank ||
            buffer->owner_device != group->devices[rank] ||
            offset_bytes > buffer->bytes ||
            bytes > buffer->bytes - offset_bytes) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        void* logical_ptr =
            reinterpret_cast<void*>(
                reinterpret_cast<std::uint8_t*>(buffer->ptr) +
                offset_bytes);

        if ((reinterpret_cast<std::uintptr_t>(logical_ptr) &
             static_cast<std::uintptr_t>(15)) != 0) {
            return OO_ERROR_UNSUPPORTED;
        }

        oo_ready_signal& ready_signal =
            group->ready_signal_slots[rank];

        if (ready_signal.ptr == nullptr ||
            ready_signal.bytes < kOoReadySignalBytes) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        if (rank == node->rank) {
            out->local_ptr = logical_ptr;
            continue;
        }

        int* peer_ready_base =
            reinterpret_cast<int*>(ready_signal.ptr);

        out->peer_ptrs[peer_idx] = logical_ptr;
        out->peer_publish_signals[peer_idx] =
            peer_ready_base +
            kOoReadySignalInboxBaseSlot +
            node->rank;
        out->local_wait_signals[peer_idx] =
            local_ready_base +
            kOoReadySignalInboxBaseSlot +
            rank;
        ++peer_idx;
    }

    if (out->local_ptr == nullptr ||
        peer_idx != group->num_devices - 1) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    out->peer_count = peer_idx;
    out->rank = node->rank;
    out->world_size = group->num_devices;
    out->local_device = node->device;
    out->collective_epoch = ++node->collective_epoch;
    out->bytes = bytes;

    return OO_SUCCESS;
}


/* OOVERLAP_ROUND_ROBIN_SLOT_POOL_PATCH_V1 */
oo_status_t prepare_collective_launch_impl(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* prebound_rank_buffers,
    int prebound_rank_buffer_count,
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

    if (prebound_rank_buffers != nullptr) {
        if (prebound_rank_buffer_count != group->num_devices ||
            node->rank < 0 ||
            node->rank >= prebound_rank_buffer_count ||
            prebound_rank_buffers[node->rank] != local) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        for (int rank = 0; rank < prebound_rank_buffer_count; ++rank) {
            oo_buffer_t* buffer = prebound_rank_buffers[rank];
            if (buffer == nullptr ||
                buffer->ptr == nullptr ||
                buffer->group != group ||
                buffer->owner_rank != rank ||
                buffer->owner_device != group->devices[rank]) {
                return OO_ERROR_INVALID_ARGUMENT;
            }
        }
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
    if (prebound_rank_buffers == nullptr) {
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

    if (local_signal.ptr == nullptr ||
        local_signal.bytes < kOoReadySignalBytes) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    int* local_device_ready_base =
        reinterpret_cast<int*>(local_signal.ptr);

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

    /*
     * OOVERLAP_OUT_OF_PLACE_ALLREDUCE_REDUCE_FANOUT_PATCH
     *
     * Default public path is still in-place. Dedicated out-of-place plumbing can
     * overwrite these fields after prepare_collective_launch() or use a separate
     * prepare helper later.
     */
    out->out_of_place = false;
    out->local_input_ptr = out->local_ptr;
    out->local_output_ptr = out->local_ptr;

    out->peer_count = peer_count;
    out->rank = node->rank;
    out->world_size = world_size;
    out->local_device = node->device;
    out->dtype_size = oo_dtype_size(dtype);
    out->bytes = bytes;
    out->collective_epoch = ++node->collective_epoch;
    out->local_ready_signal =
        local_device_ready_base + kOoReadySignalLegacySlot;
    out->local_ready_signal_by_channel[kOoReadySignalChannelDeviceMemory] =
        local_device_ready_base + kOoReadySignalLegacySlot;
    out->local_ready_signal_by_channel[kOoReadySignalChannelHostMapped] =
        local_host_ready_signal != nullptr
            ? local_host_ready_signal + kOoReadySignalLegacySlot
            : nullptr;
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

        oo_buffer_t* peer =
            prebound_rank_buffers != nullptr
                ? prebound_rank_buffers[rank]
                : group->collective_buffers[rank];

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
        out->peer_input_ptrs[peer_idx] = out->peer_ptrs[peer_idx];
        out->peer_output_ptrs[peer_idx] = out->peer_ptrs[peer_idx];

        out->peer_ranks[peer_idx] = rank;
        out->peer_devices[peer_idx] = peer->owner_device;

        oo_ready_signal& peer_signal =
            group->ready_signal_slots[rank];
        oo_ready_signal& peer_host_signal =
            group->host_ready_signal_slots[rank];

        if (peer_signal.ptr == nullptr ||
            peer_signal.bytes < kOoReadySignalBytes) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        int* peer_device_ready_base =
            reinterpret_cast<int*>(peer_signal.ptr);

        out->peer_ready_signals[peer_idx] =
            peer_device_ready_base + kOoReadySignalLegacySlot;
        out->peer_publish_signals[peer_idx] =
            peer_device_ready_base +
            kOoReadySignalInboxBaseSlot +
            node->rank;
        out->local_wait_signals[peer_idx] =
            local_device_ready_base +
            kOoReadySignalInboxBaseSlot +
            rank;
        out->peer_ready_signals_by_channel
            [peer_idx][kOoReadySignalChannelDeviceMemory] =
                peer_device_ready_base + kOoReadySignalLegacySlot;
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
                peer_host_ready_signal != nullptr
                    ? peer_host_ready_signal + kOoReadySignalLegacySlot
                    : nullptr;

        peer_idx += 1;
    }

    return OO_SUCCESS;
}

oo_status_t prepare_collective_launch(
    oo_node_t* node,
    oo_buffer_t* local,
    CollectivePlanFor collective,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    CollectiveLaunchState* out) {
    return prepare_collective_launch_impl(
        node,
        local,
        nullptr,
        0,
        collective,
        element_offset,
        count,
        dtype,
        out);
}

oo_status_t prepare_collective_launch_prebound(
    oo_node_t* node,
    oo_buffer_t* const* rank_buffers,
    int rank_buffer_count,
    CollectivePlanFor collective,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    CollectiveLaunchState* out) {
    if (node == nullptr ||
        node->group == nullptr ||
        rank_buffers == nullptr ||
        rank_buffer_count != node->group->num_devices ||
        node->rank < 0 ||
        node->rank >= rank_buffer_count) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    return prepare_collective_launch_impl(
        node,
        rank_buffers[node->rank],
        rank_buffers,
        rank_buffer_count,
        collective,
        element_offset,
        count,
        dtype,
        out);
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
        void* base_ptr =
            buffer->ipc_base_ptr != nullptr ? buffer->ipc_base_ptr : buffer->ptr;
        const size_t base_bytes =
            buffer->ipc_base_bytes != 0
                ? buffer->ipc_base_bytes
                : (buffer->mapped_bytes != 0 ? buffer->mapped_bytes : buffer->bytes);

        *out_desc =
            ooverlap::system::export_legacy_peer_buffer_range(
                base_ptr,
                base_bytes,
                buffer->ipc_logical_offset_bytes,
                buffer->bytes,
                buffer->owner_device);

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
        buffer->ipc_base_ptr =
            imported.mapping_base_ptr != nullptr ? imported.mapping_base_ptr : imported.ptr;
        buffer->ipc_base_bytes = imported.mapped_size;
        buffer->ipc_logical_offset_bytes = imported.logical_offset;
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

/*
 * OOVERLAP_ROUND_ROBIN_SLOT_POOL_PATCH_V1
 *
 * Setup-only exchange/import. There is deliberately no cache lookup and no
 * mutation of group->collective_buffers. The returned set owns imported peer
 * mappings for its full lifetime and borrows the local slot wrappers.
 */
oo_status_t oo_ipc_slot_set_create(
    oo_node_t* node,
    oo_buffer_t* const* local_slots,
    int slot_count,
    oo_ipc_slot_set_t** out_set) {
    if (out_set == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    *out_set = nullptr;

    if (node == nullptr ||
        node->group == nullptr ||
        local_slots == nullptr ||
        slot_count <= 0 ||
        node->group->bootstrap_kind !=
            oo_group_bootstrap_kind::multiprocess_ipc ||
        node->group->broker == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        oo_group_t* group = node->group;
        const int world_size = group->num_devices;

        const cudaError_t device_error = cudaSetDevice(node->device);
        if (device_error != cudaSuccess) {
            return ooverlap::comm::api::cuda_to_status(device_error);
        }

        std::unique_ptr<oo_ipc_slot_set_t> set(new oo_ipc_slot_set_t{});
        set->group = group;
        set->world_size = world_size;
        set->slot_count = slot_count;
        set->rank_buffers.resize(
            static_cast<size_t>(slot_count) * static_cast<size_t>(world_size),
            nullptr);
        set->imported_buffers.reserve(
            static_cast<size_t>(slot_count) *
            static_cast<size_t>(world_size > 0 ? world_size - 1 : 0));

        for (int slot_index = 0; slot_index < slot_count; ++slot_index) {
            oo_buffer_t* local = local_slots[slot_index];

            if (local == nullptr ||
                local->ptr == nullptr ||
                local->bytes == 0 ||
                local->group != group ||
                local->owner_rank != node->rank ||
                local->owner_device != node->device ||
                local->system_kind !=
                    ooverlap::system::peer_buffer_kind::wrapped) {
                return OO_ERROR_INVALID_ARGUMENT;
            }

            if (slot_index == 0) {
                set->slot_bytes = local->bytes;
            } else if (local->bytes != set->slot_bytes) {
                return OO_ERROR_INVALID_ARGUMENT;
            }

            ooverlap::system::legacy_peer_buffer_descriptor local_desc{};
            oo_status_t status =
                oo_buffer_export_legacy_descriptor(local, &local_desc);
            if (status != OO_SUCCESS) {
                return status;
            }

            std::vector<ooverlap::system::legacy_peer_buffer_descriptor> descs(
                static_cast<size_t>(world_size));
            group->broker->exchange_data(
                descs.data(),
                &local_desc,
                sizeof(local_desc));

            for (int rank = 0; rank < world_size; ++rank) {
                const auto& desc = descs[static_cast<size_t>(rank)];
                if (desc.bytes != set->slot_bytes ||
                    desc.mapped_size == 0 ||
                    desc.logical_offset > desc.mapped_size ||
                    desc.bytes > desc.mapped_size - desc.logical_offset ||
                    desc.owner_device != group->devices[rank]) {
                    return OO_ERROR_INVALID_ARGUMENT;
                }

                const size_t flat_index =
                    static_cast<size_t>(slot_index) *
                        static_cast<size_t>(world_size) +
                    static_cast<size_t>(rank);

                if (rank == node->rank) {
                    set->rank_buffers[flat_index] = local;
                    continue;
                }

                auto imported =
                    ooverlap::system::import_legacy_peer_buffer(
                        desc,
                        std::vector<int>{node->device});

                std::unique_ptr<oo_buffer_t> peer(new oo_buffer_t{});
                peer->ptr = imported.ptr;
                peer->bytes = imported.bytes;
                peer->mapped_bytes = imported.mapped_size;
                peer->ipc_base_ptr =
                    imported.mapping_base_ptr != nullptr
                        ? imported.mapping_base_ptr
                        : imported.ptr;
                peer->ipc_base_bytes = imported.mapped_size;
                peer->ipc_logical_offset_bytes = imported.logical_offset;
                peer->kind = OO_BUFFER_KIND_WRAPPED;
                peer->group = group;
                peer->owner_rank = rank;
                peer->owner_device = desc.owner_device;
                peer->system_kind =
                    ooverlap::system::peer_buffer_kind::imported_legacy;
                peer->imported = std::move(imported);

                set->rank_buffers[flat_index] = peer.get();
                set->imported_buffers.push_back(std::move(peer));
            }
        }

        *out_set = set.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

void oo_ipc_slot_set_destroy(
    oo_ipc_slot_set_t* set) {
    delete set;
}

int oo_ipc_slot_set_count(
    const oo_ipc_slot_set_t* set) {
    return set != nullptr ? set->slot_count : 0;
}

