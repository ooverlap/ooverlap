#include "comm/ooverlap_comm_private.h"

#include "ooverlap/system/p2p.cuh"
#include "ooverlap/system/runtime_utils.cuh"

#include "comm/plan/transfer_plan_distribution.h"
#include "comm/utils/collective_utils.h"
#include "topology/topology.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

bool valid_group_size(int num_devices) {
    return num_devices > 0 && num_devices <= kOoMaxLocalDevices;
}

std::vector<int> devices_vector(
    const int* devices,
    int num_devices) {
    std::vector<int> out;
    out.reserve(static_cast<size_t>(num_devices));

    for (int i = 0; i < num_devices; ++i) {
        out.push_back(devices[i]);
    }

    return out;
}

std::uint64_t next_buffer_ipc_cache_token() {
    static std::atomic<std::uint64_t> next{1};

    return next.fetch_add(
        1,
        std::memory_order_relaxed);
}

/*
 * Best-effort static NUMA placement for staging slots.
 *
 * Tune and recompile.  If the machine has fewer NUMA nodes, mbind may fail; that
 * is intentionally non-fatal because the staging allocation is still usable.
 */
constexpr int kOoStagingPreferredNumaNodeCount = 2;
constexpr int kOoStagingPreferredNumaNodes[kOoStagingPreferredNumaNodeCount] = {
    0,
    1,
};

int preferred_staging_numa_node(int slot) {
    if (slot < 0 || kOoStagingPreferredNumaNodeCount <= 0) {
        return -1;
    }

    return kOoStagingPreferredNumaNodes[
        slot % kOoStagingPreferredNumaNodeCount];
}

bool bind_memory_to_numa_node_best_effort(
    void* ptr,
    size_t bytes,
    int numa_node) {
    if (ptr == nullptr || bytes == 0 || numa_node < 0) {
        return false;
    }

#if defined(__linux__) && defined(SYS_mbind)
    if (numa_node >= static_cast<int>(8 * sizeof(unsigned long))) {
        return false;
    }

    unsigned long nodemask =
        1ul << static_cast<unsigned int>(numa_node);

    /*
     * MPOL_BIND before pages are touched.  This is best-effort only.
     *
     * Do not make group creation fail when mbind is unavailable or denied; the
     * staging buffer is a correctness resource, NUMA placement is optimization.
     */
    const long rc =
        syscall(
            SYS_mbind,
            ptr,
            bytes,
            MPOL_BIND,
            &nodemask,
            static_cast<unsigned long>(numa_node + 1),
            0);

    return rc == 0;
#else
    (void)ptr;
    (void)bytes;
    (void)numa_node;
    return false;
#endif
}

void clear_staging_buffer(oo_staging_buffer& slot) {
    if (slot.host_ptr != nullptr) {
        if (slot.kind == oo_staging_buffer_kind::owned_host_registered) {
            cudaHostUnregister(slot.host_ptr);

#if defined(__linux__)
            munmap(slot.host_ptr, slot.bytes);
#endif
        } else if (slot.kind == oo_staging_buffer_kind::owned_host_mapped) {
            cudaFreeHost(slot.host_ptr);
        }
    }

    slot = oo_staging_buffer{};
}

void destroy_group_staging_buffers(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    for (int slot = 0; slot < kOoMaxStagingSlots; ++slot) {
        clear_staging_buffer(group->staging_slots[slot]);
    }
}

oo_status_t allocate_group_staging_buffers(oo_group_t* group) {
    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    static_assert(
        kOoMaxStagingSlots >= 0,
        "kOoMaxStagingSlots must be non-negative");
    static_assert(
        kOoStagingSlotBytes > 0,
        "kOoStagingSlotBytes must be positive");

    for (int slot_idx = 0; slot_idx < kOoMaxStagingSlots; ++slot_idx) {
        oo_staging_buffer& slot =
            group->staging_slots[slot_idx];

        if (slot.host_ptr != nullptr || slot.device_ptr != nullptr) {
            destroy_group_staging_buffers(group);
            return OO_ERROR_INVALID_ARGUMENT;
        }

        const int numa_node =
            preferred_staging_numa_node(slot_idx);

        void* host_ptr = nullptr;
        void* device_ptr = nullptr;
        cudaError_t err = cudaSuccess;

#if defined(__linux__)
        host_ptr =
            mmap(
                nullptr,
                kOoStagingSlotBytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0);

        if (host_ptr == MAP_FAILED) {
            host_ptr = nullptr;
            destroy_group_staging_buffers(group);
            return OO_ERROR_INTERNAL;
        }

        bind_memory_to_numa_node_best_effort(
            host_ptr,
            kOoStagingSlotBytes,
            numa_node);

        std::memset(host_ptr, 0, kOoStagingSlotBytes);

        err =
            cudaHostRegister(
                host_ptr,
                kOoStagingSlotBytes,
                cudaHostRegisterMapped | cudaHostRegisterPortable);

        if (err != cudaSuccess) {
            munmap(host_ptr, kOoStagingSlotBytes);
            destroy_group_staging_buffers(group);
            return ooverlap::comm::api::cuda_to_status(err);
        }

        err =
            cudaHostGetDevicePointer(
                &device_ptr,
                host_ptr,
                0);

        if (err != cudaSuccess) {
            cudaHostUnregister(host_ptr);
            munmap(host_ptr, kOoStagingSlotBytes);
            destroy_group_staging_buffers(group);
            return ooverlap::comm::api::cuda_to_status(err);
        }

        slot.kind = oo_staging_buffer_kind::owned_host_registered;
#else
        err =
            cudaHostAlloc(
                &host_ptr,
                kOoStagingSlotBytes,
                cudaHostAllocMapped | cudaHostAllocPortable);

        if (err != cudaSuccess) {
            destroy_group_staging_buffers(group);
            return ooverlap::comm::api::cuda_to_status(err);
        }

        std::memset(host_ptr, 0, kOoStagingSlotBytes);

        err =
            cudaHostGetDevicePointer(
                &device_ptr,
                host_ptr,
                0);

        if (err != cudaSuccess) {
            cudaFreeHost(host_ptr);
            destroy_group_staging_buffers(group);
            return ooverlap::comm::api::cuda_to_status(err);
        }

        slot.kind = oo_staging_buffer_kind::owned_host_mapped;
#endif

        slot.host_ptr = host_ptr;
        slot.device_ptr = device_ptr;
        slot.bytes = kOoStagingSlotBytes;
        slot.numa_node = numa_node;
    }

    return OO_SUCCESS;
}

oo_status_t enable_group_peer_access_all_to_all(oo_group_t* group) {
    if (group == nullptr || !valid_group_size(group->num_devices)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    for (int src_rank = 0; src_rank < group->num_devices; ++src_rank) {
        const int src_device = group->devices[src_rank];

        if (src_device < 0) {
            return OO_ERROR_INVALID_DEVICE;
        }

        ooverlap::system::runtime::set_device(src_device);

        for (int dst_rank = 0; dst_rank < group->num_devices; ++dst_rank) {
            if (src_rank == dst_rank) {
                continue;
            }

            const int dst_device = group->devices[dst_rank];

            if (dst_device < 0) {
                return OO_ERROR_INVALID_DEVICE;
            }

            bool enabled = false;

            const oo_status_t status =
                ooverlap::system::p2p::enable_peer_access_one_way_status(
                    src_device,
                    dst_device,
                    &enabled);

            if (status != OO_SUCCESS) {
                return status;
            }

            if (!enabled) {
                return OO_ERROR_UNSUPPORTED;
            }
        }
    }

    return OO_SUCCESS;
}

bool topology_is_all_peer_to_peer(
    const ooverlap::topology::Topology& topology,
    int world_size) {
    if (world_size <= 0 ||
        topology.nodes.size() != static_cast<size_t>(world_size)) {
        return false;
    }

    for (int src_rank = 0; src_rank < world_size; ++src_rank) {
        for (int dst_rank = 0; dst_rank < world_size; ++dst_rank) {
            if (src_rank == dst_rank) {
                continue;
            }

            const ooverlap::topology::Link* link =
                topology.link_by_ordinals(
                    src_rank,
                    dst_rank);

            if (link == nullptr ||
                !link->cuda_peer_access_supported ||
                !link->safe_for_direct_copy ||
                !link->safe_for_tma_reduce) {
                return false;
            }
        }
    }

    return true;
}

oo_status_t initialize_group_topology(
    oo_group_t* group,
    bool enable_peer_access_in_discovery,
    bool include_shm_fallback,
    bool run_validation_probes,
    bool run_atomic_probes,
    bool run_tma_probes) {
    if (group == nullptr || !valid_group_size(group->num_devices)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        ooverlap::topology::DiscoverOptions options{};
        options.enable_peer_access = enable_peer_access_in_discovery;
        options.include_shm_fallback = include_shm_fallback;
        options.require_cuda_peer_access = false;
        options.run_validation_probes = run_validation_probes;
        options.run_tma_probes = run_tma_probes;
        options.run_atomic_probes = run_atomic_probes;

        group->topology =
            ooverlap::topology::discover_current_process_topology(
                devices_vector(
                    group->devices,
                    group->num_devices),
                options);

        group->topology_valid = true;
        group->is_all_peer_to_peer =
            topology_is_all_peer_to_peer(
                group->topology,
                group->num_devices);
        return OO_SUCCESS;
    } catch (...) {
        group->topology_valid = false;
        group->is_all_peer_to_peer = false;
        group->topology = ooverlap::topology::Topology{};
        return ooverlap::comm::api::exception_to_status();
    }
}

void register_collective_buffer(oo_buffer_t* buffer) {
    if (buffer == nullptr ||
        buffer->group == nullptr ||
        buffer->owner_rank < 0 ||
        buffer->owner_rank >= kOoMaxLocalDevices) {
        return;
    }

    buffer->group->collective_buffers[buffer->owner_rank] = buffer;
}

void unregister_collective_buffer(oo_buffer_t* buffer) {
    if (buffer == nullptr ||
        buffer->group == nullptr ||
        buffer->owner_rank < 0 ||
        buffer->owner_rank >= kOoMaxLocalDevices) {
        return;
    }

    oo_group_t* group = buffer->group;
    const int rank = buffer->owner_rank;

    if (group->collective_buffers[rank] == buffer) {
        group->collective_buffers[rank] = nullptr;
    }
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
    } else if (slot.kind == oo_ready_signal_kind::owned_host_mapped) {
        if (slot.owned_host_ptr != nullptr) {
            cudaFreeHost(slot.owned_host_ptr);
        }
    }

    slot = oo_ready_signal{};
}

void destroy_group_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    for (int rank = 0; rank < kOoMaxLocalDevices; ++rank) {
        clear_ready_signal(group->ready_signal_slots[rank]);
        clear_ready_signal(group->host_ready_signal_slots[rank]);
    }
}

oo_status_t allocate_same_process_cuda_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    for (int rank = 0; rank < group->num_devices; ++rank) {
        const int device = group->devices[rank];

        if (device < 0) {
            destroy_group_ready_signals(group);
            return OO_ERROR_INVALID_DEVICE;
        }

        ooverlap::system::runtime::set_device(device);

        void* signal = nullptr;

        cudaError_t err =
            cudaMalloc(
                &signal,
                kOoReadySignalBytes);

        if (err != cudaSuccess) {
            destroy_group_ready_signals(group);
            return ooverlap::comm::api::cuda_to_status(err);
        }

        err =
            cudaMemset(
                signal,
                0,
                kOoReadySignalBytes);

        if (err != cudaSuccess) {
            cudaFree(signal);
            destroy_group_ready_signals(group);
            return ooverlap::comm::api::cuda_to_status(err);
        }

        oo_ready_signal& slot =
            group->ready_signal_slots[rank];

        slot.ptr = signal;
        slot.bytes = kOoReadySignalBytes;
        slot.mapped_bytes = kOoReadySignalBytes;
        slot.owner_rank = rank;
        slot.owner_device = device;
        slot.kind = oo_ready_signal_kind::owned_legacy;
        slot.owned_legacy_ptr = signal;

        /*
         * Also allocate the host-mapped channel up front.  This lets the planner
         * choose HostMapped per edge without changing launch setup.
         */
        void* host_signal = nullptr;
        void* host_device_signal = nullptr;

        err =
            cudaHostAlloc(
                &host_signal,
                kOoReadySignalBytes,
                cudaHostAllocMapped | cudaHostAllocPortable);

        if (err != cudaSuccess) {
            destroy_group_ready_signals(group);
            return ooverlap::comm::api::cuda_to_status(err);
        }

        std::memset(host_signal, 0, kOoReadySignalBytes);

        err =
            cudaHostGetDevicePointer(
                &host_device_signal,
                host_signal,
                0);

        if (err != cudaSuccess) {
            cudaFreeHost(host_signal);
            destroy_group_ready_signals(group);
            return ooverlap::comm::api::cuda_to_status(err);
        }

        oo_ready_signal& host_slot =
            group->host_ready_signal_slots[rank];

        host_slot.ptr = host_device_signal;
        host_slot.bytes = kOoReadySignalBytes;
        host_slot.mapped_bytes = kOoReadySignalBytes;
        host_slot.owner_rank = rank;
        host_slot.owner_device = device;
        host_slot.kind = oo_ready_signal_kind::owned_host_mapped;
        host_slot.owned_host_ptr = host_signal;
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

    if (local_device < 0) {
        return OO_ERROR_INVALID_DEVICE;
    }

    void* local_signal = nullptr;

    ooverlap::system::runtime::set_device(local_device);
    ooverlap::system::runtime::check_cuda(
        cudaMalloc(&local_signal, kOoReadySignalBytes),
        "cudaMalloc(ipc ready signal)");
    ooverlap::system::runtime::check_cuda(
        cudaMemset(local_signal, 0, kOoReadySignalBytes),
        "cudaMemset(ipc ready signal)");

    ooverlap::system::legacy_peer_buffer_descriptor local_desc =
        ooverlap::system::export_legacy_peer_buffer(
            local_signal,
            kOoReadySignalBytes,
            local_device,
            kOoReadySignalBytes);

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
            slot.bytes = kOoReadySignalBytes;
            slot.mapped_bytes = kOoReadySignalBytes;
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

    unregister_collective_buffer(buffer);

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
        group->memory_kind = oo_group_memory_kind::same_process_vmm;

        for (int i = 0; i < num_devices; ++i) {
            if (devices[i] < 0) {
                return OO_ERROR_INVALID_DEVICE;
            }

            group->devices[i] = devices[i];
        }

        oo_status_t status =
            initialize_group_topology(
                group.get(),
                true,
                true,
                true,
                true,
                true);

        if (status != OO_SUCCESS) {
            return status;
        }

        status =
            allocate_same_process_cuda_ready_signals(group.get());

        if (status != OO_SUCCESS) {
            return status;
        }


        status =
            allocate_group_staging_buffers(group.get());

        if (status != OO_SUCCESS) {
            destroy_group_ready_signals(group.get());
            return status;
        }

        group->transfer_plan_distribution =
            ooverlap::comm::plan::make_same_process_transfer_plan_distribution_backend();

        *out_group = group.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

oo_status_t oo_group_create_p2p(
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
        group->memory_kind = oo_group_memory_kind::same_process_cuda_p2p;

        for (int i = 0; i < num_devices; ++i) {
            if (devices[i] < 0) {
                return OO_ERROR_INVALID_DEVICE;
            }

            group->devices[i] = devices[i];
        }

        oo_status_t status =
            enable_group_peer_access_all_to_all(group.get());

        if (status != OO_SUCCESS) {
            return status;
        }

        status =
            initialize_group_topology(
                group.get(),
                true,
                true,
                true,
                true,
                true);

        if (status != OO_SUCCESS) {
            return status;
        }

        status =
            allocate_same_process_cuda_ready_signals(group.get());

        if (status != OO_SUCCESS) {
            destroy_group_ready_signals(group.get());
            return status;
        }


        status =
            allocate_group_staging_buffers(group.get());

        if (status != OO_SUCCESS) {
            destroy_group_ready_signals(group.get());
            return status;
        }

        group->transfer_plan_distribution =
            ooverlap::comm::plan::make_same_process_transfer_plan_distribution_backend();

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
        group->memory_kind = oo_group_memory_kind::multiprocess_legacy_ipc;
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

        /*
         * OOVERLAP_IPC_SHARED_PLAN_GROUP_SETUP_PATCH:
         *
         * Setup-only IPC initialization:
         *   - initialize topology once, with probes/staging disabled for now
         *   - allocate/exchange device ready signals once
         *   - map a shared-memory TransferPlan arena and install IPC plan backend
         *
         * User buffer IPC registration/import is intentionally not handled here.
         */
        oo_status_t status =
            initialize_group_topology(
                group.get(),
                false,  // do not enable all-to-all peer access here
                false,  // no IPC SHM/staging fallback yet
                false,  // skip validation probes for setup-only IPC
                false,  // skip atomic probes
                false); // skip TMA probes

        if (status != OO_SUCCESS) {
            return status;
        }

        status =
            allocate_ipc_ready_signals(group.get());

        if (status != OO_SUCCESS) {
            return status;
        }

        group->transfer_plan_distribution =
            ooverlap::comm::plan::
                make_ipc_shared_plan_transfer_plan_distribution_backend(
                    broker_key,
                    local_rank,
                    num_devices);

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

    for (int rank = 0; rank < kOoMaxLocalDevices; ++rank) {
        group->collective_buffers[rank] = nullptr;

        for (int slot = 0; slot < kOoIpcImportCacheEntriesPerRank; ++slot) {
            auto& entry = group->ipc_import_cache[rank][slot];
            entry.buffer.reset();
            entry.key = {};
            entry.valid = false;
            entry.last_used = 0;
        }
    }

    group->ipc_import_cache_clock = 0;

    destroy_group_staging_buffers(group);
    destroy_group_ready_signals(group);
    group->transfer_plan_distribution.reset();
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
        buffer->ipc_cache_token = next_buffer_ipc_cache_token();
        buffer->system_kind = ooverlap::system::peer_buffer_kind::owned_vmm;
        buffer->mapped = mapped;

        register_collective_buffer(buffer.get());

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
    return oo_buffer_wrap_ipc_range(
        node,
        ptr,
        bytes,
        nullptr,
        out_buffer);
}

oo_status_t oo_buffer_wrap_ipc_range(
    oo_node_t* node,
    void* ptr,
    size_t bytes,
    const oo_buffer_ipc_range_t* range,
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

    void* ipc_base_ptr = ptr;
    size_t ipc_base_bytes = bytes;
    size_t ipc_logical_offset_bytes = 0;

    if (range != nullptr) {
        ipc_base_ptr = range->allocation_base_ptr;
        ipc_base_bytes = range->allocation_bytes;
        ipc_logical_offset_bytes = range->logical_offset_bytes;

        if (ipc_base_ptr == nullptr ||
            ipc_base_bytes == 0 ||
            ipc_logical_offset_bytes > ipc_base_bytes ||
            bytes > ipc_base_bytes - ipc_logical_offset_bytes) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        const std::uintptr_t base_addr =
            reinterpret_cast<std::uintptr_t>(ipc_base_ptr);
        const std::uintptr_t logical_addr =
            reinterpret_cast<std::uintptr_t>(ptr);
        const std::uintptr_t expected_addr =
            base_addr + static_cast<std::uintptr_t>(ipc_logical_offset_bytes);

        if (expected_addr < base_addr ||
            expected_addr != logical_addr) {
            return OO_ERROR_INVALID_ARGUMENT;
        }
    }

    try {
        std::unique_ptr<oo_buffer_t> buffer(new oo_buffer_t{});
        buffer->ptr = ptr;
        buffer->bytes = bytes;
        buffer->mapped_bytes = ipc_base_bytes;
        buffer->ipc_base_ptr = ipc_base_ptr;
        buffer->ipc_base_bytes = ipc_base_bytes;
        buffer->ipc_logical_offset_bytes = ipc_logical_offset_bytes;
        buffer->kind = OO_BUFFER_KIND_WRAPPED;
        buffer->group = node->group;
        buffer->owner_rank = node->rank;
        buffer->owner_device = node->device;
        buffer->ipc_cache_token = next_buffer_ipc_cache_token();
        buffer->system_kind = ooverlap::system::peer_buffer_kind::wrapped;

        /*
         * OOVERLAP_IPC_WRAP_NO_PREREGISTER_PATCH:
         *
         * Same-process public collectives expect oo_buffer_wrap() to publish the
         * local rank's current buffer immediately.
         *
         * Multiprocess IPC is different: ensure_ipc_legacy_collective_buffers_registered()
         * uses group->collective_buffers[local_rank] as the previous local key
         * for its symmetric fast-path check.  If wrap() publishes a newly-created
         * IPC buffer here, that new buffer incorrectly appears to be the previous
         * buffer.  Then a new tensor size can reuse stale imported peer buffers
         * from the prior size and fail prepare_collective_launch() with
         * OO_ERROR_INVALID_ARGUMENT.
         *
         * In IPC mode, publish the local buffer only after the key exchange/import
         * path has decided whether peers need to be refreshed.
         */
        if (node->group->bootstrap_kind !=
            oo_group_bootstrap_kind::multiprocess_ipc) {
            register_collective_buffer(buffer.get());
        }

        *out_buffer = buffer.release();
        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}

oo_status_t oo_buffer_register_ipc(
    oo_node_t* node,
    oo_buffer_t* buffer) {
    if (node == nullptr ||
        node->group == nullptr ||
        buffer == nullptr ||
        buffer->ptr == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    return ooverlap::comm::api::register_ipc_collective_buffers(
        node,
        buffer);
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

        for (int rank = 0; rank < group->num_devices; ++rank) {
            ooverlap::system::runtime::set_device(group->devices[rank]);
            ooverlap::system::runtime::check_cuda(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize");
        }

        return OO_SUCCESS;
    } catch (...) {
        return ooverlap::comm::api::exception_to_status();
    }
}
