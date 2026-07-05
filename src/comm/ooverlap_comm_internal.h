#pragma once

#include "ooverlap/comm.h"
#include "ooverlap/system/peer_buffer.cuh"
#include "topology/topology.h"

/*
 * Include <type_traits> before broker.cuh because broker.cuh uses
 * std::is_trivially_copyable in its exchange_pod template.
 */
#include <type_traits>
#include "ooverlap/system/broker.cuh"

#include <cstddef>
#include <memory>

constexpr int kOoMaxLocalDevices = 16;

enum class oo_group_memory_kind {
    /*
     * Same-process VMM allocations with cuMemSetAccess.
     * This is the current oo_group_create() behavior.
     */
    same_process_vmm = 0,

    /*
     * Same-process normal CUDA allocations / external wrapped pointers.
     * Peer visibility comes from cudaDeviceEnablePeerAccess.
     */
    same_process_cuda_p2p = 1,

    /*
     * Multiprocess legacy CUDA IPC.
     * Peer buffers are imported with cudaIpcOpenMemHandle.
     */
    multiprocess_legacy_ipc = 2
};

enum class oo_group_bootstrap_kind {
    same_process = 0,
    multiprocess_ipc = 1
};

enum class oo_ready_signal_kind {
    empty = 0,

    // Same-process VMM allocation visible to all local devices.
    owned_vmm = 1,

    // cudaMalloc allocation owned by this process/rank, exported by legacy IPC.
    owned_legacy = 2,

    // cudaIpcOpenMemHandle mapping imported from another process/rank.
    imported_legacy = 3,

    // VMM FD imported mapping. Kept for future extension.
    imported_vmm = 4
};

struct oo_ready_signal {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t mapped_bytes = 0;

    int owner_rank = -1;
    int owner_device = -1;

    oo_ready_signal_kind kind = oo_ready_signal_kind::empty;

    // Valid for kind == owned_vmm.
    ooverlap::system::mapped_peer_buffer owned_vmm{};

    // Valid for kind == owned_legacy.
    void* owned_legacy_ptr = nullptr;

    // Valid for kind == imported_legacy/imported_vmm.
    ooverlap::system::imported_peer_buffer imported{};
};

struct oo_group {
    int num_devices = 0;
    int devices[kOoMaxLocalDevices] = {};

    oo_group_bootstrap_kind bootstrap_kind = oo_group_bootstrap_kind::same_process;

    /*
     * Meaningful only for multiprocess_ipc.
     *
     * In that mode, each OS process owns exactly one rank/node. That rank
     * allocates its local ready signal, exports it, then imports peers' ready
     * signals through Broker + CUDA IPC.
     */
    int local_rank = -1;
    int local_world_size = 0;

    std::unique_ptr<ooverlap::system::Broker> broker{};

    /*
     * New first-class ready-signal state.
     *
     * ready_signal_slots[r] knows whether the pointer is owned in this process
     * or imported from another process, and therefore how to clean it up.
     */
    oo_ready_signal ready_signal_slots[kOoMaxLocalDevices] = {};

    /*
     * Compatibility mirror for older same-process code/tests that may inspect
     * group->ready_signals directly. Do not free through this array anymore;
     * free through ready_signal_slots.
     */
    ooverlap::system::mapped_peer_buffer ready_signals[kOoMaxLocalDevices] = {};

    oo_group_memory_kind memory_kind =
                oo_group_memory_kind::same_process_vmm;
    
    /*
     * Meaningful for same_process_cuda_p2p.
     *
     * peer_access_enabled[src_rank][dst_rank] means the CUDA context for
     * devices[src_rank] enabled access to allocations owned by devices[dst_rank].
     */
    bool peer_access_enabled[kOoMaxLocalDevices][kOoMaxLocalDevices] = {};

    bool topology_valid = false;
    ooverlap::topology::Topology topology{};
};

struct oo_node {
    oo_group_t* group = nullptr;
    int rank = -1;
    int device = -1;

    // Monotonic per-node collective sequence.
    // Rank-local calls must be issued in matching order, same as NCCL.
    int collective_epoch = 0;
};

struct oo_buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t mapped_bytes = 0;

    /*
     * Public coarse kind. Until include/ooverlap/comm.h grows more enum values,
     * imported buffers report as WRAPPED publicly but are distinguished by
     * system_kind/imported internally.
     */
    oo_buffer_kind_t kind = OO_BUFFER_KIND_WRAPPED;

    // Internal validation/debug metadata.
    oo_group_t* group = nullptr;
    int owner_rank = -1;
    int owner_device = -1;

    /*
     * Internal, precise provenance.
     *
     * This is the important change: imported CUDA IPC mappings are no longer
     * indistinguishable from plain wrapped pointers.
     */
    ooverlap::system::peer_buffer_kind system_kind =
        ooverlap::system::peer_buffer_kind::empty;

    // Valid only for public kind == OO_BUFFER_KIND_VMM and system_kind == owned_vmm.
    ooverlap::system::mapped_peer_buffer mapped{};

    // Valid for system_kind == imported_legacy/imported_vmm.
    ooverlap::system::imported_peer_buffer imported{};
};

/*
 * Optional C ABI extension for node-local multiprocess bootstrap.
 *
 * You should add this declaration to include/ooverlap/comm.h too if you want
 * external users/Python bindings to call it directly.
 */
extern "C" oo_status_t oo_group_create_ipc(
    const int* devices,
    int num_devices,
    int local_rank,
    const char* broker_key,
    oo_group_t** out_group);

/*
 * C++ helper APIs for the IPC path.
 *
 * These are intentionally in the internal header because they use C++ descriptor
 * types from peer_buffer.cuh. Add C ABI wrappers later if needed.
 */
oo_status_t oo_buffer_export_legacy_descriptor(
    oo_buffer_t* buffer,
    ooverlap::system::legacy_peer_buffer_descriptor* out_desc);

oo_status_t oo_buffer_import_legacy_descriptor(
    oo_node_t* node,
    const ooverlap::system::legacy_peer_buffer_descriptor& desc,
    oo_buffer_t** out_buffer);

oo_status_t oo_buffer_adopt_imported_peer_buffer(
    oo_node_t* node,
    ooverlap::system::imported_peer_buffer&& imported,
    oo_buffer_t** out_buffer);
