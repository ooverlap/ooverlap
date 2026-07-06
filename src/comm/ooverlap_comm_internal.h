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
     *
     * Buffer allocation and cleanup are VMM-backed. Transport choice must still
     * come from topology/transfer planning, not from this enum.
     */
    same_process_vmm = 0,

    /*
     * Same-process normal CUDA allocations or external wrapped pointers.
     *
     * This is the external-buffer path. Peer access enabling is a setup detail;
     * whether a collective can use direct NVLink, direct PCIe/SYS, SHM, or a
     * fallback route is represented by topology.
     */
    same_process_cuda_p2p = 1,

    /*
     * Multiprocess legacy CUDA IPC.
     *
     * Peer memory views are imported/opened by the launch-exchange backend, not
     * passed through public collective APIs.
     */
    multiprocess_legacy_ipc = 2
};

enum class oo_group_bootstrap_kind {
    same_process = 0,
    multiprocess_ipc = 1
};

enum class oo_ready_signal_kind {
    empty = 0,

    /*
     * Same-process VMM allocation visible to all local devices.
     */
    owned_vmm = 1,

    /*
     * cudaMalloc allocation owned by this process/rank. Same-process groups may
     * use this directly; IPC groups export it through legacy CUDA IPC.
     */
    owned_legacy = 2,

    /*
     * cudaIpcOpenMemHandle mapping imported from another process/rank.
     */
    imported_legacy = 3,

    /*
     * VMM FD imported mapping. Kept for future extension.
     */
    imported_vmm = 4
};

struct oo_ready_signal {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t mapped_bytes = 0;

    int owner_rank = -1;
    int owner_device = -1;

    oo_ready_signal_kind kind = oo_ready_signal_kind::empty;

    /*
     * Valid for kind == owned_vmm.
     */
    ooverlap::system::mapped_peer_buffer owned_vmm{};

    /*
     * Valid for kind == owned_legacy.
     */
    void* owned_legacy_ptr = nullptr;

    /*
     * Valid for kind == imported_legacy/imported_vmm.
     */
    ooverlap::system::imported_peer_buffer imported{};
};

namespace ooverlap {
namespace comm {

namespace api {
class CollectiveLaunchExchangeBackend;
}

namespace plan {
class TransferPlanDistributionBackend;
}

} // namespace comm
} // namespace ooverlap

struct oo_group {
    int num_devices = 0;
    int devices[kOoMaxLocalDevices] = {};

    oo_group_bootstrap_kind bootstrap_kind =
        oo_group_bootstrap_kind::same_process;

    oo_group_memory_kind memory_kind =
        oo_group_memory_kind::same_process_vmm;

    /*
     * Meaningful only for multiprocess_ipc.
     *
     * In IPC mode, each OS process owns exactly one rank/node. That rank owns
     * its local user buffer and ready signal; the launch-exchange backend is
     * responsible for exchanging/importing the rank memory views needed by a
     * collective epoch.
     */
    int local_rank = -1;
    int local_world_size = 0;

    std::unique_ptr<ooverlap::system::Broker> broker{};

    /*
     * Ready-signal state indexed by logical rank.
     *
     * ready_signal_slots[r] owns or imports the cleanup state for rank r's
     * ready signal. Do not keep a second mirror array; all code should use this
     * field.
     */
    oo_ready_signal ready_signal_slots[kOoMaxLocalDevices] = {};

    /*
     * Topology is the source of truth for transport capability.
     *
     * Do not store separate peer_access_enabled matrices in oo_group. Peer
     * access enabling is setup side-effect; transport choice belongs in
     * topology + logical transfer planning.
     */
    bool topology_valid = false;
    ooverlap::topology::Topology topology{};

    /*
     * Group-owned collective launch exchange.
     *
     * Public collectives should pass only the local rank buffer. This backend
     * converts that local contribution into a rank-indexed CollectiveLaunchState
     * containing the memory views and ready signals needed by lowering.
     *
     * Same-process implementation:
     *   all ranks contribute local oo_buffer_t for the active collective epoch;
     *   the backend waits until every rank arrived, then returns rank views.
     *
     * IPC implementation later:
     *   ranks exchange/import descriptors through broker/IPC, then return rank
     *   views. Public collective signatures do not change.
     */
    std::unique_ptr<ooverlap::comm::api::CollectiveLaunchExchangeBackend>
        collective_launch_exchange{};

    /*
     * Group-owned logical TransferPlan distribution.
     *
     * Same-process implementation:
     *   first arriving rank builds the logical plan; others wait and copy it.
     *
     * IPC implementation later:
     *   rank 0 builds/broadcasts the logical plan.
     */
    std::unique_ptr<ooverlap::comm::plan::TransferPlanDistributionBackend>
        transfer_plan_distribution{};
};

struct oo_node {
    oo_group_t* group = nullptr;
    int rank = -1;
    int device = -1;

    /*
     * Monotonic per-node collective sequence.
     *
     * Rank-local calls must be issued in matching order across ranks, same as
     * NCCL. The launch-exchange backend should use this to detect ordering
     * mismatches.
     */
    int collective_epoch = 0;
};

struct oo_buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t mapped_bytes = 0;

    /*
     * Public coarse kind.
     *
     * Imported buffers may still report as WRAPPED publicly until the public ABI
     * grows more detailed buffer kinds. Internally, system_kind is the precise
     * provenance.
     */
    oo_buffer_kind_t kind = OO_BUFFER_KIND_WRAPPED;

    /*
     * Internal validation/debug metadata.
     */
    oo_group_t* group = nullptr;
    int owner_rank = -1;
    int owner_device = -1;

    /*
     * Internal precise provenance.
     */
    ooverlap::system::peer_buffer_kind system_kind =
        ooverlap::system::peer_buffer_kind::empty;

    /*
     * Valid only for public kind == OO_BUFFER_KIND_VMM and
     * system_kind == owned_vmm.
     */
    ooverlap::system::mapped_peer_buffer mapped{};

    /*
     * Valid for system_kind == imported_legacy/imported_vmm.
     */
    ooverlap::system::imported_peer_buffer imported{};
};

/*
 * C++ helper APIs for the IPC path.
 *
 * These are intentionally in the internal header because they use C++
 * descriptor types from peer_buffer.cuh. Add C ABI wrappers later only if
 * external users/Python bindings need to call these directly.
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
