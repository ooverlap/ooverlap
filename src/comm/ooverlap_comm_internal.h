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

enum class oo_ready_signal_channel : int {
    device_memory = 0,
    host_mapped = 1,
};

constexpr int kOoReadySignalChannelCount = 2;
constexpr int kOoReadySignalChannelDeviceMemory =
    static_cast<int>(oo_ready_signal_channel::device_memory);
constexpr int kOoReadySignalChannelHostMapped =
    static_cast<int>(oo_ready_signal_channel::host_mapped);

/*
 * Fixed staging-buffer configuration.
 *
 * First implementation: preallocate a small fixed number of large host-mapped
 * staging slots at group creation time.  Planners can later reference these by
 * LogicalBufferRole::ShmStaging + staging_slot.
 *
 * Tune by changing these constants and recompiling.
 */
constexpr int kOoMaxStagingSlots = 4;
constexpr size_t kOoStagingSlotBytes = 256ull * 1024ull * 1024ull;

enum class oo_group_memory_kind {
    /*
     * Same-process VMM allocations with cuMemSetAccess.
     *
     * Buffer allocation/cleanup is VMM-backed. Transport choice still comes
     * from topology + logical transfer planning.
     */
    same_process_vmm = 0,

    /*
     * Same-process normal CUDA allocations or external wrapped pointers.
     *
     * Peer access enabling is only a setup side-effect for external pointers.
     * Do not store peer-access state here; topology is the source of truth for
     * transport planning.
     */
    same_process_cuda_p2p = 1,

    /*
     * Multiprocess legacy CUDA IPC.
     *
     * IPC peer memory views should be registered/imported into group-owned
     * buffer state. Public collectives should still only take this rank's local
     * buffer.
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
    imported_vmm = 4,

    /*
     * Mapped pinned host allocation. ptr is the device pointer returned by
     * cudaHostGetDevicePointer(); owned_host_ptr is the host pointer.
     */
    owned_host_mapped = 5
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

    void* owned_host_ptr = nullptr;

    /*
     * Valid for kind == imported_legacy/imported_vmm.
     */
    ooverlap::system::imported_peer_buffer imported{};
};

enum class oo_staging_buffer_kind {
    empty = 0,

    /*
     * Host virtual allocation, best-effort NUMA placement, then
     * cudaHostRegister(...Mapped...) so every local GPU gets a device-visible
     * pointer for the same backing pages.
     */
    owned_host_registered = 1,

    /*
     * Fallback path from cudaHostAllocMapped when mmap/cudaHostRegister is not
     * used or not available.
     */
    owned_host_mapped = 2,
};

struct oo_staging_buffer {
    /*
     * CPU pointer used for lifetime management and optional host initialization.
     */
    void* host_ptr = nullptr;

    /*
     * Device-visible pointer returned by cudaHostGetDevicePointer().  This is
     * what RankPointerBinding::shm_staging[slot] should receive later.
     */
    void* device_ptr = nullptr;

    size_t bytes = 0;

    /*
     * Requested NUMA node.  -1 means no NUMA binding was requested/available.
     * Placement is best-effort and allocation is still valid if binding fails.
     */
    int numa_node = -1;

    oo_staging_buffer_kind kind = oo_staging_buffer_kind::empty;
};

namespace ooverlap {
namespace comm {
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
     * its local user buffer and ready signal. IPC buffer registration/import
     * will later populate collective_buffers[] with this process's pointer view
     * for each logical rank.
     */
    int local_rank = -1;
    int local_world_size = 0;

    std::unique_ptr<ooverlap::system::Broker> broker{};

    /*
     * Ready-signal state indexed by logical rank.
     *
     * ready_signal_slots[r] owns or imports the cleanup state for rank r's ready
     * signal. All code should use this field directly.
     */
    oo_ready_signal ready_signal_slots[kOoMaxLocalDevices] = {};

    /* Additional channel: mapped pinned host ready signals. */
    oo_ready_signal host_ready_signal_slots[kOoMaxLocalDevices] = {};

    /*
     * Fixed group-owned staging slots.
     *
     * These are intentionally independent from user tensor buffers.  The planner
     * will later reference them by staging_slot index; lowering will bind that
     * index to the device_ptr stored here.
     */
    oo_staging_buffer staging_slots[kOoMaxStagingSlots] = {};

    /*
     * Current rank-buffer registry for public collectives.
     *
     * Public collective APIs pass only this rank's local buffer. The launch
     * helper builds peer pointer views from this registry instead of requiring
     * peers[]/peer_count from the caller.
     *
     * Same-process behavior:
     *   oo_buffer_alloc()/oo_buffer_wrap() register the returned buffer at
     *   collective_buffers[node->rank]. prepare_collective_launch() reads all
     *   ranks from this array and returns immediately; GPU ready signals do the
     *   actual rendezvous.
     *
     * Limitation:
     *   This is intentionally one current collective buffer per rank. If we need
     *   multiple live tensors per group later, replace this with an explicit
     *   oo_buffer_set_t.
     */
    oo_buffer_t* collective_buffers[kOoMaxLocalDevices] = {};

    /*
     * OOVERLAP_IPC_LEGACY_BUFFER_IMPORT_STORAGE_PATCH:
     *
     * Group-owned imported peer buffers for multiprocess legacy CUDA IPC.
     *
     * collective_buffers[] stays the non-owning rank -> current buffer view.
     * For IPC peers, those views point into this owning storage.  The local
     * rank still points at the caller's local oo_buffer_t.
     *
     * First implementation deliberately does not cache across collectives:
     * every IPC collective re-exchanges legacy CUDA IPC descriptors and refreshes
     * these imports, because the registered tensor may be different each call.
     */
    std::unique_ptr<oo_buffer_t>
        ipc_imported_collective_buffers[kOoMaxLocalDevices] = {};

    /*
     * Topology is the source of truth for transport capability.
     *
     * Do not store a separate peer_access_enabled matrix in oo_group. Peer
     * access enabling is setup side-effect; transport choice belongs in topology
     * + logical transfer planning.
     */
    bool topology_valid = false;
    ooverlap::topology::Topology topology{};

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
     * NCCL. prepare_collective_launch() increments this and passes the epoch to
     * GPU-side ready-signal rendezvous.
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
