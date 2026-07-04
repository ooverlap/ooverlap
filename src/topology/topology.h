#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ooverlap {
namespace topology {

/*
 * Keep topology objects as structs, consistent with the comm-side code style.
 *
 * Terminology:
 *
 *   Node:
 *     One CUDA GPU/device in the local process.
 *
 *   Link:
 *     Information for one directed source GPU -> destination GPU relation.
 *
 *   Transport:
 *     A possible implementation route for communication.
 *
 * A GPU pair can have more than one possible transport. For example:
 *
 *   GPU0 -> GPU2:
 *     direct peer access over PCIe/SYS may exist, but native peer atomics may
 *     not exist. In that case DirectPcie is still available for normal loads
 *     and stores, but TMA peer reduction should not be selected. Shm is also a
 *     possible fallback transport.
 */

enum class LinkKind : int {
    Self = 0,
    Nvlink = 1,
    Pcie = 2,
    Shm = 3,
    Unsupported = 4,
    Unknown = 5,
};

enum class TransportKind : int {
    DirectNvlink = 0,
    DirectPcie = 1,
    Shm = 2,
};

enum class AtomicOperationKind : int {
    Signal32 = 0,
    GlobalLoadStore = 1,
    GlobalAtomic32 = 2,
    GlobalAtomic64 = 3,
    GlobalAtomicF32 = 4,
    GlobalAtomicF16 = 5,
    TmaReduceF16 = 6,
    TmaReduceBf16 = 7,
    TmaReduceF32 = 8,
};

struct AtomicCapability {
    bool signal32 = false;

    /*
     * Generic global loads/stores can be used over CUDA peer access.
     * This is not an atomic capability by itself, but it is often the
     * minimum required functionality for a transport.
     */
    bool global_load_store = false;

    /*
     * Native atomics over the transport.
     *
     * For direct GPU peer links this comes from cudaDevP2PAttrNativeAtomicSupported.
     * For SHM this is conservative and based on cudaDevAttrHostNativeAtomicSupported.
     */
    bool global_atomic_32 = false;
    bool global_atomic_64 = false;
    bool global_atomic_f32 = false;
    bool global_atomic_f16 = false;

    /*
     * Async bulk reductions are treated separately. If this is false, do not
     * route TMA reduce-based collectives through this link.
     */
    bool tma_reduce_f16 = false;
    bool tma_reduce_bf16 = false;
    bool tma_reduce_f32 = false;
};

struct TransportInfo {
    TransportKind kind = TransportKind::DirectPcie;
    bool available = false;
    AtomicCapability atomics{};

    /*
     * CUDA gives a relative P2P performance rank, not a bandwidth number.
     * Lower is better. -1 means unknown/not applicable.
     */
    int performance_rank = -1;

    /*
     * Best-effort descriptive bandwidth field.
     *
     * CUDA runtime does not expose measured/effective bandwidth. This field is
     * left 0 unless a future backend fills it from a vendor-specific source or
     * a benchmark.
     */
    double estimated_bandwidth_gbps = 0.0;

    std::string description;
};

struct Node {
    int device = -1;
    int ordinal = -1;

    int pci_domain_id = -1;
    int pci_bus_id = -1;
    int pci_device_id = -1;

    int numa_node = -1;

    int compute_major = 0;
    int compute_minor = 0;

    std::string name;
    std::string pci_bus_id_string;
};

struct Link {
    int src_ordinal = -1;
    int dst_ordinal = -1;
    int src_device = -1;
    int dst_device = -1;

    LinkKind preferred_kind = LinkKind::Unknown;

    bool cuda_peer_access_supported = false;
    bool cuda_peer_access_enabled = false;
    bool cuda_array_peer_access_supported = false;

    /*
     * Directed P2P attributes from CUDA.
     */
    bool native_atomic_supported = false;
    int performance_rank = -1;

    /*
     * True if this link is safe for TMA peer reduction / remote reductions.
     * This is intentionally stricter than cuda_peer_access_supported.
     */
    bool safe_for_tma_reduce = false;

    /*
     * True if direct GPU loads/stores are usable. This can be true even when
     * safe_for_tma_reduce is false.
     */
    bool safe_for_direct_copy = false;

    std::vector<TransportInfo> transports;
};

struct Topology {
    std::vector<Node> nodes;
    std::vector<Link> links;

    const Node* node_by_ordinal(int ordinal) const;
    const Link* link_by_ordinals(int src_ordinal, int dst_ordinal) const;
};

/*
 * Options for discovery.
 */
struct DiscoverOptions {
    /*
     * If true, call cudaDeviceEnablePeerAccess for every directed accessible
     * peer relation. This is useful when the topology object will be used to
     * immediately run kernels over external cudaMalloc/PyTorch buffers.
     */
    bool enable_peer_access = true;

    /*
     * If true, include SHM as an available fallback transport for all non-self
     * GPU pairs. This does not allocate any SHM buffer. It only records that
     * a future planner may allocate pinned/mapped host memory or POSIX shm.
     */
    bool include_shm_fallback = true;

    /*
     * If true, return failure if any directed non-self GPU pair has no CUDA P2P
     * access. If false, the direct transport is marked unavailable and SHM can
     * still be reported as fallback.
     */
    bool require_cuda_peer_access = false;
};

const char* link_kind_name(LinkKind kind);
const char* transport_kind_name(TransportKind kind);

bool link_supports_atomic_operation(
    const Link& link,
    TransportKind transport,
    AtomicOperationKind op);

Topology discover_current_process_topology(
    const std::vector<int>& devices,
    const DiscoverOptions& options = DiscoverOptions{});

Topology discover_all_cuda_devices_topology(
    const DiscoverOptions& options = DiscoverOptions{});

std::string topology_to_string(const Topology& topology);
std::string topology_to_json(const Topology& topology);

} // namespace topology
} // namespace ooverlap
