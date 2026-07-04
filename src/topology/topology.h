#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ooverlap {
namespace topology {

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
    TmaLoadF16 = 6,
    TmaStoreF16 = 7,
    TmaReduceF16 = 8,
    TmaReduceBf16 = 9,
    TmaReduceF32 = 10,
};

struct ProbeResult {
    bool attempted = false;
    bool passed = false;
    std::string error;
};

struct AtomicCapability {
    bool signal32 = false;
    bool global_load_store = false;

    bool global_atomic_32 = false;
    bool global_atomic_64 = false;
    bool global_atomic_f32 = false;
    bool global_atomic_f16 = false;

    bool tma_load_f16 = false;
    bool tma_store_f16 = false;
    bool tma_reduce_f16 = false;
    bool tma_reduce_bf16 = false;
    bool tma_reduce_f32 = false;
};

struct TransportInfo {
    TransportKind kind = TransportKind::DirectPcie;
    bool available = false;
    AtomicCapability atomics{};

    int performance_rank = -1;
    double estimated_bandwidth_gbps = 0.0;

    ProbeResult copy_probe{};
    ProbeResult atomic32_probe{};
    ProbeResult tma_load_f16_probe{};
    ProbeResult tma_store_f16_probe{};
    ProbeResult tma_reduce_f16_probe{};

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

    bool native_atomic_supported = false;
    bool partial_native_atomic_supported = false;
    int performance_rank = -1;

    bool safe_for_tma_reduce = false;
    bool safe_for_direct_copy = false;

    ProbeResult direct_copy_probe{};
    ProbeResult direct_atomic32_probe{};
    ProbeResult direct_tma_load_f16_probe{};
    ProbeResult direct_tma_store_f16_probe{};
    ProbeResult direct_tma_reduce_f16_probe{};

    ProbeResult shm_copy_probe{};
    ProbeResult shm_atomic32_probe{};
    ProbeResult shm_tma_load_f16_probe{};
    ProbeResult shm_tma_store_f16_probe{};
    ProbeResult shm_tma_reduce_f16_probe{};

    std::vector<TransportInfo> transports;
};

struct Topology {
    std::vector<Node> nodes;
    std::vector<Link> links;

    const Node* node_by_ordinal(int ordinal) const;
    const Link* link_by_ordinals(int src_ordinal, int dst_ordinal) const;
};

struct DiscoverOptions {
    bool enable_peer_access = true;
    bool include_shm_fallback = true;
    bool require_cuda_peer_access = false;

    bool run_validation_probes = true;
    bool run_tma_probes = true;
    bool run_atomic_probes = true;
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
