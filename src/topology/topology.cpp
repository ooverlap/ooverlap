#include "topology/topology.h"
#include "topology/topology_probe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define TOPO_TRACE(fmt, ...)                                                   \
    do {                                                                       \
        std::fprintf(                                                          \
            stderr,                                                            \
            "[topo] %s:%d " fmt "\n",                                          \
            __func__,                                                          \
            __LINE__,                                                          \
            ##__VA_ARGS__);                                                    \
        std::fflush(stderr);                                                   \
    } while (0)

namespace ooverlap {
namespace topology {
namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

int get_device_attr_or_default(
    int attr,
    int device,
    int default_value = 0) {
    int value = default_value;

    cudaError_t err =
        cudaDeviceGetAttribute(
            &value,
            static_cast<cudaDeviceAttr>(attr),
            device);

    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return default_value;
    }

    return value;
}

int get_p2p_attr_or_default(
    cudaDeviceP2PAttr attr,
    int src_device,
    int dst_device,
    int default_value = 0) {
    int value = default_value;

    cudaError_t err =
        cudaDeviceGetP2PAttribute(
            &value,
            attr,
            src_device,
            dst_device);

    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return default_value;
    }

    return value;
}

bool enable_peer_access_one_way(
    int src_device,
    int dst_device) {
    if (src_device == dst_device) {
        return true;
    }

    int can_access = 0;

    check_cuda(
        cudaDeviceCanAccessPeer(
            &can_access,
            src_device,
            dst_device),
        "cudaDeviceCanAccessPeer");

    if (!can_access) {
        return false;
    }

    check_cuda(
        cudaSetDevice(src_device),
        "cudaSetDevice(src_device)");

    cudaError_t err =
        cudaDeviceEnablePeerAccess(
            dst_device,
            0);

    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
        return true;
    }

    check_cuda(err, "cudaDeviceEnablePeerAccess");
    return true;
}

int query_numa_node_from_pci(
    int pci_domain_id,
    int pci_bus_id,
    int pci_device_id) {
#if defined(__linux__)
    char path[256];

    std::snprintf(
        path,
        sizeof(path),
        "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node",
        pci_domain_id,
        pci_bus_id,
        pci_device_id);

    FILE* f = std::fopen(path, "r");
    if (f == nullptr) {
        return -1;
    }

    int numa = -1;
    const int scanned =
        std::fscanf(f, "%d", &numa);

    std::fclose(f);

    if (scanned != 1) {
        return -1;
    }

    return numa;
#else
    (void)pci_domain_id;
    (void)pci_bus_id;
    (void)pci_device_id;
    return -1;
#endif
}

std::string escape_json(const std::string& value) {
    std::ostringstream out;

    for (char c : value) {
        switch (c) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << c;
                break;
        }
    }

    return out.str();
}

const char* bool_text(bool value) {
    return value ? "true" : "false";
}

ProbeResult not_attempted_result() {
    return ProbeResult{};
}

AtomicCapability direct_atomic_capability_from_results(
    bool direct_copy_ok,
    bool native_atomic_supported,
    bool partial_native_atomic_supported,
    const ProbeResult& atomic32_probe,
    const ProbeResult& tma_load_probe,
    const ProbeResult& tma_store_probe,
    const ProbeResult& tma_reduce_probe) {
    AtomicCapability caps{};

    caps.signal32 = direct_copy_ok;
    caps.global_load_store = direct_copy_ok;

    /*
     * Attribute says all native atomics, or partial native atomics. The actual
     * atomic32 probe decides the 32-bit flag when it was attempted.
     */
    caps.global_atomic_32 =
        atomic32_probe.attempted
            ? atomic32_probe.passed
            : (direct_copy_ok &&
               (native_atomic_supported || partial_native_atomic_supported));

    caps.global_atomic_64 =
        direct_copy_ok && native_atomic_supported;

    caps.global_atomic_f32 =
        direct_copy_ok && native_atomic_supported;

    caps.global_atomic_f16 = false;

    caps.tma_load_f16 =
        tma_load_probe.attempted && tma_load_probe.passed;

    caps.tma_store_f16 =
        tma_store_probe.attempted && tma_store_probe.passed;

    caps.tma_reduce_f16 =
        tma_reduce_probe.attempted && tma_reduce_probe.passed;

    /*
     * These are not probed in this first pass. Keep false until a BF16/F32
     * probe is added.
     */
    caps.tma_reduce_bf16 = false;
    caps.tma_reduce_f32 = false;

    return caps;
}

AtomicCapability shm_atomic_capability_from_results(
    const ProbeResult& copy_probe,
    const ProbeResult& atomic32_probe,
    const ProbeResult& tma_load_probe,
    const ProbeResult& tma_store_probe,
    const ProbeResult& tma_reduce_probe) {
    AtomicCapability caps{};

    const bool copy_ok =
        copy_probe.attempted && copy_probe.passed;

    caps.signal32 = copy_ok;
    caps.global_load_store = copy_ok;
    caps.global_atomic_32 =
        atomic32_probe.attempted && atomic32_probe.passed;
    caps.global_atomic_64 = false;
    caps.global_atomic_f32 = false;
    caps.global_atomic_f16 = false;

    caps.tma_load_f16 =
        tma_load_probe.attempted && tma_load_probe.passed;
    caps.tma_store_f16 =
        tma_store_probe.attempted && tma_store_probe.passed;
    caps.tma_reduce_f16 =
        tma_reduce_probe.attempted && tma_reduce_probe.passed;

    caps.tma_reduce_bf16 = false;
    caps.tma_reduce_f32 = false;

    return caps;
}

LinkKind infer_direct_link_kind(
    bool peer_access_supported,
    bool native_atomic_supported,
    int performance_rank) {
    if (!peer_access_supported) {
        return LinkKind::Unsupported;
    }

    if (native_atomic_supported && performance_rank == 0) {
        return LinkKind::Nvlink;
    }

    return LinkKind::Pcie;
}

Node discover_node(
    int ordinal,
    int device) {
    TOPO_TRACE("enter discover_node ordinal=%d device=%d", ordinal, device);

    cudaDeviceProp prop{};

    TOPO_TRACE("before cudaGetDeviceProperties device=%d", device);
    check_cuda(
        cudaGetDeviceProperties(
            &prop,
            device),
        "cudaGetDeviceProperties");
    TOPO_TRACE("after cudaGetDeviceProperties device=%d name=%s", device, prop.name);

    Node node{};
    node.ordinal = ordinal;
    node.device = device;
    node.name = prop.name;
    node.compute_major = prop.major;
    node.compute_minor = prop.minor;
    node.pci_domain_id = prop.pciDomainID;
    node.pci_bus_id = prop.pciBusID;
    node.pci_device_id = prop.pciDeviceID;

    char pci_bus_id[64] = {};

    TOPO_TRACE("before cudaDeviceGetPCIBusId device=%d", device);
    cudaError_t bus_err =
        cudaDeviceGetPCIBusId(
            pci_bus_id,
            static_cast<int>(sizeof(pci_bus_id)),
            device);
    TOPO_TRACE("after cudaDeviceGetPCIBusId device=%d err=%d", device, int(bus_err));

    if (bus_err == cudaSuccess) {
        node.pci_bus_id_string = pci_bus_id;
    } else {
        (void)cudaGetLastError();
    }

    TOPO_TRACE("before query_numa_node_from_pci device=%d", device);
    node.numa_node =
        query_numa_node_from_pci(
            node.pci_domain_id,
            node.pci_bus_id,
            node.pci_device_id);
    TOPO_TRACE("after query_numa_node_from_pci device=%d numa=%d", device, node.numa_node);

    TOPO_TRACE("leave discover_node ordinal=%d device=%d", ordinal, device);
    return node;
}

void run_direct_probes(
    Link* link,
    const DiscoverOptions& options) {
    if (link == nullptr ||
        !options.run_validation_probes ||
        !link->cuda_peer_access_supported ||
        (options.enable_peer_access && !link->cuda_peer_access_enabled)) {
        return;
    }

    link->direct_copy_probe =
        detail::probe_direct_load_store(
            link->src_device,
            link->dst_device);

    if (options.run_atomic_probes) {
        link->direct_atomic32_probe =
            detail::probe_direct_atomic_add_i32(
                link->src_device,
                link->dst_device);
    }

    if (options.run_tma_probes) {
        link->direct_tma_load_f16_probe =
            detail::probe_direct_tma_load_f16(
                link->src_device,
                link->dst_device);

        link->direct_tma_store_f16_probe =
            detail::probe_direct_tma_store_f16(
                link->src_device,
                link->dst_device);

        link->direct_tma_reduce_f16_probe =
            detail::probe_direct_tma_reduce_f16(
                link->src_device,
                link->dst_device);
    }
}

void run_shm_probes(
    Link* link,
    const DiscoverOptions& options) {
    if (link == nullptr ||
        !options.include_shm_fallback ||
        !options.run_validation_probes) {
        return;
    }

    link->shm_copy_probe =
        detail::probe_shm_load_store(
            link->src_device,
            link->dst_device);

    if (options.run_atomic_probes) {
        link->shm_atomic32_probe =
            detail::probe_shm_atomic_add_i32(
                link->src_device,
                link->dst_device);
    }

    if (options.run_tma_probes) {
        link->shm_tma_load_f16_probe =
            detail::probe_shm_tma_load_f16(
                link->src_device,
                link->dst_device);

        link->shm_tma_store_f16_probe =
            detail::probe_shm_tma_store_f16(
                link->src_device,
                link->dst_device);

        link->shm_tma_reduce_f16_probe =
            detail::probe_shm_tma_reduce_f16(
                link->src_device,
                link->dst_device);
    }
}

Link discover_link(
    int src_ordinal,
    int dst_ordinal,
    int src_device,
    int dst_device,
    const DiscoverOptions& options) {
    Link link{};
    link.src_ordinal = src_ordinal;
    link.dst_ordinal = dst_ordinal;
    link.src_device = src_device;
    link.dst_device = dst_device;

    if (src_device == dst_device) {
        link.preferred_kind = LinkKind::Self;
        link.cuda_peer_access_supported = true;
        link.cuda_peer_access_enabled = true;
        link.cuda_array_peer_access_supported = true;
        link.native_atomic_supported = true;
        link.partial_native_atomic_supported = true;
        link.performance_rank = 0;
        link.safe_for_tma_reduce = true;
        link.safe_for_direct_copy = true;
        return link;
    }

    int can_access = 0;

    check_cuda(
        cudaDeviceCanAccessPeer(
            &can_access,
            src_device,
            dst_device),
        "cudaDeviceCanAccessPeer");

    link.cuda_peer_access_supported = (can_access != 0);

    if (!link.cuda_peer_access_supported) {
        link.preferred_kind = LinkKind::Unsupported;

        if (options.require_cuda_peer_access) {
            std::ostringstream oss;
            oss << "CUDA peer access is not supported for GPU"
                << src_device
                << " -> GPU"
                << dst_device;
            throw std::runtime_error(oss.str());
        }
    }

    if (link.cuda_peer_access_supported) {
        link.native_atomic_supported =
            get_p2p_attr_or_default(
                cudaDevP2PAttrNativeAtomicSupported,
                src_device,
                dst_device,
                0) != 0;

#if defined(cudaDevP2PAttrOnlyPartialNativeAtomicSupported)
        link.partial_native_atomic_supported =
            get_p2p_attr_or_default(
                cudaDevP2PAttrOnlyPartialNativeAtomicSupported,
                src_device,
                dst_device,
                0) != 0;
#endif

        link.performance_rank =
            get_p2p_attr_or_default(
                cudaDevP2PAttrPerformanceRank,
                src_device,
                dst_device,
                -1);

        link.cuda_array_peer_access_supported =
            get_p2p_attr_or_default(
                cudaDevP2PAttrCudaArrayAccessSupported,
                src_device,
                dst_device,
                0) != 0;

        if (options.enable_peer_access) {
            link.cuda_peer_access_enabled =
                enable_peer_access_one_way(
                    src_device,
                    dst_device);
        }

        run_direct_probes(&link, options);

        const bool direct_copy_ok =
            options.run_validation_probes
                ? (link.direct_copy_probe.attempted &&
                   link.direct_copy_probe.passed)
                : (link.cuda_peer_access_supported &&
                   (options.enable_peer_access
                        ? link.cuda_peer_access_enabled
                        : true));

        link.safe_for_direct_copy = direct_copy_ok;

        link.safe_for_tma_reduce =
            link.direct_tma_reduce_f16_probe.attempted
                ? link.direct_tma_reduce_f16_probe.passed
                : (direct_copy_ok && link.native_atomic_supported);

        link.preferred_kind =
            infer_direct_link_kind(
                link.cuda_peer_access_supported,
                link.native_atomic_supported,
                link.performance_rank);

        TransportInfo direct{};
        direct.kind =
            link.preferred_kind == LinkKind::Nvlink
                ? TransportKind::DirectNvlink
                : TransportKind::DirectPcie;
        direct.available = direct_copy_ok;
        direct.copy_probe = link.direct_copy_probe;
        direct.atomic32_probe = link.direct_atomic32_probe;
        direct.tma_load_f16_probe = link.direct_tma_load_f16_probe;
        direct.tma_store_f16_probe = link.direct_tma_store_f16_probe;
        direct.tma_reduce_f16_probe = link.direct_tma_reduce_f16_probe;
        direct.atomics =
            direct_atomic_capability_from_results(
                direct_copy_ok,
                link.native_atomic_supported,
                link.partial_native_atomic_supported,
                link.direct_atomic32_probe,
                link.direct_tma_load_f16_probe,
                link.direct_tma_store_f16_probe,
                link.direct_tma_reduce_f16_probe);
        direct.performance_rank = link.performance_rank;
        direct.description =
            link.preferred_kind == LinkKind::Nvlink
                ? "direct CUDA peer access, inferred NVLink-class link"
                : "direct CUDA peer access over PCIe/SYS-class link";

        link.transports.push_back(direct);
    }

    if (options.include_shm_fallback) {
        run_shm_probes(&link, options);

        const bool shm_available =
            options.run_validation_probes
                ? (link.shm_copy_probe.attempted &&
                   link.shm_copy_probe.passed)
                : true;

        TransportInfo shm{};
        shm.kind = TransportKind::Shm;
        shm.available = shm_available;
        shm.copy_probe = link.shm_copy_probe;
        shm.atomic32_probe = link.shm_atomic32_probe;
        shm.tma_load_f16_probe = link.shm_tma_load_f16_probe;
        shm.tma_store_f16_probe = link.shm_tma_store_f16_probe;
        shm.tma_reduce_f16_probe = link.shm_tma_reduce_f16_probe;
        shm.atomics =
            shm_atomic_capability_from_results(
                link.shm_copy_probe,
                link.shm_atomic32_probe,
                link.shm_tma_load_f16_probe,
                link.shm_tma_store_f16_probe,
                link.shm_tma_reduce_f16_probe);
        shm.performance_rank = -1;
        shm.description =
            "host shared-memory fallback transport using cudaHostAllocMapped probe";

        link.transports.push_back(shm);

        if (link.preferred_kind == LinkKind::Unsupported ||
            link.preferred_kind == LinkKind::Unknown) {
            link.preferred_kind =
                shm_available ? LinkKind::Shm : LinkKind::Unsupported;
        }
    }

    return link;
}

const TransportInfo* find_transport(
    const Link& link,
    TransportKind transport) {
    for (const TransportInfo& info : link.transports) {
        if (info.kind == transport) {
            return &info;
        }
    }

    return nullptr;
}

void append_probe_text(
    std::ostringstream& out,
    const char* name,
    const ProbeResult& probe) {
    if (!probe.attempted) {
        out << " " << name << "=na";
        return;
    }

    out << " " << name << "=" << (probe.passed ? 1 : 0);

    if (!probe.passed && !probe.error.empty()) {
        out << "(" << probe.error << ")";
    }
}

void append_probe_json(
    std::ostringstream& out,
    const char* name,
    const ProbeResult& probe,
    bool leading_comma = true) {
    if (leading_comma) {
        out << ",";
    }

    out << "\"" << name << "\":{"
        << "\"attempted\":" << bool_text(probe.attempted) << ","
        << "\"passed\":" << bool_text(probe.passed) << ","
        << "\"error\":\"" << escape_json(probe.error) << "\""
        << "}";
}

} // namespace

const Node* Topology::node_by_ordinal(int ordinal) const {
    for (const Node& node : nodes) {
        if (node.ordinal == ordinal) {
            return &node;
        }
    }

    return nullptr;
}

const Link* Topology::link_by_ordinals(
    int src_ordinal,
    int dst_ordinal) const {
    for (const Link& link : links) {
        if (link.src_ordinal == src_ordinal &&
            link.dst_ordinal == dst_ordinal) {
            return &link;
        }
    }

    return nullptr;
}

const char* link_kind_name(LinkKind kind) {
    switch (kind) {
        case LinkKind::Self:
            return "self";
        case LinkKind::Nvlink:
            return "nvlink";
        case LinkKind::Pcie:
            return "pcie";
        case LinkKind::Shm:
            return "shm";
        case LinkKind::Unsupported:
            return "unsupported";
        case LinkKind::Unknown:
        default:
            return "unknown";
    }
}

const char* transport_kind_name(TransportKind kind) {
    switch (kind) {
        case TransportKind::DirectNvlink:
            return "direct_nvlink";
        case TransportKind::DirectPcie:
            return "direct_pcie";
        case TransportKind::Shm:
            return "shm";
        default:
            return "unknown";
    }
}

bool link_supports_atomic_operation(
    const Link& link,
    TransportKind transport,
    AtomicOperationKind op) {
    const TransportInfo* info =
        find_transport(
            link,
            transport);

    if (info == nullptr || !info->available) {
        return false;
    }

    const AtomicCapability& caps = info->atomics;

    switch (op) {
        case AtomicOperationKind::Signal32:
            return caps.signal32;
        case AtomicOperationKind::GlobalLoadStore:
            return caps.global_load_store;
        case AtomicOperationKind::GlobalAtomic32:
            return caps.global_atomic_32;
        case AtomicOperationKind::GlobalAtomic64:
            return caps.global_atomic_64;
        case AtomicOperationKind::GlobalAtomicF32:
            return caps.global_atomic_f32;
        case AtomicOperationKind::GlobalAtomicF16:
            return caps.global_atomic_f16;
        case AtomicOperationKind::TmaLoadF16:
            return caps.tma_load_f16;
        case AtomicOperationKind::TmaStoreF16:
            return caps.tma_store_f16;
        case AtomicOperationKind::TmaReduceF16:
            return caps.tma_reduce_f16;
        case AtomicOperationKind::TmaReduceBf16:
            return caps.tma_reduce_bf16;
        case AtomicOperationKind::TmaReduceF32:
            return caps.tma_reduce_f32;
        default:
            return false;
    }
}

Topology discover_current_process_topology(
    const std::vector<int>& devices,
    const DiscoverOptions& options) {
    TOPO_TRACE(
        "enter discover_current_process_topology num_devices=%zu",
        devices.size());

    if (devices.empty()) {
        throw std::invalid_argument("discover_current_process_topology: no devices");
    }

    int device_count = 0;

    TOPO_TRACE("before cudaGetDeviceCount");
    check_cuda(
        cudaGetDeviceCount(&device_count),
        "cudaGetDeviceCount");
    TOPO_TRACE("after cudaGetDeviceCount device_count=%d", device_count);

    for (int device : devices) {
        TOPO_TRACE("validate device=%d", device);

        if (device < 0 || device >= device_count) {
            std::ostringstream oss;
            oss << "invalid CUDA device ordinal " << device;
            throw std::invalid_argument(oss.str());
        }

        TOPO_TRACE("before cudaSetDevice device=%d", device);
        check_cuda(
            cudaSetDevice(device),
            "cudaSetDevice");
        TOPO_TRACE("after cudaSetDevice device=%d", device);

        TOPO_TRACE("before cudaFree(nullptr) device=%d", device);
        check_cuda(
            cudaFree(nullptr),
            "cudaFree(nullptr)");
        TOPO_TRACE("after cudaFree(nullptr) device=%d", device);
    }

    Topology topology{};

    TOPO_TRACE("reserve nodes");
    topology.nodes.reserve(devices.size());

    for (size_t i = 0; i < devices.size(); ++i) {
        TOPO_TRACE(
            "before discover_node ordinal=%zu device=%d",
            i,
            devices[i]);

        topology.nodes.push_back(
            discover_node(
                static_cast<int>(i),
                devices[i]));

        TOPO_TRACE(
            "after discover_node ordinal=%zu device=%d",
            i,
            devices[i]);
    }

    TOPO_TRACE("reserve links");
    topology.links.reserve(devices.size() * devices.size());

    for (size_t i = 0; i < devices.size(); ++i) {
        for (size_t j = 0; j < devices.size(); ++j) {
            TOPO_TRACE(
                "before discover_link src_ord=%zu dst_ord=%zu src_dev=%d dst_dev=%d",
                i,
                j,
                devices[i],
                devices[j]);

            topology.links.push_back(
                discover_link(
                    static_cast<int>(i),
                    static_cast<int>(j),
                    devices[i],
                    devices[j],
                    options));

            TOPO_TRACE(
                "after discover_link src_ord=%zu dst_ord=%zu src_dev=%d dst_dev=%d",
                i,
                j,
                devices[i],
                devices[j]);
        }
    }

    TOPO_TRACE(
        "return topology nodes=%zu links=%zu",
        topology.nodes.size(),
        topology.links.size());

    return topology;
}

Topology discover_all_cuda_devices_topology(
    const DiscoverOptions& options) {
    TOPO_TRACE("enter discover_all_cuda_devices_topology");

    int device_count = 0;

    TOPO_TRACE("before cudaGetDeviceCount");
    check_cuda(
        cudaGetDeviceCount(&device_count),
        "cudaGetDeviceCount");
    TOPO_TRACE("after cudaGetDeviceCount device_count=%d", device_count);

    std::vector<int> devices;
    devices.reserve(static_cast<size_t>(device_count));

    for (int device = 0; device < device_count; ++device) {
        TOPO_TRACE("push device=%d", device);
        devices.push_back(device);
    }

    TOPO_TRACE("before discover_current_process_topology");
    Topology topo =
        discover_current_process_topology(
            devices,
            options);
    TOPO_TRACE("after discover_current_process_topology");

    return topo;
}

std::string topology_to_string(const Topology& topology) {
    std::ostringstream out;

    out << "ooverlap topology\n";
    out << "nodes:\n";

    for (const Node& node : topology.nodes) {
        out << "  ordinal=" << node.ordinal
            << " device=" << node.device
            << " name=\"" << node.name << "\""
            << " cc=" << node.compute_major << "." << node.compute_minor
            << " pci=" << node.pci_bus_id_string
            << " numa=" << node.numa_node
            << "\n";
    }

    out << "links:\n";

    for (const Link& link : topology.links) {
        if (link.src_ordinal == link.dst_ordinal) {
            continue;
        }

        out << "  GPU" << link.src_device
            << " -> GPU" << link.dst_device
            << " kind=" << link_kind_name(link.preferred_kind)
            << " access=" << (link.cuda_peer_access_supported ? 1 : 0)
            << " enabled=" << (link.cuda_peer_access_enabled ? 1 : 0)
            << " nativeAtomic=" << (link.native_atomic_supported ? 1 : 0)
            << " partialAtomic=" << (link.partial_native_atomic_supported ? 1 : 0)
            << " perfRank=" << link.performance_rank
            << " arrayAccess=" << (link.cuda_array_peer_access_supported ? 1 : 0)
            << " directCopy=" << (link.safe_for_direct_copy ? 1 : 0)
            << " tmaReduce=" << (link.safe_for_tma_reduce ? 1 : 0);

        append_probe_text(out, "directProbe", link.direct_copy_probe);
        append_probe_text(out, "directAtomic32Probe", link.direct_atomic32_probe);
        append_probe_text(out, "directTmaLoadF16Probe", link.direct_tma_load_f16_probe);
        append_probe_text(out, "directTmaStoreF16Probe", link.direct_tma_store_f16_probe);
        append_probe_text(out, "directTmaReduceF16Probe", link.direct_tma_reduce_f16_probe);

        out << "\n";

        for (const TransportInfo& transport : link.transports) {
            out << "    transport=" << transport_kind_name(transport.kind)
                << " available=" << (transport.available ? 1 : 0)
                << " signal32=" << (transport.atomics.signal32 ? 1 : 0)
                << " loadStore=" << (transport.atomics.global_load_store ? 1 : 0)
                << " atomic32=" << (transport.atomics.global_atomic_32 ? 1 : 0)
                << " atomic64=" << (transport.atomics.global_atomic_64 ? 1 : 0)
                << " atomicF32=" << (transport.atomics.global_atomic_f32 ? 1 : 0)
                << " atomicF16=" << (transport.atomics.global_atomic_f16 ? 1 : 0)
                << " tmaLoadF16=" << (transport.atomics.tma_load_f16 ? 1 : 0)
                << " tmaStoreF16=" << (transport.atomics.tma_store_f16 ? 1 : 0)
                << " tmaF16=" << (transport.atomics.tma_reduce_f16 ? 1 : 0)
                << " tmaBf16=" << (transport.atomics.tma_reduce_bf16 ? 1 : 0)
                << " tmaF32=" << (transport.atomics.tma_reduce_f32 ? 1 : 0)
                << " perfRank=" << transport.performance_rank
                << " estGBps=" << transport.estimated_bandwidth_gbps;

            append_probe_text(out, "copyProbe", transport.copy_probe);
            append_probe_text(out, "atomic32Probe", transport.atomic32_probe);
            append_probe_text(out, "tmaLoadF16Probe", transport.tma_load_f16_probe);
            append_probe_text(out, "tmaStoreF16Probe", transport.tma_store_f16_probe);
            append_probe_text(out, "tmaReduceF16Probe", transport.tma_reduce_f16_probe);

            out << "\n";
        }
    }

    return out.str();
}

std::string topology_to_json(const Topology& topology) {
    std::ostringstream out;

    out << "{";
    out << "\"nodes\":[";

    for (size_t i = 0; i < topology.nodes.size(); ++i) {
        const Node& node = topology.nodes[i];

        if (i != 0) {
            out << ",";
        }

        out << "{"
            << "\"ordinal\":" << node.ordinal << ","
            << "\"device\":" << node.device << ","
            << "\"name\":\"" << escape_json(node.name) << "\","
            << "\"compute_major\":" << node.compute_major << ","
            << "\"compute_minor\":" << node.compute_minor << ","
            << "\"pci_domain_id\":" << node.pci_domain_id << ","
            << "\"pci_bus_id\":" << node.pci_bus_id << ","
            << "\"pci_device_id\":" << node.pci_device_id << ","
            << "\"pci_bus_id_string\":\"" << escape_json(node.pci_bus_id_string) << "\","
            << "\"numa_node\":" << node.numa_node
            << "}";
    }

    out << "],";
    out << "\"links\":[";

    for (size_t i = 0; i < topology.links.size(); ++i) {
        const Link& link = topology.links[i];

        if (i != 0) {
            out << ",";
        }

        out << "{"
            << "\"src_ordinal\":" << link.src_ordinal << ","
            << "\"dst_ordinal\":" << link.dst_ordinal << ","
            << "\"src_device\":" << link.src_device << ","
            << "\"dst_device\":" << link.dst_device << ","
            << "\"preferred_kind\":\"" << link_kind_name(link.preferred_kind) << "\","
            << "\"cuda_peer_access_supported\":" << bool_text(link.cuda_peer_access_supported) << ","
            << "\"cuda_peer_access_enabled\":" << bool_text(link.cuda_peer_access_enabled) << ","
            << "\"cuda_array_peer_access_supported\":" << bool_text(link.cuda_array_peer_access_supported) << ","
            << "\"native_atomic_supported\":" << bool_text(link.native_atomic_supported) << ","
            << "\"partial_native_atomic_supported\":" << bool_text(link.partial_native_atomic_supported) << ","
            << "\"performance_rank\":" << link.performance_rank << ","
            << "\"safe_for_tma_reduce\":" << bool_text(link.safe_for_tma_reduce) << ","
            << "\"safe_for_direct_copy\":" << bool_text(link.safe_for_direct_copy);

        append_probe_json(out, "direct_copy_probe", link.direct_copy_probe);
        append_probe_json(out, "direct_atomic32_probe", link.direct_atomic32_probe);
        append_probe_json(out, "direct_tma_load_f16_probe", link.direct_tma_load_f16_probe);
        append_probe_json(out, "direct_tma_store_f16_probe", link.direct_tma_store_f16_probe);
        append_probe_json(out, "direct_tma_reduce_f16_probe", link.direct_tma_reduce_f16_probe);
        append_probe_json(out, "shm_copy_probe", link.shm_copy_probe);
        append_probe_json(out, "shm_atomic32_probe", link.shm_atomic32_probe);
        append_probe_json(out, "shm_tma_load_f16_probe", link.shm_tma_load_f16_probe);
        append_probe_json(out, "shm_tma_store_f16_probe", link.shm_tma_store_f16_probe);
        append_probe_json(out, "shm_tma_reduce_f16_probe", link.shm_tma_reduce_f16_probe);

        out << ",\"transports\":[";

        for (size_t t = 0; t < link.transports.size(); ++t) {
            const TransportInfo& transport = link.transports[t];

            if (t != 0) {
                out << ",";
            }

            const AtomicCapability& a = transport.atomics;

            out << "{"
                << "\"kind\":\"" << transport_kind_name(transport.kind) << "\","
                << "\"available\":" << bool_text(transport.available) << ","
                << "\"performance_rank\":" << transport.performance_rank << ","
                << "\"estimated_bandwidth_gbps\":" << transport.estimated_bandwidth_gbps << ","
                << "\"description\":\"" << escape_json(transport.description) << "\","
                << "\"atomics\":{"
                << "\"signal32\":" << bool_text(a.signal32) << ","
                << "\"global_load_store\":" << bool_text(a.global_load_store) << ","
                << "\"global_atomic_32\":" << bool_text(a.global_atomic_32) << ","
                << "\"global_atomic_64\":" << bool_text(a.global_atomic_64) << ","
                << "\"global_atomic_f32\":" << bool_text(a.global_atomic_f32) << ","
                << "\"global_atomic_f16\":" << bool_text(a.global_atomic_f16) << ","
                << "\"tma_load_f16\":" << bool_text(a.tma_load_f16) << ","
                << "\"tma_store_f16\":" << bool_text(a.tma_store_f16) << ","
                << "\"tma_reduce_f16\":" << bool_text(a.tma_reduce_f16) << ","
                << "\"tma_reduce_bf16\":" << bool_text(a.tma_reduce_bf16) << ","
                << "\"tma_reduce_f32\":" << bool_text(a.tma_reduce_f32)
                << "}";

            append_probe_json(out, "copy_probe", transport.copy_probe);
            append_probe_json(out, "atomic32_probe", transport.atomic32_probe);
            append_probe_json(out, "tma_load_f16_probe", transport.tma_load_f16_probe);
            append_probe_json(out, "tma_store_f16_probe", transport.tma_store_f16_probe);
            append_probe_json(out, "tma_reduce_f16_probe", transport.tma_reduce_f16_probe);

            out << "}";
        }

        out << "]"
            << "}";
    }

    out << "]";
    out << "}";

    return out.str();
}

} // namespace topology
} // namespace ooverlap
