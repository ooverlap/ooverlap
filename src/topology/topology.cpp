#include "topology/topology.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

AtomicCapability direct_atomic_capability_from_p2p(
    bool peer_access_supported,
    bool native_atomic_supported) {
    AtomicCapability caps{};

    caps.signal32 = peer_access_supported;
    caps.global_load_store = peer_access_supported;

    if (peer_access_supported && native_atomic_supported) {
        caps.global_atomic_32 = true;
        caps.global_atomic_64 = true;
        caps.global_atomic_f32 = true;
        caps.global_atomic_f16 = false;
        caps.tma_reduce_f16 = true;
        caps.tma_reduce_bf16 = true;
        caps.tma_reduce_f32 = true;
    }

    return caps;
}

AtomicCapability shm_atomic_capability_for_device_pair(
    int src_device,
    int dst_device) {
    AtomicCapability caps{};

    caps.signal32 = true;
    caps.global_load_store = true;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 8000
    const int src_host_native_atomic =
        get_device_attr_or_default(
            cudaDevAttrHostNativeAtomicSupported,
            src_device,
            0);

    const int dst_host_native_atomic =
        get_device_attr_or_default(
            cudaDevAttrHostNativeAtomicSupported,
            dst_device,
            0);

    if (src_host_native_atomic && dst_host_native_atomic) {
        caps.global_atomic_32 = true;
        caps.global_atomic_64 = true;
    }
#else
    (void)src_device;
    (void)dst_device;
#endif

    return caps;
}

LinkKind infer_direct_link_kind(
    bool peer_access_supported,
    bool native_atomic_supported,
    int performance_rank) {
    if (!peer_access_supported) {
        return LinkKind::Unsupported;
    }

    /*
     * CUDA runtime does not expose "NVLink vs SYS/PCIe" directly.
     *
     * On the Hopper systems we are targeting, NVLink island pairs report
     * native peer atomics and best performance rank, while SYS/PCIe pairs can
     * still report access=1 but nativeAtomic=0/perfRank=1.
     */
    if (native_atomic_supported && performance_rank == 0) {
        return LinkKind::Nvlink;
    }

    return LinkKind::Pcie;
}

Node discover_node(
    int ordinal,
    int device) {
    cudaDeviceProp prop{};
    check_cuda(
        cudaGetDeviceProperties(
            &prop,
            device),
        "cudaGetDeviceProperties");

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
    cudaError_t bus_err =
        cudaDeviceGetPCIBusId(
            pci_bus_id,
            static_cast<int>(sizeof(pci_bus_id)),
            device);

    if (bus_err == cudaSuccess) {
        node.pci_bus_id_string = pci_bus_id;
    } else {
        (void)cudaGetLastError();
    }

    node.numa_node =
        query_numa_node_from_pci(
            node.pci_domain_id,
            node.pci_bus_id,
            node.pci_device_id);

    return node;
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

        if (options.enable_peer_access && link.cuda_peer_access_supported) {
            link.cuda_peer_access_enabled =
                enable_peer_access_one_way(
                    src_device,
                    dst_device);
        }

        link.safe_for_direct_copy =
            link.cuda_peer_access_supported &&
            (options.enable_peer_access ? link.cuda_peer_access_enabled : true);

        link.safe_for_tma_reduce =
            link.safe_for_direct_copy &&
            link.native_atomic_supported;

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
        direct.available = link.safe_for_direct_copy;
        direct.atomics =
            direct_atomic_capability_from_p2p(
                link.safe_for_direct_copy,
                link.native_atomic_supported);
        direct.performance_rank = link.performance_rank;
        direct.description =
            link.preferred_kind == LinkKind::Nvlink
                ? "direct CUDA peer access, inferred NVLink-class link"
                : "direct CUDA peer access over PCIe/SYS-class link";

        link.transports.push_back(direct);
    }

    if (options.include_shm_fallback) {
        TransportInfo shm{};
        shm.kind = TransportKind::Shm;
        shm.available = true;
        shm.atomics =
            shm_atomic_capability_for_device_pair(
                src_device,
                dst_device);
        shm.performance_rank = -1;
        shm.description =
            "host shared-memory fallback transport; no CUDA bandwidth estimate";

        link.transports.push_back(shm);

        if (link.preferred_kind == LinkKind::Unsupported ||
            link.preferred_kind == LinkKind::Unknown) {
            link.preferred_kind = LinkKind::Shm;
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
    if (devices.empty()) {
        throw std::invalid_argument("discover_current_process_topology: no devices");
    }

    int device_count = 0;
    check_cuda(
        cudaGetDeviceCount(&device_count),
        "cudaGetDeviceCount");

    for (int device : devices) {
        if (device < 0 || device >= device_count) {
            std::ostringstream oss;
            oss << "invalid CUDA device ordinal " << device;
            throw std::invalid_argument(oss.str());
        }

        check_cuda(
            cudaSetDevice(device),
            "cudaSetDevice");
        check_cuda(
            cudaFree(nullptr),
            "cudaFree(nullptr)");
    }

    Topology topology{};

    topology.nodes.reserve(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        topology.nodes.push_back(
            discover_node(
                static_cast<int>(i),
                devices[i]));
    }

    topology.links.reserve(devices.size() * devices.size());

    for (size_t i = 0; i < devices.size(); ++i) {
        for (size_t j = 0; j < devices.size(); ++j) {
            topology.links.push_back(
                discover_link(
                    static_cast<int>(i),
                    static_cast<int>(j),
                    devices[i],
                    devices[j],
                    options));
        }
    }

    return topology;
}

Topology discover_all_cuda_devices_topology(
    const DiscoverOptions& options) {
    int device_count = 0;
    check_cuda(
        cudaGetDeviceCount(&device_count),
        "cudaGetDeviceCount");

    std::vector<int> devices;
    devices.reserve(static_cast<size_t>(device_count));

    for (int device = 0; device < device_count; ++device) {
        devices.push_back(device);
    }

    return discover_current_process_topology(
        devices,
        options);
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
            << " perfRank=" << link.performance_rank
            << " arrayAccess=" << (link.cuda_array_peer_access_supported ? 1 : 0)
            << " directCopy=" << (link.safe_for_direct_copy ? 1 : 0)
            << " tmaReduce=" << (link.safe_for_tma_reduce ? 1 : 0)
            << "\n";

        for (const TransportInfo& transport : link.transports) {
            out << "    transport=" << transport_kind_name(transport.kind)
                << " available=" << (transport.available ? 1 : 0)
                << " signal32=" << (transport.atomics.signal32 ? 1 : 0)
                << " loadStore=" << (transport.atomics.global_load_store ? 1 : 0)
                << " atomic32=" << (transport.atomics.global_atomic_32 ? 1 : 0)
                << " atomic64=" << (transport.atomics.global_atomic_64 ? 1 : 0)
                << " atomicF32=" << (transport.atomics.global_atomic_f32 ? 1 : 0)
                << " atomicF16=" << (transport.atomics.global_atomic_f16 ? 1 : 0)
                << " tmaF16=" << (transport.atomics.tma_reduce_f16 ? 1 : 0)
                << " tmaBf16=" << (transport.atomics.tma_reduce_bf16 ? 1 : 0)
                << " tmaF32=" << (transport.atomics.tma_reduce_f32 ? 1 : 0)
                << " perfRank=" << transport.performance_rank
                << " estGBps=" << transport.estimated_bandwidth_gbps
                << "\n";
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
            << "\"cuda_peer_access_supported\":" << (link.cuda_peer_access_supported ? "true" : "false") << ","
            << "\"cuda_peer_access_enabled\":" << (link.cuda_peer_access_enabled ? "true" : "false") << ","
            << "\"cuda_array_peer_access_supported\":" << (link.cuda_array_peer_access_supported ? "true" : "false") << ","
            << "\"native_atomic_supported\":" << (link.native_atomic_supported ? "true" : "false") << ","
            << "\"performance_rank\":" << link.performance_rank << ","
            << "\"safe_for_tma_reduce\":" << (link.safe_for_tma_reduce ? "true" : "false") << ","
            << "\"safe_for_direct_copy\":" << (link.safe_for_direct_copy ? "true" : "false") << ","
            << "\"transports\":[";

        for (size_t t = 0; t < link.transports.size(); ++t) {
            const TransportInfo& transport = link.transports[t];

            if (t != 0) {
                out << ",";
            }

            const AtomicCapability& a = transport.atomics;

            out << "{"
                << "\"kind\":\"" << transport_kind_name(transport.kind) << "\"," 
                << "\"available\":" << (transport.available ? "true" : "false") << ","
                << "\"performance_rank\":" << transport.performance_rank << ","
                << "\"estimated_bandwidth_gbps\":" << transport.estimated_bandwidth_gbps << ","
                << "\"description\":\"" << escape_json(transport.description) << "\"," 
                << "\"atomics\":{"
                << "\"signal32\":" << (a.signal32 ? "true" : "false") << ","
                << "\"global_load_store\":" << (a.global_load_store ? "true" : "false") << ","
                << "\"global_atomic_32\":" << (a.global_atomic_32 ? "true" : "false") << ","
                << "\"global_atomic_64\":" << (a.global_atomic_64 ? "true" : "false") << ","
                << "\"global_atomic_f32\":" << (a.global_atomic_f32 ? "true" : "false") << ","
                << "\"global_atomic_f16\":" << (a.global_atomic_f16 ? "true" : "false") << ","
                << "\"tma_reduce_f16\":" << (a.tma_reduce_f16 ? "true" : "false") << ","
                << "\"tma_reduce_bf16\":" << (a.tma_reduce_bf16 ? "true" : "false") << ","
                << "\"tma_reduce_f32\":" << (a.tma_reduce_f32 ? "true" : "false")
                << "}"
                << "}";
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
