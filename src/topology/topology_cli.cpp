#include "topology/topology.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0
        << " [--json] [--no-enable-peer] [--no-shm] [--require-peer]"
        << " [--no-probe] [--no-tma-probe] [--no-atomic-probe] [devices...]\n"
        << "\n"
        << "examples:\n"
        << "  " << argv0 << "\n"
        << "  " << argv0 << " --json 0 1 2 3\n"
        << "  " << argv0 << " --no-tma-probe 0 2\n";
}

} // namespace

int main(int argc, char** argv) {
    bool json = false;

    ooverlap::topology::DiscoverOptions options{};
    options.enable_peer_access = true;
    options.include_shm_fallback = true;
    options.require_cuda_peer_access = false;
    options.run_validation_probes = true;
    options.run_tma_probes = true;
    options.run_atomic_probes = true;

    std::vector<int> devices;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--json") {
            json = true;
            continue;
        }

        if (arg == "--no-enable-peer") {
            options.enable_peer_access = false;
            continue;
        }

        if (arg == "--no-shm") {
            options.include_shm_fallback = false;
            continue;
        }

        if (arg == "--require-peer") {
            options.require_cuda_peer_access = true;
            continue;
        }

        if (arg == "--no-probe") {
            options.run_validation_probes = false;
            options.run_tma_probes = false;
            options.run_atomic_probes = false;
            continue;
        }

        if (arg == "--no-tma-probe") {
            options.run_tma_probes = false;
            continue;
        }

        if (arg == "--no-atomic-probe") {
            options.run_atomic_probes = false;
            continue;
        }

        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        }

        char* end = nullptr;
        const long value =
            std::strtol(
                arg.c_str(),
                &end,
                10);

        if (end == arg.c_str() || *end != '\0' || value < 0) {
            usage(argv[0]);
            return 2;
        }

        devices.push_back(static_cast<int>(value));
    }

    try {
        ooverlap::topology::Topology topo =
            devices.empty()
                ? ooverlap::topology::discover_all_cuda_devices_topology(options)
                : ooverlap::topology::discover_current_process_topology(devices, options);

        if (json) {
            std::cout << ooverlap::topology::topology_to_json(topo) << "\n";
        } else {
            std::cout << ooverlap::topology::topology_to_string(topo);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "topology discovery failed: " << e.what() << "\n";
        return 1;
    }
}
