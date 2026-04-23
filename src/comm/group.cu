#include "comm/group.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace {

void validate_devices(
    const std::vector<int>& devices) {
    if (devices.empty()) {
        throw std::invalid_argument("group_init: devices must not be empty");
    }

    for (size_t i = 0; i < devices.size(); ++i) {
        for (size_t j = i + 1; j < devices.size(); ++j) {
            if (devices[i] == devices[j]) {
                throw std::invalid_argument("group_init: duplicate devices are not allowed");
            }
        }
    }
}

void enable_peer_access_all_to_all(
    const std::vector<int>& devices) {
    for (size_t i = 0; i < devices.size(); ++i) {
        for (size_t j = 0; j < devices.size(); ++j) {
            if (i == j) {
                continue;
            }

            int can_access = 0;
            system::runtime::check_cuda(
                cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]),
                "cudaDeviceCanAccessPeer");

            if (can_access == 0) {
                throw std::runtime_error("group_init: selected devices do not support peer access");
            }

            system::runtime::set_device(devices[i]);
            cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
            } else {
                system::runtime::check_cuda(err, "cudaDeviceEnablePeerAccess");
            }
        }
    }
}

} // namespace

void group_init(
    Group* group,
    const std::vector<int>& devices) {
    if (group == nullptr) {
        throw std::invalid_argument("group_init: group is null");
    }

    group_destroy(group);
    validate_devices(devices);

    try {
        group->devices = devices;
        group->nodes.resize(devices.size());

        for (size_t i = 0; i < devices.size(); ++i) {
            node_init(
                &group->nodes[i],
                static_cast<int>(i),
                devices[i]);
        }

        enable_peer_access_all_to_all(group->devices);
    } catch (...) {
        group_destroy(group);
        throw;
    }
}

void group_destroy(
    Group* group) {
    if (group == nullptr) {
        return;
    }

    for (auto& node : group->nodes) {
        node_destroy(&node);
    }

    group->nodes.clear();
    group->devices.clear();
}

Node* group_get_node(
    Group* group,
    int rank) {
    if (group == nullptr) {
        return nullptr;
    }
    if (rank < 0 || rank >= static_cast<int>(group->nodes.size())) {
        return nullptr;
    }
    return &group->nodes[static_cast<size_t>(rank)];
}

const Node* group_get_node(
    const Group* group,
    int rank) {
    if (group == nullptr) {
        return nullptr;
    }
    if (rank < 0 || rank >= static_cast<int>(group->nodes.size())) {
        return nullptr;
    }
    return &group->nodes[static_cast<size_t>(rank)];
}

} // namespace comm
} // namespace ooverlap
