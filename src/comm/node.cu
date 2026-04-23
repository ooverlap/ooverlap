#include "comm/node.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <stdexcept>

namespace ooverlap {
namespace comm {

void node_init(
    Node* node,
    int rank,
    int device) {
    if (node == nullptr) {
        throw std::invalid_argument("node_init: node is null");
    }

    node_destroy(node);

    node->rank = rank;
    node->device = device;

    system::runtime::ensure_context_on_device(device);
    node->stream = system::runtime::create_stream_on_device(device);
}

void node_destroy(
    Node* node) {
    if (node == nullptr) {
        return;
    }

    for (auto& buf : node->buffers) {
        buffer_destroy(&buf);
    }
    node->buffers.clear();

    if (node->stream != nullptr && node->device >= 0) {
        system::runtime::destroy_stream_on_device(node->device, node->stream);
    }

    node->rank = -1;
    node->device = -1;
}

int node_add_buffer(
    Node* node,
    size_t bytes,
    const std::vector<int>& visible_devices) {
    if (node == nullptr) {
        throw std::invalid_argument("node_add_buffer: node is null");
    }
    if (node->device < 0) {
        throw std::invalid_argument("node_add_buffer: node is not initialized");
    }

    Buffer buf{};
    buffer_init(
        &buf,
        node->rank,
        node->device,
        bytes,
        visible_devices);

    node->buffers.push_back(buf);
    return static_cast<int>(node->buffers.size() - 1);
}

Buffer* node_get_buffer(
    Node* node,
    int index) {
    if (node == nullptr) {
        return nullptr;
    }
    if (index < 0 || index >= static_cast<int>(node->buffers.size())) {
        return nullptr;
    }
    return &node->buffers[static_cast<size_t>(index)];
}

const Buffer* node_get_buffer(
    const Node* node,
    int index) {
    if (node == nullptr) {
        return nullptr;
    }
    if (index < 0 || index >= static_cast<int>(node->buffers.size())) {
        return nullptr;
    }
    return &node->buffers[static_cast<size_t>(index)];
}

} // namespace comm
} // namespace ooverlap
