#pragma once

#include "comm/buffer.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace ooverlap {
namespace comm {

struct Node {
    int rank = -1;
    int device = -1;
    cudaStream_t stream = nullptr;
    std::vector<Buffer> buffers{};
};

void node_init(
    Node* node,
    int rank,
    int device);

void node_destroy(
    Node* node);

int node_add_buffer(
    Node* node,
    size_t bytes,
    const std::vector<int>& visible_devices);

Buffer* node_get_buffer(
    Node* node,
    int index);

const Buffer* node_get_buffer(
    const Node* node,
    int index);

} // namespace comm
} // namespace ooverlap
