#pragma once

#include <cuda_runtime.h>

namespace ooverlap {
namespace comm {

struct Endpoint {
    int rank = -1;
    int device = -1;
    cudaStream_t stream = nullptr;
};

inline bool endpoint_is_valid(const Endpoint& ep) {
    return ep.rank >= 0 &&
           ep.device >= 0 &&
           ep.stream != nullptr;
}

} // namespace comm
} // namespace ooverlap
