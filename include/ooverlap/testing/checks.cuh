#pragma once

#include "ooverlap/comm.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include <stdexcept>
#include <string>

namespace ooverlap {
namespace testing {

inline void check_oo(
    oo_status_t status,
    const char* what) {
    if (status != OO_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + " failed: " + oo_status_string(status));
    }
}

inline void check_cuda(
    cudaError_t err,
    const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + " failed: " + cudaGetErrorString(err));
    }
}

inline void check_nccl(
    ncclResult_t result,
    const char* what,
    const char* file,
    int line) {
    if (result != ncclSuccess) {
        throw std::runtime_error(
            std::string("NCCL error at ") +
            file +
            ":" +
            std::to_string(line) +
            " in " +
            what +
            ": " +
            ncclGetErrorString(result));
    }
}

} // namespace testing
} // namespace ooverlap

#define OOVERLAP_TEST_NCCL_CHECK(cmd) \
    ::ooverlap::testing::check_nccl((cmd), #cmd, __FILE__, __LINE__)
