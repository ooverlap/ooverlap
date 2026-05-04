#pragma once

#include "ooverlap/comm.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include <stdexcept>
#include <string>

namespace ooverlap {
namespace testing {

inline const char* oo_status_string(oo_status_t status) {
    switch (status) {
        case OO_SUCCESS:
            return "OO_SUCCESS";
        case OO_ERROR_INVALID_ARGUMENT:
            return "OO_ERROR_INVALID_ARGUMENT";
        case OO_ERROR_INVALID_DEVICE:
            return "OO_ERROR_INVALID_DEVICE";
        case OO_ERROR_UNSUPPORTED:
            return "OO_ERROR_UNSUPPORTED";
        case OO_ERROR_CUDA:
            return "OO_ERROR_CUDA";
        case OO_ERROR_INTERNAL:
            return "OO_ERROR_INTERNAL";
        default:
            return "OO_ERROR_UNKNOWN";
    }
}

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
