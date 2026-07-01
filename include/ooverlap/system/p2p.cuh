#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include "ooverlap/comm.h"

namespace ooverlap {
namespace system {
namespace p2p {

inline oo_status_t cuda_status_to_oo(cudaError_t err) {
    if (err == cudaSuccess) {
        return OO_SUCCESS;
    }

    if (err == cudaErrorInvalidDevice) {
        return OO_ERROR_INVALID_DEVICE;
    }

    if (err == cudaErrorInvalidValue) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    return OO_ERROR_CUDA;
}

inline oo_status_t ensure_context_on_device_status(int device) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return cuda_status_to_oo(err);
    }

    err = cudaFree(nullptr);
    return cuda_status_to_oo(err);
}

inline oo_status_t enable_peer_access_one_way_status(
    int local_device,
    int peer_device,
    bool* out_enabled) {
    if (out_enabled != nullptr) {
        *out_enabled = false;
    }

    if (local_device < 0 || peer_device < 0) {
        return OO_ERROR_INVALID_DEVICE;
    }

    if (local_device == peer_device) {
        if (out_enabled != nullptr) {
            *out_enabled = true;
        }

        return OO_SUCCESS;
    }

    oo_status_t status =
        ensure_context_on_device_status(local_device);

    if (status != OO_SUCCESS) {
        return status;
    }

    status =
        ensure_context_on_device_status(peer_device);

    if (status != OO_SUCCESS) {
        return status;
    }

    cudaError_t err =
        cudaSetDevice(local_device);

    if (err != cudaSuccess) {
        return cuda_status_to_oo(err);
    }

    int can_access = 0;

    err =
        cudaDeviceCanAccessPeer(
            &can_access,
            local_device,
            peer_device);

    if (err != cudaSuccess) {
        return cuda_status_to_oo(err);
    }

    if (!can_access) {
        return OO_ERROR_UNSUPPORTED;
    }

    err =
        cudaDeviceEnablePeerAccess(
            peer_device,
            0);

    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        /*
         * Clear sticky runtime error state.
         */
        (void)cudaGetLastError();

        if (out_enabled != nullptr) {
            *out_enabled = true;
        }

        return OO_SUCCESS;
    }

    if (err != cudaSuccess) {
        return cuda_status_to_oo(err);
    }

    if (out_enabled != nullptr) {
        *out_enabled = true;
    }

    return OO_SUCCESS;
}

} // namespace p2p
} // namespace system
} // namespace ooverlap
