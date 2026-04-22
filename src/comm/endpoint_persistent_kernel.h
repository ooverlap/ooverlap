#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comm/collective/operation.h"
#include "comm/endpoint_runtime.h"

namespace ooverlap {
namespace comm {

static constexpr int kEndpointPersistentThreads = 128;
static constexpr size_t kEndpointPersistentChunkBytes = 16 * 1024;

struct EndpointPersistentControl {
    uint32_t* stop_flag = nullptr;  // device pointer
    cudaStream_t control_stream = nullptr;
    int device = -1;
};

__host__ __device__ __forceinline__ bool endpoint_persistent_control_is_valid(
    const EndpointPersistentControl* ctl) {
    return ctl != nullptr &&
           ctl->stop_flag != nullptr &&
           ctl->control_stream != nullptr &&
           ctl->device >= 0;
}

bool endpoint_persistent_control_init(
    EndpointPersistentControl* ctl,
    int device);

void endpoint_persistent_control_reset(
    EndpointPersistentControl* ctl);

void endpoint_persistent_control_request_stop(
    EndpointPersistentControl* ctl);

void endpoint_persistent_control_destroy(
    EndpointPersistentControl* ctl);

size_t endpoint_persistent_kernel_dynamic_smem_bytes();

cudaError_t launch_endpoint_persistent_kernel_sm90(
    const DeviceEndpointRuntime* runtime,
    const collective::OperationDesc* operation,
    const EndpointPersistentControl* control,
    cudaStream_t stream = nullptr);

} // namespace comm
} // namespace ooverlap
