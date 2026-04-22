#include "comm/endpoint_persistent_kernel.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace {

__device__ __forceinline__ bool endpoint_persistent_should_stop(
    const uint32_t* stop_flag) {
    return (*reinterpret_cast<volatile const uint32_t*>(stop_flag)) != 0u;
}

__device__ __forceinline__ void endpoint_persistent_seed_chunk_state(
    collective::ChunkState* st,
    const exec::Chunk* chunk,
    const collective::OperationDesc* op) {
    if (st == nullptr || chunk == nullptr || op == nullptr) {
        return;
    }

    st->op_id = op->op_id;
    st->epoch = op->epoch;
    st->chunk_idx = static_cast<uint32_t>(chunk->chunk_idx);
    st->flags = collective::kChunkStateFlagSeeded |
                collective::kChunkStateFlagActive;
    st->contributions_seen = 1;
    st->expected_contributions = op->expected_contributions;
    st->step = 0;
    st->send_count = 0;
    st->offset_bytes = chunk->range_offset_bytes;
    st->bytes = chunk->bytes;

    if (op->expected_contributions <= 1) {
        st->flags |= collective::kChunkStateFlagDone;
        st->flags &= ~collective::kChunkStateFlagActive;
    }
}

__device__ __forceinline__ void endpoint_persistent_copy_bytes(
    unsigned char* dst,
    const unsigned char* src,
    size_t bytes) {
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ void endpoint_persistent_seed_local_partial(
    const exec::Chunk* chunk,
    const collective::OperationDesc* op) {
    const unsigned char* src = chunk->src;
    unsigned char* partial =
        collective::operation_desc_partial_chunk_ptr(
            op,
            static_cast<uint32_t>(chunk->chunk_idx));

    endpoint_persistent_copy_bytes(partial, src, chunk->bytes);
}

__device__ __forceinline__ void endpoint_persistent_finalize_chunk(
    const exec::Chunk* chunk,
    const collective::OperationDesc* op) {
    const unsigned char* partial =
        collective::operation_desc_partial_chunk_ptr(
            op,
            static_cast<uint32_t>(chunk->chunk_idx));
    unsigned char* dst =
        collective::operation_desc_dst_chunk_ptr(
            op,
            static_cast<uint32_t>(chunk->chunk_idx));

    endpoint_persistent_copy_bytes(dst, partial, chunk->bytes);
}

__global__ void endpoint_persistent_kernel_sm90(
    DeviceEndpointRuntime runtime,
    const collective::OperationDesc* operation,
    collective::ChunkState* chunk_states,
    const uint32_t* stop_flag) {
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ int shared_should_stop;
    __shared__ int shared_has_chunk;
    __shared__ exec::Chunk shared_chunk;
    __shared__ int shared_need_seed;

    exec::ChunkScheduler<1> scheduler{};

    if (threadIdx.x == 0) {
        exec::chunk_scheduler_init(
            &scheduler,
            runtime.submission,
            kEndpointPersistentChunkBytes);
        shared_has_chunk = 0;
        shared_need_seed = 0;
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            shared_should_stop = endpoint_persistent_should_stop(stop_flag) ? 1 : 0;
        }
        __syncthreads();

        if (shared_should_stop) {
            return;
        }

        if (threadIdx.x == 0) {
            if (!exec::chunk_scheduler_try_prime_current(&scheduler)) {
                shared_has_chunk = 0;
            } else {
                shared_chunk = *exec::chunk_scheduler_current(&scheduler);
                shared_has_chunk = 1;

                if (shared_chunk.chunk_idx < 0 ||
                    static_cast<uint32_t>(shared_chunk.chunk_idx) >= operation->num_chunks) {
                    shared_need_seed = 0;
                } else {
                    const collective::ChunkState* st =
                        &chunk_states[shared_chunk.chunk_idx];
                    shared_need_seed = collective::chunk_state_is_seeded(st) ? 0 : 1;
                }
            }
        }
        __syncthreads();

        if (!shared_has_chunk) {
#if defined(__CUDA_ARCH__)
            if (threadIdx.x == 0) {
                __nanosleep(256);
            }
#endif
            __syncthreads();
            continue;
        }

        if (shared_need_seed) {
            endpoint_persistent_seed_local_partial(&shared_chunk, operation);
            __syncthreads();

            if (collective::operation_desc_is_active(operation) &&
                operation->expected_contributions <= 1) {
                endpoint_persistent_finalize_chunk(&shared_chunk, operation);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                endpoint_persistent_seed_chunk_state(
                    &chunk_states[shared_chunk.chunk_idx],
                    &shared_chunk,
                    operation);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            exec::chunk_scheduler_advance(&scheduler);
        }
        __syncthreads();
    }
}

void configure_endpoint_persistent_kernel_smem(
    int device,
    size_t dynamic_smem_bytes) {
    if (dynamic_smem_bytes == 0) {
        return;
    }

    system::runtime::set_device(device);

    cudaDeviceProp prop{};
    system::runtime::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties(endpoint persistent)");

    if (dynamic_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "endpoint persistent kernel requested shared memory exceeds device opt-in limit");
    }

    if (dynamic_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                endpoint_persistent_kernel_sm90,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize endpoint persistent)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                endpoint_persistent_kernel_sm90,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout endpoint persistent)");
    }
}

} // namespace

bool endpoint_persistent_control_init(
    EndpointPersistentControl* ctl,
    int device) {
    if (ctl == nullptr) {
        throw std::invalid_argument("endpoint_persistent_control_init: ctl is null");
    }
    if (device < 0) {
        throw std::invalid_argument("endpoint_persistent_control_init: invalid device");
    }

    endpoint_persistent_control_destroy(ctl);

    ctl->device = device;
    system::runtime::set_device(device);

    system::runtime::check_cuda(
        cudaMalloc(&ctl->stop_flag, sizeof(uint32_t)),
        "cudaMalloc(endpoint persistent stop_flag)");

    system::runtime::check_cuda(
        cudaStreamCreateWithFlags(&ctl->control_stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags(endpoint persistent control_stream)");

    system::runtime::check_cuda(
        cudaMemsetAsync(
            ctl->stop_flag,
            0,
            sizeof(uint32_t),
            ctl->control_stream),
        "cudaMemsetAsync(endpoint persistent stop_flag init)");

    system::runtime::check_cuda(
        cudaStreamSynchronize(ctl->control_stream),
        "cudaStreamSynchronize(endpoint persistent stop_flag init)");

    return true;
}

void endpoint_persistent_control_reset(
    EndpointPersistentControl* ctl) {
    if (!endpoint_persistent_control_is_valid(ctl)) {
        throw std::invalid_argument(
            "endpoint_persistent_control_reset: control not configured");
    }

    system::runtime::set_device(ctl->device);
    system::runtime::check_cuda(
        cudaMemsetAsync(
            ctl->stop_flag,
            0,
            sizeof(uint32_t),
            ctl->control_stream),
        "cudaMemsetAsync(endpoint persistent reset)");

    system::runtime::check_cuda(
        cudaStreamSynchronize(ctl->control_stream),
        "cudaStreamSynchronize(endpoint persistent reset)");
}

void endpoint_persistent_control_request_stop(
    EndpointPersistentControl* ctl) {
    if (!endpoint_persistent_control_is_valid(ctl)) {
        throw std::invalid_argument(
            "endpoint_persistent_control_request_stop: control not configured");
    }

    system::runtime::set_device(ctl->device);

    system::runtime::check_cuda(
        cudaMemsetAsync(
            ctl->stop_flag,
            1,
            sizeof(uint32_t),
            ctl->control_stream),
        "cudaMemsetAsync(endpoint persistent request_stop)");

    system::runtime::check_cuda(
        cudaStreamSynchronize(ctl->control_stream),
        "cudaStreamSynchronize(endpoint persistent request_stop)");
}

void endpoint_persistent_control_destroy(
    EndpointPersistentControl* ctl) {
    if (ctl == nullptr) {
        return;
    }

    if (ctl->device >= 0) {
        system::runtime::set_device(ctl->device);
    }

    if (ctl->control_stream != nullptr) {
        system::runtime::check_cuda(
            cudaStreamDestroy(ctl->control_stream),
            "cudaStreamDestroy(endpoint persistent control_stream)");
    }

    if (ctl->stop_flag != nullptr) {
        system::runtime::check_cuda(
            cudaFree(ctl->stop_flag),
            "cudaFree(endpoint persistent stop_flag)");
    }

    ctl->stop_flag = nullptr;
    ctl->control_stream = nullptr;
    ctl->device = -1;
}

size_t endpoint_persistent_kernel_dynamic_smem_bytes() {
    return 0;
}

cudaError_t launch_endpoint_persistent_kernel_sm90(
    const DeviceEndpointRuntime* runtime,
    const collective::OperationDesc* operation,
    collective::ChunkState* chunk_states,
    const EndpointPersistentControl* control,
    cudaStream_t stream) {
    if (!device_endpoint_runtime_is_valid(runtime)) {
        return cudaErrorInvalidValue;
    }
    if (operation == nullptr || !collective::operation_desc_is_active(operation)) {
        return cudaErrorInvalidValue;
    }
    if (chunk_states == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!endpoint_persistent_control_is_valid(control)) {
        return cudaErrorInvalidValue;
    }
    if (runtime->device != control->device) {
        return cudaErrorInvalidDevice;
    }
    if (!collective::operation_desc_matches_submission(operation, runtime->submission)) {
        return cudaErrorInvalidValue;
    }

    const size_t dynamic_smem_bytes =
        endpoint_persistent_kernel_dynamic_smem_bytes();

    configure_endpoint_persistent_kernel_smem(
        runtime->device,
        dynamic_smem_bytes);

    cudaStream_t launch_stream =
        (stream != nullptr) ? stream : runtime->stream;

    system::runtime::set_device(runtime->device);

    endpoint_persistent_kernel_sm90<<<
        1,
        kEndpointPersistentThreads,
        dynamic_smem_bytes,
        launch_stream>>>(
        *runtime,
        operation,
        chunk_states,
        control->stop_flag);

    return cudaGetLastError();
}

} // namespace comm
} // namespace ooverlap
