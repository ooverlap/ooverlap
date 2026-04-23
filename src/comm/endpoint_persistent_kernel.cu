#include "comm/endpoint_persistent_kernel.h"

#include "comm/exec/chunk_pipeline.h"
#include "comm/exec/chunk_scheduler.h"
#include "comm/exec/pipeline_load.h"
#include "comm/exec/pipeline_reduce.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#define OOVERLAP_ENDPOINT_DEBUG 0

namespace ooverlap {
namespace comm {
namespace {

static constexpr int kPersistentStageDepth = 4;
static constexpr uint32_t kPersistentPrimeDepth =
    (kPersistentStageDepth > 1)
        ? static_cast<uint32_t>(kPersistentStageDepth - 1)
        : 1u;
static constexpr size_t kEndpointPersistentStaticSharedBytes =
    sizeof(sync::semaphore) * kPersistentStageDepth;

__device__ __forceinline__ bool endpoint_persistent_should_stop(
    const uint32_t* stop_flag) {
    return (*reinterpret_cast<volatile const uint32_t*>(stop_flag)) != 0u;
}

__device__ __forceinline__ bool endpoint_persistent_runtime_is_minimally_valid(
    const DeviceEndpointRuntime* runtime) {
    return runtime != nullptr &&
           runtime->rank >= 0 &&
           runtime->device >= 0 &&
           runtime->stream != nullptr;
}

__device__ __forceinline__ void endpoint_persistent_init_chunk_state(
    collective::ChunkState* st,
    const collective::OperationDesc* op,
    uint32_t chunk_idx) {
    collective::chunk_state_clear(st);
    st->chunk_idx = chunk_idx;
    st->last_step_started = 0;
    st->last_step_completed = 0;
    st->flags = collective::kChunkStateFlagInitialized;
    st->offset_bytes = collective::operation_desc_chunk_offset_bytes(op, chunk_idx);
    st->bytes = collective::operation_desc_chunk_bytes_at(op, chunk_idx);

    if (collective::operation_desc_total_ring_steps(op) == 0) {
        st->flags |= collective::kChunkStateFlagDone;
    }
}

__global__ void endpoint_persistent_kernel_sm90(
    DeviceEndpointRuntime runtime,
    const collective::OperationDesc* operation,
    const uint32_t* stop_flag) {
    if (blockIdx.x != 0) {
        return;
    }
    if (!endpoint_persistent_runtime_is_minimally_valid(&runtime)) {
        return;
    }
    if (operation == nullptr || !collective::operation_desc_is_active(operation)) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore stage_barriers[kPersistentStageDepth];
    __shared__ int should_stop;
    __shared__ int has_work;
        __shared__ uint32_t debug_idle_loops;

    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(operation);
    volatile uint32_t* done =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_done(operation));

    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(operation);

    // For fast thing (but should add later)
/*    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x);*/
         /*idx < operation->num_chunks;*/
         /*idx += static_cast<uint32_t>(blockDim.x)) {*/
        /*endpoint_persistent_init_chunk_state(*/
            /*&chunk_states[idx],*/
            /*operation,*/
            /*idx);*/

        /*if (total_steps == 0) {*/
            /*atomicExch(*/
                /*reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(&done[idx])),*/
                /*1u);*/
        /*}*/
    /*}*/
    /*__syncthreads();*/

    using Scheduler = exec::ChunkScheduler;
    using Pipe = exec::ChunkPipeline<
        kPersistentStageDepth,
        Scheduler,
        exec::PipelineTMALoad,
        exec::PipelineTMAStepApplyNoFtzF16>;

    __shared__ Scheduler shared_scheduler;
    __shared__ Pipe shared_pipe;

    if (threadIdx.x == 0) {
        exec::chunk_scheduler_init_operation(&shared_scheduler, operation);

        exec::chunk_pipeline_bind_stage_storage<
            kPersistentStageDepth,
            kEndpointPersistentChunkBytes>(
            &shared_pipe,
            shared_raw,
            stage_barriers);

        exec::chunk_pipeline_init(&shared_pipe, &shared_scheduler);
        debug_idle_loops = 0;
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            should_stop = endpoint_persistent_should_stop(stop_flag) ? 1 : 0;
        }
        __syncthreads();

        if (should_stop) {
            return;
        }

        if (threadIdx.x == 0) {
            // Keep one stage empty before issuing the current bulk op.
            // That empty slot is immediately refilled after issue, which
            // gives us true "issue current / preload next" overlap.
            has_work = exec::chunk_pipeline_try_prime(
                &shared_pipe,
                kPersistentPrimeDepth) ? 1 : 0;
        }
        __syncthreads();

        
        if (!has_work) {
#if OOVERLAP_ENDPOINT_DEBUG
            if (threadIdx.x == 0) {
                ++debug_idle_loops;
                if ((debug_idle_loops & 0x3ffffu) == 0u) {
                    volatile uint32_t* local_done =
                        reinterpret_cast<volatile uint32_t*>(
                            collective::operation_desc_local_done(operation));
                    volatile uint32_t* local_head =
                        reinterpret_cast<volatile uint32_t*>(
                            collective::operation_desc_local_ready_head(operation));
                    volatile uint32_t* local_tail =
                        reinterpret_cast<volatile uint32_t*>(
                            collective::operation_desc_local_ready_tail(operation));

                    printf(
                        "[idle] rank=%d idle_loops=%u queue_count=%u head=%u tail=%u done0=%u\n",
                        operation->rank,
                        debug_idle_loops,
                        shared_pipe.scheduler.ready_count,
                        exec::chunk_scheduler_atomic_load_u32(local_head),
                        exec::chunk_scheduler_atomic_load_u32(local_tail),
                        (operation->num_chunks > 0)
                            ? exec::chunk_scheduler_atomic_load_u32(&local_done[0])
                            : 0u);
                }
            }
#endif
#if defined(__CUDA_ARCH__)
            if (threadIdx.x == 0) {
                __nanosleep(256);
            }
#endif
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) {
            debug_idle_loops = 0;
        }
        __syncthreads();

        exec::chunk_pipeline_wait_current_stage(&shared_pipe);
        __syncthreads();

        exec::chunk_pipeline_issue_current_apply(&shared_pipe);
        __syncthreads();

        if (threadIdx.x == 0) {
            // Refill the one reserved stage right after the current bulk op
            // is issued, so the next load overlaps with the current
            // store/store-reduce.
            (void)exec::chunk_pipeline_try_prime(
                &shared_pipe,
                static_cast<uint32_t>(kPersistentStageDepth));
        }
        __syncthreads();

        exec::chunk_pipeline_finish_current_apply(&shared_pipe);
        __syncthreads();

        exec::chunk_pipeline_wait_current_complete(&shared_pipe);
        __syncthreads();

        if (threadIdx.x == 0) {
            exec::chunk_pipeline_retire_current(&shared_pipe);
        }
        __syncthreads();
    }
}

void configure_endpoint_persistent_kernel_smem(
    int device,
    size_t dynamic_smem_bytes) {
    system::runtime::set_device(device);

    cudaDeviceProp prop{};
    system::runtime::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties(endpoint persistent)");

    const size_t total_smem_bytes =
        dynamic_smem_bytes + kEndpointPersistentStaticSharedBytes;

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "endpoint persistent kernel requested shared memory exceeds device opt-in limit");
    }

    if (total_smem_bytes >
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
    return kEndpointPersistentChunkBytes * kPersistentStageDepth;
}

cudaError_t launch_endpoint_persistent_kernel_sm90(
    const DeviceEndpointRuntime* runtime,
    const collective::OperationDesc* operation,
    const EndpointPersistentControl* control,
    cudaStream_t stream) {
    if (runtime == nullptr ||
        runtime->rank < 0 ||
        runtime->device < 0 ||
        runtime->stream == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (operation == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (!endpoint_persistent_control_is_valid(control)) {
        return cudaErrorInvalidValue;
    }
    if (runtime->device != control->device) {
        return cudaErrorInvalidDevice;
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
        control->stop_flag);

    return cudaGetLastError();
}

} // namespace comm
} // namespace ooverlap
