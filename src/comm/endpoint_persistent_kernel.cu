#include "comm/endpoint_persistent_kernel.h"

#include "comm/exec/chunk_pipeline.h"
#include "comm/exec/chunk_scheduler.h"
#include "comm/exec/pipeline_load.h"
#include "comm/exec/pipeline_reduce.h"
#include "comm/exec/pipeline_noop.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#ifndef OOVERLAP_ENDPOINT_DEBUG
#define OOVERLAP_ENDPOINT_DEBUG 0
#endif

#ifndef OOVERLAP_ENDPOINT_DISABLE_DATAPATH
#define OOVERLAP_ENDPOINT_DISABLE_DATAPATH 1
#endif

#ifndef OOVERLAP_ENDPOINT_PROFILE_SCHEDULER
#define OOVERLAP_ENDPOINT_PROFILE_SCHEDULER 1
#endif

#ifndef OOVERLAP_ENDPOINT_PROFILE_SCHEDULER_PRINT
#define OOVERLAP_ENDPOINT_PROFILE_SCHEDULER_PRINT 1
#endif

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

#if OOVERLAP_ENDPOINT_PROFILE_SCHEDULER

    using Scheduler = exec::ChunkScheduler;
    __shared__ Scheduler shared_scheduler;

    __shared__ uint32_t debug_idle_loops;
    __shared__ unsigned int dbg_retired;
    __shared__ unsigned long long dbg_cycles_activate;
    __shared__ unsigned long long dbg_cycles_retire;

    if (threadIdx.x == 0) {
        exec::chunk_scheduler_init_operation(&shared_scheduler, operation);
        debug_idle_loops = 0u;
        dbg_retired = 0u;
        dbg_cycles_activate = 0ull;
        dbg_cycles_retire = 0ull;
    }

    if (threadIdx.x != 0) {
        return;
    }

    while (true) {
        if (endpoint_persistent_should_stop(stop_flag)) {
#if OOVERLAP_ENDPOINT_PROFILE_SCHEDULER_PRINT
            if (dbg_retired > 0u) {
                const unsigned long long avg_activate =
                    dbg_cycles_activate / static_cast<unsigned long long>(dbg_retired);
                const unsigned long long avg_retire =
                    dbg_cycles_retire / static_cast<unsigned long long>(dbg_retired);

                printf(
                    "[sched-prof] rank=%d retired=%u avg_activate_cycles=%llu avg_retire_cycles=%llu total_activate_cycles=%llu total_retire_cycles=%llu idle_loops=%u\n",
                    operation->rank,
                    dbg_retired,
                    avg_activate,
                    avg_retire,
                    dbg_cycles_activate,
                    dbg_cycles_retire,
                    debug_idle_loops);
            } else {
                printf(
                    "[sched-prof] rank=%d retired=0 avg_activate_cycles=0 avg_retire_cycles=0 total_activate_cycles=0 total_retire_cycles=0 idle_loops=%u\n",
                    operation->rank,
                    debug_idle_loops);
            }
#endif
            return;
        }

        const unsigned long long t0 = clock64();
        const bool has_work =
            exec::chunk_scheduler_try_activate_next_chunk(&shared_scheduler);
        const unsigned long long t1 = clock64();

        if (!has_work) {
            ++debug_idle_loops;
#if defined(__CUDA_ARCH__)
            __nanosleep(256);
#endif
            continue;
        }

        debug_idle_loops = 0u;

        exec::chunk_scheduler_retire_current(&shared_scheduler);
        const unsigned long long t2 = clock64();

        dbg_cycles_activate += (t1 - t0);
        dbg_cycles_retire += (t2 - t1);
        ++dbg_retired;
    }

#else

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore stage_barriers[kPersistentStageDepth];
    __shared__ int should_stop;
    __shared__ int has_work;
    __shared__ uint32_t debug_idle_loops;

    using Scheduler = exec::ChunkScheduler;
#if OOVERLAP_ENDPOINT_DISABLE_DATAPATH
    using Pipe = exec::ChunkPipeline<
        kPersistentStageDepth,
        Scheduler,
        exec::PipelineNoOpLoad,
        exec::PipelineNoOpApply>;
#else
    using Pipe = exec::ChunkPipeline<
        kPersistentStageDepth,
        Scheduler,
        exec::PipelineTMALoad,
        exec::PipelineTMAStepApplyNoFtzF16>;
#endif

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
        debug_idle_loops = 0u;
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
                    volatile uint32_t* local_progress =
                        reinterpret_cast<volatile uint32_t*>(
                            collective::operation_desc_local_progress(operation));

                    const uint32_t progress0 =
                        (operation->num_chunks > 0)
                            ? exec::chunk_scheduler_volatile_load_u32(&local_progress[0])
                            : 0u;

                    printf(
                        "[idle] rank=%d idle_loops=%u step_cursor=%u next_search_idx=%u remaining_local=%u progress0=%u\n",
                        operation->rank,
                        debug_idle_loops,
                        shared_pipe.scheduler.step_cursor,
                        shared_pipe.scheduler.next_search_idx,
                        shared_pipe.scheduler.ready_count,
                        progress0);
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
            debug_idle_loops = 0u;
        }
        __syncthreads();

        exec::chunk_pipeline_wait_current_stage(&shared_pipe);
        __syncthreads();

        exec::chunk_pipeline_issue_current_apply(&shared_pipe);
        __syncthreads();

        if (threadIdx.x == 0) {
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

#endif
}

void configure_endpoint_persistent_kernel_smem(
    int device,
    size_t dynamic_smem_bytes) {
    system::runtime::set_device(device);

    cudaDeviceProp prop{};
    system::runtime::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties(endpoint persistent)");

#if OOVERLAP_ENDPOINT_PROFILE_SCHEDULER
    const size_t total_smem_bytes = dynamic_smem_bytes;
#else
    const size_t total_smem_bytes =
        dynamic_smem_bytes + kEndpointPersistentStaticSharedBytes;
#endif

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
#if OOVERLAP_ENDPOINT_PROFILE_SCHEDULER
    return 0;
#else
    return kEndpointPersistentChunkBytes * kPersistentStageDepth;
#endif
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
