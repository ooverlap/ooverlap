#include "comm/endpoint_persistent_kernel.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace {

static constexpr size_t kEndpointPersistentStaticSharedBytes =
    static_cast<size_t>(kEndpointPersistentStageDepth) * sizeof(sync::semaphore) +
    sizeof(exec::ChunkSchedulerScratch<kEndpointPersistentWatchThreads>) +
    sizeof(int);

__device__ __forceinline__ bool endpoint_persistent_should_stop(
    const uint32_t* stop_flag) {
    return (*reinterpret_cast<volatile const uint32_t*>(stop_flag)) != 0u;
}

/*__global__ void endpoint_persistent_kernel_sm90(*/
    /*DeviceEndpointRuntime runtime,*/
    /*const uint32_t* stop_flag) {*/
    /*if (blockIdx.x != 0) {*/
        /*return;*/
    /*}*/

    /*extern __shared__ uint4 shared_storage_u4[];*/
    /*unsigned char* shared_raw =*/
        /*reinterpret_cast<unsigned char*>(shared_storage_u4);*/

    /*__shared__ sync::semaphore load_barriers[kEndpointPersistentStageDepth];*/
    /*__shared__ exec::ChunkSchedulerScratch<kEndpointPersistentWatchThreads> sched_scratch;*/
    /*__shared__ int shared_should_stop;*/

    /*EndpointPersistentScheduler scheduler{};*/
    /*exec::chunk_scheduler_init(*/
        /*&scheduler,*/
        /*runtime.scheduler_bindings,*/
        /*runtime.num_scheduler_bindings,*/
        /*kEndpointPersistentChunkBytes,*/
        /*&sched_scratch);*/

    /*EndpointPersistentPipeline pipe{};*/
    /*exec::chunk_pipeline_bind_stage_storage<*/
        /*kEndpointPersistentStageDepth,*/
        /*kEndpointPersistentChunkBytes>(*/
        /*&pipe,*/
        /*shared_raw,*/
        /*load_barriers);*/
    /*exec::chunk_pipeline_init(&pipe, &scheduler);*/

    /*while (true) {*/
        /*if (threadIdx.x == 0) {*/
            /*shared_should_stop = endpoint_persistent_should_stop(stop_flag) ? 1 : 0;*/
        /*}*/
        /*__syncthreads();*/

        /*if (shared_should_stop) {*/
            /*return;*/
        /*}*/

        /*if (!exec::chunk_pipeline_has_current(&pipe)) {*/
            /*if (!exec::chunk_pipeline_try_prime(&pipe)) {*/
/*#if defined(__CUDA_ARCH__)*/
                /*if (threadIdx.x == 0) {*/
                    /*__nanosleep(256);*/
                /*}*/
/*#endif*/
                /*__syncthreads();*/
                /*continue;*/
            /*}*/
            /*__syncthreads();*/
        /*}*/

        /*exec::chunk_pipeline_wait_current_stage(&pipe);*/
        /*__syncthreads();*/

        /*exec::chunk_pipeline_issue_current_reduce(&pipe);*/
        /*exec::chunk_pipeline_schedule_next_load(&pipe);*/
        /*__syncthreads();*/

        /*exec::chunk_pipeline_finish_current_tail(&pipe);*/
        /*__syncthreads();*/

        /*exec::chunk_pipeline_advance(&pipe);*/
        /*__syncthreads();*/
    /*}*/
/*}*/

/*__global__ void endpoint_persistent_kernel_sm90(*/
    /*DeviceEndpointRuntime runtime,*/
    /*const uint32_t* stop_flag) {*/
    /*if (blockIdx.x != 0) {*/
        /*return;*/
    /*}*/

    /*__shared__ exec::ChunkSchedulerScratch<kEndpointPersistentWatchThreads> sched_scratch;*/
    /*__shared__ int shared_should_stop;*/

    /*EndpointPersistentScheduler scheduler{};*/
    /*exec::chunk_scheduler_init(*/
        /*&scheduler,*/
        /*runtime.scheduler_bindings,*/
        /*runtime.num_scheduler_bindings,*/
        /*kEndpointPersistentChunkBytes,*/
        /*&sched_scratch);*/

    /*while (true) {*/
        /*if (threadIdx.x == 0) {*/
            /*shared_should_stop =*/
                /*(*reinterpret_cast<volatile const uint32_t*>(stop_flag) != 0u) ? 1 : 0;*/
        /*}*/
        /*__syncthreads();*/

        /*if (shared_should_stop) {*/
            /*return;*/
        /*}*/

        /*if (!exec::chunk_scheduler_try_prime_current(&scheduler)) {*/
/*#if defined(__CUDA_ARCH__)*/
            /*if (threadIdx.x == 0) {*/
                /*__nanosleep(256);*/
            /*}*/
/*#endif*/
            /*__syncthreads();*/
            /*continue;*/
        /*}*/

        /*__syncthreads();*/

        /*// Debug mode: consume the work without pipeline/TMA.*/
        
        /*exec::chunk_scheduler_advance(&scheduler);*/

        /*__syncthreads();*/
    /*}*/
/*}*/

__global__ void endpoint_persistent_kernel_sm90(
    DeviceEndpointRuntime runtime,
    const uint32_t* stop_flag) {
  if (blockIdx.x != 0) return;

  __shared__ int should_stop;
  __shared__ int has_work;
  __shared__ int active_binding_idx;
  __shared__ comm::exec::Chunk current_chunk;

  while (true) {
    if (threadIdx.x == 0) {
      should_stop = (*reinterpret_cast<volatile const uint32_t*>(stop_flag) != 0u) ? 1 : 0;
    }
    __syncthreads();
    if (should_stop) return;

    if (threadIdx.x == 0) {
      comm::exec::ChunkScheduler<1> sched{};
      comm::exec::chunk_scheduler_init<1>(
          &sched,
          runtime.scheduler_bindings,
          runtime.num_scheduler_bindings,
          comm::kEndpointPersistentChunkBytes);

      if (!comm::exec::chunk_scheduler_try_prime_current(&sched)) {
        has_work = 0;
      } else {
        has_work = 1;
        active_binding_idx = sched.active_binding_idx;
        current_chunk = *comm::exec::chunk_scheduler_current(&sched);
      }
    }
    __syncthreads();

    if (!has_work) {
#if defined(__CUDA_ARCH__)
      if (threadIdx.x == 0) __nanosleep(256);
#endif
      __syncthreads();
      continue;
    }

    // direct reduction for now
    const auto& span = current_chunk.tile_spans[0];
    half* dst = reinterpret_cast<half*>(current_chunk.dst);
    const half* src = reinterpret_cast<const half*>(span.src);
    const size_t elems = current_chunk.bytes / sizeof(half);

    for (size_t i = threadIdx.x; i < elems; i += blockDim.x) {
      const float oldv = __half2float(dst[i]);
      const float addv = __half2float(src[i]);
      dst[i] = __float2half_rn(oldv + addv);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      auto* q = runtime.scheduler_bindings[active_binding_idx].queue;
      *q->head = current_chunk.span_ticket + current_chunk.num_tile_spans;
      __threadfence();
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
    cudaStreamCreateWithPriority(
        &ctl->control_stream,
        cudaStreamNonBlocking,
        5),
    "cudaStreamCreateWithPriority(producer_stream)");

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

    // Any nonzero value is fine; the kernel only checks != 0.
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
    /*return static_cast<size_t>(kEndpointPersistentStageDepth) **/
           /*kEndpointPersistentChunkBytes;*/
}

cudaError_t launch_endpoint_persistent_kernel_sm90(
    const DeviceEndpointRuntime* runtime,
    const EndpointPersistentControl* control,
    cudaStream_t stream) {
    if (!device_endpoint_runtime_is_valid(runtime)) {
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
        control->stop_flag);

    return cudaGetLastError();
}

} // namespace comm
} // namespace ooverlap
