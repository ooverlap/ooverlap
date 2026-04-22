#include "comm/endpoint_persistent_kernel.h"

#include "comm/exec/pipeline_load.h"
#include "comm/exec/pipeline_reduce.h"
#include "comm/exec/pipeline_stage.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/tma/tma_reduce.cuh"

#include <cmath>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace {

__device__ __forceinline__ float endpoint_debug_half0(
    const unsigned char* ptr) {
    return __half2float(*reinterpret_cast<const half*>(ptr));
}

__device__ __forceinline__ unsigned long long endpoint_debug_ptr_u64(
    const void* ptr) {
    return static_cast<unsigned long long>(
        reinterpret_cast<uintptr_t>(ptr));
}

static constexpr size_t kEndpointPersistentStaticSharedBytes =
    sizeof(sync::semaphore) +
    sizeof(exec::Chunk) +
    sizeof(uint32_t) * 4 +
    sizeof(int) * 2;

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

__device__ __forceinline__ uint32_t endpoint_persistent_atomic_load_u32(
    volatile uint32_t* ptr) {
    return atomicAdd(
        reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(ptr)),
        0u);
}

__device__ __forceinline__ void endpoint_persistent_atomic_store_u32(
    volatile uint32_t* ptr,
    uint32_t value) {
    atomicExch(
        reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(ptr)),
        value);
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

__device__ __forceinline__ void endpoint_persistent_build_chunk_for_step(
    exec::Chunk* chunk,
    const collective::OperationDesc* op,
    uint32_t chunk_idx,
    uint32_t step) {
    exec::chunk_clear(chunk);

    chunk->src =
        collective::operation_desc_accum_chunk_ptr(op, chunk_idx);
    chunk->dst =
        collective::operation_desc_next_accum_chunk_ptr(op, chunk_idx);
    chunk->bytes =
        collective::operation_desc_chunk_bytes_at(op, chunk_idx);
    chunk->op_id = op->op_id;
    chunk->dst_rank = op->next_rank;
    chunk->user_tag = op->user_tag;
    chunk->chunk_idx = static_cast<int>(chunk_idx);
    chunk->range_offset_bytes =
        collective::operation_desc_chunk_offset_bytes(op, chunk_idx);
    chunk->op =
        collective::operation_desc_chunk_op_for_step(op, step);
}

__device__ __forceinline__ void endpoint_persistent_copy_smem_to_gmem(
    const exec::PipelineStage* stage) {
    if (stage == nullptr || !exec::chunk_is_valid(&stage->chunk)) {
        return;
    }

    unsigned char* dst = stage->chunk.dst;
    const unsigned char* src = stage->smem;
    const size_t bytes = stage->chunk.bytes;

    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ void endpoint_persistent_publish_to_next(
    const collective::OperationDesc* op,
    uint32_t chunk_idx,
    uint32_t current_step,
    uint32_t total_steps) {
    volatile uint32_t* next_done =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_next_done(op));
    volatile uint32_t* next_inbound =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_next_inbound_steps(op));
    volatile uint32_t* local_done =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_done(op));

    const uint32_t next_step = current_step + 1;
    const uint32_t reduce_steps =
        static_cast<uint32_t>(op->world_size - 1);

    // Make chunk data visible on the next rank before publishing tokens.
    __threadfence_system();

    if (current_step < reduce_steps) {
        // The step that lands on the owner makes the owner's local chunk final.
        if (next_step == reduce_steps) {
            endpoint_persistent_atomic_store_u32(&next_done[chunk_idx], 1u);
        }

        if (next_step < total_steps) {
            endpoint_persistent_atomic_store_u32(&next_inbound[chunk_idx], next_step);
        }
    } else {
        // All-gather/copy phase: the next rank now has a final copy too.
        endpoint_persistent_atomic_store_u32(&next_done[chunk_idx], 1u);

        if (next_step < total_steps) {
            endpoint_persistent_atomic_store_u32(&next_inbound[chunk_idx], next_step);
        }
    }

    // Final actor marks itself done as well.
    if (next_step >= total_steps) {
        endpoint_persistent_atomic_store_u32(&local_done[chunk_idx], 1u);
    }

    __threadfence_system();
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

    __shared__ sync::semaphore load_barrier;
    __shared__ exec::Chunk shared_chunk;
    __shared__ uint32_t shared_chunk_idx;
    __shared__ uint32_t shared_chunk_step;
    __shared__ int shared_has_work;
    __shared__ int shared_should_stop;

    collective::ChunkState* chunk_states =
        collective::operation_desc_chunk_states(operation);
    volatile uint32_t* inbound_steps =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_inbound_steps(operation));
    volatile uint32_t* done =
        reinterpret_cast<volatile uint32_t*>(
            collective::operation_desc_local_done(operation));

    const uint32_t total_steps =
        collective::operation_desc_total_ring_steps(operation);

    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x);
         idx < operation->num_chunks;
         idx += static_cast<uint32_t>(blockDim.x)) {
        endpoint_persistent_init_chunk_state(
            &chunk_states[idx],
            operation,
            idx);

        if (total_steps == 0) {
            endpoint_persistent_atomic_store_u32(&done[idx], 1u);
        }
    }
    __syncthreads();

    exec::PipelineTMALoad load_op{};
    exec::PipelineTMAReduceAddNoFtzF16 reduce_op{};

    while (true) {
        if (threadIdx.x == 0) {
            shared_should_stop = endpoint_persistent_should_stop(stop_flag) ? 1 : 0;
        }
        __syncthreads();

        if (shared_should_stop) {
            return;
        }

        if (threadIdx.x == 0) {
            shared_has_work = 0;

            for (uint32_t idx = 0; idx < operation->num_chunks; ++idx) {
                if (collective::chunk_state_is_in_flight(&chunk_states[idx])) {
                    continue;
                }

                const uint32_t step =
                    endpoint_persistent_atomic_load_u32(&inbound_steps[idx]);

                if (step == collective::kOperationInboundStepInvalid) {
                    continue;
                }
                if (step >= total_steps) {
                    continue;
                }

                const int actor =
                    collective::operation_desc_actor_rank_for_step(
                        operation,
                        idx,
                        step);
                if (actor != operation->rank) {
                    continue;
                }

                endpoint_persistent_build_chunk_for_step(
                    &shared_chunk,
                    operation,
                    idx,
                    step);

                if (idx == 0) {
                    const unsigned char* local_ptr =
                        collective::operation_desc_accum_chunk_ptr(operation, idx);
                    const unsigned char* next_ptr =
                        collective::operation_desc_next_accum_chunk_ptr(operation, idx);
                
                    const float local0 = endpoint_debug_half0(local_ptr);
                    const float next0 = endpoint_debug_half0(next_ptr);
                
                    printf(
                        "[pk pick] rank=%d step=%u chunk=%u op=%u src=0x%llx dst=0x%llx local0=%.6f next0_before=%.6f\n",
                        operation->rank,
                        step,
                        idx,
                        static_cast<unsigned>(shared_chunk.op),
                        endpoint_debug_ptr_u64(local_ptr),
                        endpoint_debug_ptr_u64(next_ptr),
                        local0,
                        next0);
                }

                shared_chunk_idx = idx;
                shared_chunk_step = step;
                shared_has_work = 1;

                chunk_states[idx].last_step_started = step;
                chunk_states[idx].flags |= collective::kChunkStateFlagInFlight;
                break;
            }
        }
        __syncthreads();

        if (!shared_has_work) {
#if defined(__CUDA_ARCH__)
            if (threadIdx.x == 0) {
                __nanosleep(256);
            }
#endif
            __syncthreads();
            continue;
        }

        exec::PipelineStage stage{};
        stage.smem = shared_raw;
        stage.load_barrier = &load_barrier;
        stage.chunk = shared_chunk;

        if (threadIdx.x == 0) {
            load_op.issue(&stage);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            load_op.wait_ready(&stage);
        }
        __syncthreads();

        if (stage.chunk.op == exec::ChunkOpKind::kReduceAddNoFtzF16) {
            if (threadIdx.x == 0) {
                reduce_op.issue_bulk(&stage);
            }
            __syncthreads();

            reduce_op.finish_tail(&stage);
            __syncthreads();

            if (threadIdx.x == 0) {
                tma::reduce_async_wait<0>();
            }
            __syncthreads();
        } else if (stage.chunk.op == exec::ChunkOpKind::kCopy) {
            endpoint_persistent_copy_smem_to_gmem(&stage);
            __syncthreads();
        } else {
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) {
            // Consume local inbound token only after this step is fully retired.
            endpoint_persistent_atomic_store_u32(
                &inbound_steps[shared_chunk_idx],
                collective::kOperationInboundStepInvalid);
            __threadfence_system();

            
            if (shared_chunk_idx == 0) {
                const unsigned char* local_ptr_after =
                    collective::operation_desc_accum_chunk_ptr(operation, shared_chunk_idx);
                const unsigned char* next_ptr_after =
                    collective::operation_desc_next_accum_chunk_ptr(operation, shared_chunk_idx);
            
                const float local0_after = endpoint_debug_half0(local_ptr_after);
                const float next0_after = endpoint_debug_half0(next_ptr_after);
            
                printf(
                    "[pk done] rank=%d step=%u chunk=%u src=0x%llx dst=0x%llx local0_after=%.6f next0_after=%.6f\n",
                    operation->rank,
                    shared_chunk_step,
                    shared_chunk_idx,
                    endpoint_debug_ptr_u64(local_ptr_after),
                    endpoint_debug_ptr_u64(next_ptr_after),
                    local0_after,
                    next0_after);
            }

            endpoint_persistent_publish_to_next(
                operation,
                shared_chunk_idx,
                shared_chunk_step,
                total_steps);

            if (shared_chunk_idx == 0) {
                volatile uint32_t* next_inbound =
                    reinterpret_cast<volatile uint32_t*>(
                        collective::operation_desc_next_inbound_steps(operation));
                volatile uint32_t* next_done =
                    reinterpret_cast<volatile uint32_t*>(
                        collective::operation_desc_next_done(operation));
                volatile uint32_t* local_done =
                    reinterpret_cast<volatile uint32_t*>(
                        collective::operation_desc_local_done(operation));
            
                printf(
                    "[pk pub ] rank=%d step=%u chunk=%u wrote next_inbound=%u next_done=%u local_done=%u\n",
                    operation->rank,
                    shared_chunk_step,
                    shared_chunk_idx,
                    next_inbound[shared_chunk_idx],
                    next_done[shared_chunk_idx],
                    local_done[shared_chunk_idx]);
            }

            chunk_states[shared_chunk_idx].last_step_completed = shared_chunk_step + 1u;
            chunk_states[shared_chunk_idx].flags &= ~collective::kChunkStateFlagInFlight;

            if (endpoint_persistent_atomic_load_u32(&done[shared_chunk_idx]) == 1u) {
                chunk_states[shared_chunk_idx].flags |= collective::kChunkStateFlagDone;
            }
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
    return kEndpointPersistentChunkBytes;
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
