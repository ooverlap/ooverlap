#include "comm/fast_multi_gpu_allreduce_sm90.h"

#include "comm/kernels/multi_gpu_ready_signal.cuh"
#include "comm/pipeline/pipeline_tma_reduce.h"
#include "ooverlap/sync/sync.cuh"
#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace ooverlap {
namespace {

constexpr int kFastAllreduceThreads = 32;
constexpr int kFastAllreduceScratchSlots = 257;
constexpr int kReadySignalPhaseStride = 1024;
constexpr int kLoadedReadyPhase = 1;
constexpr int kFanoutDoneReadyPhase = 2;
constexpr int kFastAllreduceMaxPeers =
    TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS;

struct FastAllreduceKernelParams {
    const void* local_ptr = nullptr;
    void* peer_ptrs[kFastAllreduceMaxPeers] = {};

    int* local_ready_signal = nullptr;
    const int* peer_ready_signals[kFastAllreduceMaxPeers] = {};

    size_t bytes = 0;
    int peer_count = 0;
    int collective_epoch = 0;
};

struct FastAllreduceBarrierScratch {
    unsigned int* counter = nullptr;
    unsigned int* last_value = nullptr;
};

cudaError_t allocate_zeroed_fast_allreduce_counter(
    int device,
    unsigned int** out) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }

    *out = nullptr;

    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        return error;
    }

    unsigned int* counter = nullptr;
    error = cudaMalloc(
        reinterpret_cast<void**>(&counter),
        sizeof(unsigned int));
    if (error != cudaSuccess) {
        return error;
    }

    error = cudaMemset(counter, 0, sizeof(unsigned int));
    if (error != cudaSuccess) {
        (void)cudaFree(counter);
        return error;
    }

    *out = counter;
    return cudaSuccess;
}

cudaError_t get_fast_allreduce_barrier_scratch(
    int device,
    int scratch_index,
    FastAllreduceBarrierScratch* out) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }

    *out = FastAllreduceBarrierScratch{};

    if (device < 0 || device >= 32) {
        return cudaErrorInvalidDevice;
    }

    const int storage_index = scratch_index == -1 ? 0 : scratch_index;

    if (storage_index < 0 ||
        storage_index >= kFastAllreduceScratchSlots) {
        return cudaErrorInvalidValue;
    }

    static std::mutex mutex;
    static unsigned int*
        counters[32][kFastAllreduceScratchSlots] = {};
    static unsigned int
        last_values[32][kFastAllreduceScratchSlots] = {};
    static std::atomic<bool>
        ready[32][kFastAllreduceScratchSlots] = {};

    if (ready[device][storage_index].load(std::memory_order_acquire)) {
        out->counter = counters[device][storage_index];
        out->last_value = &last_values[device][storage_index];
        return out->counter != nullptr
            ? cudaSuccess
            : cudaErrorInvalidValue;
    }

    std::lock_guard<std::mutex> lock(mutex);

    unsigned int*& counter =
        counters[device][storage_index];

    if (counter == nullptr) {
        const cudaError_t error =
            allocate_zeroed_fast_allreduce_counter(
                device,
                &counter);
        if (error != cudaSuccess) {
            return error;
        }
    }

    ready[device][storage_index].store(
        true,
        std::memory_order_release);

    out->counter = counter;
    out->last_value = &last_values[device][storage_index];
    return cudaSuccess;
}

__device__ __forceinline__ void wait_fast_allreduce_counter(
    unsigned int* counter,
    unsigned int target) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> state(*counter);

    while (state.load(cuda::memory_order_acquire) < target) {
        __nanosleep(64);
    }
}

__device__ __forceinline__ void arrive_and_wait_fast_allreduce_counter(
    unsigned int* counter,
    unsigned int target) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> state(*counter);
    state.fetch_add(1u, cuda::memory_order_acq_rel);

    while (state.load(cuda::memory_order_acquire) < target) {
        __nanosleep(64);
    }
}

__device__ __forceinline__ void advance_fast_allreduce_counter(
    unsigned int* counter) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> state(*counter);
    state.fetch_add(1u, cuda::memory_order_release);
}

__device__ __forceinline__ int fast_allreduce_ready_value(
    int collective_epoch,
    int phase) {
    return collective_epoch * kReadySignalPhaseStride + phase;
}

template <typename ReduceOp>
__global__ void fast_multi_gpu_allreduce_kernel_sm90(
    FastAllreduceKernelParams params,
    unsigned int* grid_counter,
    unsigned int counter_base) {
    if (threadIdx.x != 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_chunk =
        reinterpret_cast<unsigned char*>(shared_storage_u4);
    __shared__ sync::semaphore load_barrier;

    const size_t chunk_offset =
        static_cast<size_t>(blockIdx.x) *
        static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES);

    if (chunk_offset >= params.bytes) {
        return;
    }

    const size_t remaining = params.bytes - chunk_offset;
    const uint32_t chunk_bytes =
        static_cast<uint32_t>(
            remaining <
                    static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES)
                ? remaining
                : static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES));

    const unsigned char* local_src =
        reinterpret_cast<const unsigned char*>(params.local_ptr) +
        chunk_offset;

    sync::init_semaphore(load_barrier, 1);
    tma::expect_bytes(load_barrier, chunk_bytes);
    tma::load_async(
        shared_chunk,
        local_src,
        chunk_bytes,
        load_barrier);
    sync::wait(load_barrier, 0);

    const unsigned int cta_count =
        static_cast<unsigned int>(gridDim.x);
    const unsigned int loaded_target =
        counter_base + cta_count;

    arrive_and_wait_fast_allreduce_counter(
        grid_counter,
        loaded_target);

    const unsigned int peers_loaded_target =
        loaded_target + 1u;

    if (blockIdx.x == 0) {
        const int ready_value =
            fast_allreduce_ready_value(
                params.collective_epoch,
                kLoadedReadyPhase);

        comm::kernels::publish_ready_signal(
            params.local_ready_signal,
            ready_value,
            comm::kernels::MultiGpuReadySignalProtocol::
                DeviceMemoryStoreRelease);

        for (int peer_idx = 0;
             peer_idx < params.peer_count;
             ++peer_idx) {
            comm::kernels::wait_until_ready_signal_at_least(
                params.peer_ready_signals[peer_idx],
                ready_value,
                64);
        }

        advance_fast_allreduce_counter(grid_counter);
    }

    wait_fast_allreduce_counter(
        grid_counter,
        peers_loaded_target);

    tma::reduce_fence_proxy_async_shared_cta();

    for (int peer_idx = 0;
         peer_idx < params.peer_count;
         ++peer_idx) {
        unsigned char* peer_dst =
            reinterpret_cast<unsigned char*>(params.peer_ptrs[peer_idx]) +
            chunk_offset;

        ReduceOp::template issue_bulk_op_nofence<
            tma::TmaReduceScope::Default>(
                peer_dst,
                shared_chunk,
                chunk_bytes);
    }

    tma::reduce_commit_group();
    tma::reduce_async_wait<0>();
}

template <typename ReduceOp>
cudaError_t launch_fast_multi_gpu_allreduce_typed(
    const comm::api::FastAllreduceLaunchState& launch,
    cudaStream_t stream,
    int scratch_index) {
    if (launch.local_ptr == nullptr ||
        launch.local_ready_signal == nullptr ||
        launch.peer_count <= 0 ||
        launch.peer_count > kFastAllreduceMaxPeers ||
        launch.bytes == 0 ||
        launch.bytes > kFastAllreduceMaxBytes ||
        (launch.bytes % 16) != 0 ||
        launch.collective_epoch <= 0) {
        return cudaErrorInvalidValue;
    }

    FastAllreduceKernelParams params{};
    params.local_ptr = launch.local_ptr;
    params.local_ready_signal = launch.local_ready_signal;
    params.bytes = launch.bytes;
    params.peer_count = launch.peer_count;
    params.collective_epoch = launch.collective_epoch;

    for (int peer_idx = 0;
         peer_idx < launch.peer_count;
         ++peer_idx) {
        if (launch.peer_ptrs[peer_idx] == nullptr ||
            launch.peer_ready_signals[peer_idx] == nullptr) {
            return cudaErrorInvalidValue;
        }

        params.peer_ptrs[peer_idx] =
            launch.peer_ptrs[peer_idx];
        params.peer_ready_signals[peer_idx] =
            launch.peer_ready_signals[peer_idx];
    }

    const int num_ctas =
        static_cast<int>(
            (launch.bytes +
             static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES) - 1) /
            static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES));

    if (num_ctas <= 0 || num_ctas > TMA_TWO_GPU_PEER_MAX_CTAS) {
        return cudaErrorInvalidConfiguration;
    }

    cudaError_t error = cudaSetDevice(launch.local_device);
    if (error != cudaSuccess) {
        return error;
    }

    FastAllreduceBarrierScratch scratch{};
    error =
        get_fast_allreduce_barrier_scratch(
            launch.local_device,
            scratch_index,
            &scratch);

    if (error != cudaSuccess ||
        scratch.counter == nullptr ||
        scratch.last_value == nullptr) {
        return error != cudaSuccess
            ? error
            : cudaErrorInvalidValue;
    }

    const unsigned int counter_base =
        *scratch.last_value;
    const unsigned int counter_final =
        counter_base +
        static_cast<unsigned int>(num_ctas + 1);

    fast_multi_gpu_allreduce_kernel_sm90<ReduceOp><<<
        num_ctas,
        kFastAllreduceThreads,
        TMA_TWO_GPU_PEER_SMALL_TASK_BYTES,
        stream>>>(
            params,
            scratch.counter,
            counter_base);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }

    *scratch.last_value = counter_final;
    return cudaSuccess;
}

} // namespace

cudaError_t enqueue_fast_multi_gpu_allreduce_rank_sm90(
    const comm::api::FastAllreduceLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    int scratch_index) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_fast_multi_gpu_allreduce_typed<
                comm::pipeline::PipelineReduceAddNoFtzF16>(
                    launch,
                    stream,
                    scratch_index);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_fast_multi_gpu_allreduce_typed<
                comm::pipeline::PipelineReduceMinF16>(
                    launch,
                    stream,
                    scratch_index);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_fast_multi_gpu_allreduce_typed<
                comm::pipeline::PipelineReduceMaxF16>(
                    launch,
                    stream,
                    scratch_index);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_fast_multi_gpu_allreduce_typed<
                comm::pipeline::PipelineReduceAddBF16>(
                    launch,
                    stream,
                    scratch_index);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_fast_multi_gpu_allreduce_typed<
                comm::pipeline::PipelineReduceMinBF16>(
                    launch,
                    stream,
                    scratch_index);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_fast_multi_gpu_allreduce_typed<
                comm::pipeline::PipelineReduceMaxBF16>(
                    launch,
                    stream,
                    scratch_index);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32 &&
        (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM)) {
        return launch_fast_multi_gpu_allreduce_typed<
            comm::pipeline::PipelineReduceAddF32>(
                launch,
                stream,
                scratch_index);
    }

    return cudaErrorInvalidValue;
}

} // namespace ooverlap
