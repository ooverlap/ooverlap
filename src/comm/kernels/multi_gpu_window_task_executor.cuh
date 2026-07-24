#pragma once

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/multi_gpu_ready_signal.cuh"
#include "comm/kernels/window_task_executor.cuh"
#include "comm/plan/transfer_plan.h"
#include "comm/plan/window_plan.cuh"
#include "comm/plan/plan_params.cuh"
#include "comm/tma_variant_config.h"
#include "comm/params.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <driver_types.h>
#include <mutex>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace kernels {

/*
 * OOVERLAP_ALL_COLLECTIVES_PLAN_BY_VALUE_V1
 *
 * The lowered WindowTaskExecutorPlan is captured in CUDA kernel parameter
 * storage at launch. __grid_constant__ prevents nvcc from creating a private
 * per-thread copy when plan.tasks is passed by address to the stripe executor.
 */
template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxTasks,
    int MaxPeers,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = TMA_TWO_GPU_PEER_SMALL_TASK_BYTES>
__global__ void multi_gpu_window_task_executor_kernel_sm90(
    const __grid_constant__
        comm::plan::WindowTaskExecutorPlan<MaxTasks> plan,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch,
    unsigned int* cta_barrier_counter,
    unsigned int cta_barrier_start) {

    using Variant = comm::TmaPipelineVariant<
        ChunkBytes,
        StageDepth,
        FillDepth,
        LoadFillDepth,
        SmallTaskBytes>;

    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[Variant::barrier_count];

    if (cta_barrier_counter != nullptr) {
        if (threadIdx.x == 0) {
            if (blockIdx.x == 0) {
                const int entry_ready_value =
                    collective_epoch * comm::plan::kReadySignalPhaseStride;

                publish_ready_signal(
                    local_ready_signal,
                    entry_ready_value,
                    ready_plan.protocol);

                for (int peer_idx = 0;
                     peer_idx < ready_plan.peer_count;
                     ++peer_idx) {
                    wait_until_ready_signal_at_least(
                        ready_plan.peer_ready_signals[peer_idx],
                        entry_ready_value,
                        ready_plan.poll_sleep_cycles);
                }

                advance_cta_barrier_counter(cta_barrier_counter, 1u);
            } else {
                wait_until_cta_barrier_counter_at_least(
                    cta_barrier_counter,
                    cta_barrier_start);
            }
        }

        __syncthreads();
    }

    execute_window_task_stripe<
        Variant::stage_depth,
        FillDepth,
        Variant::chunk_bytes,
        ReduceApply,
        uint4,
        TMA_TWO_GPU_PEER_FAST_COPY_UNROLL,
        LoadFillDepth,
        SmallTaskBytes>(
            plan.tasks,
            plan.total_tasks,
            plan.tasks_per_cta,
            static_cast<int>(blockIdx.x),
            shared_raw,
            barriers,
            cta_barrier_counter);
}


/*
 * The plan itself is now passed by value. All-reduce still needs a persistent
 * device counter because its lowered Barrier tasks use absolute counter values
 * across stream-ordered launches. This cache owns only that small state.
 */
constexpr int kOoCtaBarrierScratchSlots = 257;

inline cudaError_t allocate_zeroed_cta_barrier_counter(
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
        cudaFree(counter);
        return error;
    }

    *out = counter;
    return cudaSuccess;
}

struct CtaBarrierScratch {
    unsigned int* counter = nullptr;
    unsigned int* last_value = nullptr;
};

inline cudaError_t get_cached_cta_barrier_scratch(
    int device,
    int scratch_index,
    CtaBarrierScratch* out) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }

    *out = CtaBarrierScratch{};

    if (device < 0 || device >= 32) {
        return cudaErrorInvalidDevice;
    }

    const int storage_index = scratch_index == -1 ? 0 : scratch_index;

    if (storage_index < 0 ||
        storage_index >= kOoCtaBarrierScratchSlots) {
        return cudaErrorInvalidValue;
    }

    static std::mutex mutex;
    static unsigned int* counters[32][kOoCtaBarrierScratchSlots] = {};
    static unsigned int last_values[32][kOoCtaBarrierScratchSlots] = {};
    static std::atomic<bool> ready[32][kOoCtaBarrierScratchSlots] = {};

    if (ready[device][storage_index].load(std::memory_order_acquire)) {
        out->counter = counters[device][storage_index];
        out->last_value = &last_values[device][storage_index];
        return out->counter != nullptr
            ? cudaSuccess
            : cudaErrorInvalidValue;
    }

    std::lock_guard<std::mutex> lock(mutex);

    unsigned int*& counter = counters[device][storage_index];

    if (counter == nullptr) {
        const cudaError_t error =
            allocate_zeroed_cta_barrier_counter(device, &counter);
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


template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxTasks,
    int MaxPeers,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = TMA_TWO_GPU_PEER_SMALL_TASK_BYTES>
cudaError_t launch_multi_gpu_window_task_executor_sm90(
    const comm::plan::WindowTaskExecutorPlan<MaxTasks>& window_plan,
    int num_blocks,
    int threads,
    size_t dynamic_shared_bytes,
    cudaStream_t stream,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch,
    unsigned int* cta_barrier_counter = nullptr,
    unsigned int cta_barrier_start = 0u) {
    if (num_blocks <= 0 || threads <= 0) {
        return cudaSuccess;
    }

    multi_gpu_window_task_executor_kernel_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxTasks,
        MaxPeers,
        FillDepth,
        LoadFillDepth,
        SmallTaskBytes><<<
            num_blocks,
            threads,
            dynamic_shared_bytes,
            stream>>>(
                window_plan,
                local_ready_signal,
                ready_plan,
                collective_epoch,
                cta_barrier_counter,
                cta_barrier_start);

    return cudaGetLastError();
}


template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxTasks,
    int MaxPeers,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = TMA_TWO_GPU_PEER_SMALL_TASK_BYTES>
void configure_multi_gpu_window_task_executor_once(
    int device,
    const char* error_prefix) {
    using Variant = comm::TmaPipelineVariant<
        ChunkBytes,
        StageDepth,
        FillDepth,
        LoadFillDepth,
        SmallTaskBytes>;

    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(SmallTaskBytes >= 0, "SmallTaskBytes must be >= 0");

    if (device < 0 || device >= 32) {
        throw std::runtime_error("invalid device");
    }

    static std::mutex mutex;
    static unsigned int configured_mask = 0;

    const unsigned int bit =
        1u << static_cast<unsigned int>(device);

    if ((configured_mask & bit) != 0u) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);

    if ((configured_mask & bit) != 0u) {
        return;
    }

    constexpr size_t dynamic_smem_bytes =
        Variant::dynamic_shared_bytes;
    constexpr size_t total_smem_bytes =
        Variant::total_shared_bytes;

    system::runtime::set_device(device);

    cudaDeviceProp prop{};

    system::runtime::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties");

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            error_prefix != nullptr
                ? error_prefix
                : "requested shared memory exceeds opt-in limit");
    }

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                multi_gpu_window_task_executor_kernel_sm90<
                    ReduceApply,
                    ChunkBytes,
                    StageDepth,
                    MaxTasks,
                    MaxPeers,
                    FillDepth,
                    LoadFillDepth,
                    SmallTaskBytes>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    }

    system::runtime::check_cuda(
        cudaFuncSetAttribute(
            multi_gpu_window_task_executor_kernel_sm90<
                ReduceApply,
                ChunkBytes,
                StageDepth,
                MaxTasks,
                MaxPeers,
                FillDepth,
                LoadFillDepth,
                SmallTaskBytes>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");

    configured_mask |= bit;
}


/* OOVERLAP_BY_VALUE_CAPACITY_DISPATCH_V1 */
template <
    int ByValueMaxTasks,
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxSourceTasks,
    int MaxPeers,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = TMA_TWO_GPU_PEER_SMALL_TASK_BYTES>
cudaError_t pack_configure_launch_multi_gpu_window_task_executor_sm90(
    const comm::plan::WindowTaskExecutorPlan<MaxSourceTasks>& window_plan,
    int num_blocks,
    int threads,
    size_t dynamic_shared_bytes,
    cudaStream_t stream,
    int device,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch,
    const char* error_prefix,
    unsigned int* cta_barrier_counter = nullptr,
    unsigned int cta_barrier_start = 0u) {
    using ByValuePlan =
        comm::plan::WindowTaskExecutorPlan<ByValueMaxTasks>;

    static_assert(
        sizeof(ByValuePlan) <= 16 * 1024,
        "by-value WindowTask plan unexpectedly exceeds 16 KiB");

    if (window_plan.total_tasks <= 0) {
        return cudaSuccess;
    }

    if (window_plan.total_tasks > ByValueMaxTasks) {
        return cudaErrorInvalidConfiguration;
    }

    static thread_local ByValuePlan by_value_plan;

    if (!comm::plan::window_task_executor_plan_pack(
            window_plan,
            &by_value_plan)) {
        return cudaErrorInvalidConfiguration;
    }

    configure_multi_gpu_window_task_executor_once<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        ByValueMaxTasks,
        MaxPeers,
        FillDepth,
        LoadFillDepth,
        SmallTaskBytes>(
            device,
            error_prefix);

    return launch_multi_gpu_window_task_executor_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        ByValueMaxTasks,
        MaxPeers,
        FillDepth,
        LoadFillDepth,
        SmallTaskBytes>(
            by_value_plan,
            num_blocks,
            threads,
            dynamic_shared_bytes,
            stream,
            local_ready_signal,
            ready_plan,
            collective_epoch,
            cta_barrier_counter,
            cta_barrier_start);
}


template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxSourceTasks,
    int MaxPeers,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = TMA_TWO_GPU_PEER_SMALL_TASK_BYTES>
cudaError_t dispatch_multi_gpu_window_task_executor_by_value_sm90(
    const comm::plan::WindowTaskExecutorPlan<MaxSourceTasks>& window_plan,
    int num_blocks,
    int threads,
    size_t dynamic_shared_bytes,
    cudaStream_t stream,
    int device,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch,
    const char* error_prefix,
    unsigned int* cta_barrier_counter = nullptr,
    unsigned int cta_barrier_start = 0u) {
    if (window_plan.total_tasks < 0 ||
        window_plan.total_tasks > MaxSourceTasks) {
        return cudaErrorInvalidConfiguration;
    }

    if (window_plan.total_tasks <=
        comm::plan::kTmaMultiGpuByValueCapacity16) {
        return pack_configure_launch_multi_gpu_window_task_executor_sm90<
            comm::plan::kTmaMultiGpuByValueCapacity16,
            ReduceApply,
            ChunkBytes,
            StageDepth,
            MaxSourceTasks,
            MaxPeers,
            FillDepth,
            LoadFillDepth,
            SmallTaskBytes>(
                window_plan,
                num_blocks,
                threads,
                dynamic_shared_bytes,
                stream,
                device,
                local_ready_signal,
                ready_plan,
                collective_epoch,
                error_prefix,
                cta_barrier_counter,
                cta_barrier_start);
    }

    if (window_plan.total_tasks <=
        comm::plan::kTmaMultiGpuByValueCapacity32) {
        return pack_configure_launch_multi_gpu_window_task_executor_sm90<
            comm::plan::kTmaMultiGpuByValueCapacity32,
            ReduceApply,
            ChunkBytes,
            StageDepth,
            MaxSourceTasks,
            MaxPeers,
            FillDepth,
            LoadFillDepth,
            SmallTaskBytes>(
                window_plan,
                num_blocks,
                threads,
                dynamic_shared_bytes,
                stream,
                device,
                local_ready_signal,
                ready_plan,
                collective_epoch,
                error_prefix,
                cta_barrier_counter,
                cta_barrier_start);
    }

    if (window_plan.total_tasks <=
        comm::plan::kTmaMultiGpuByValueCapacity64) {
        return pack_configure_launch_multi_gpu_window_task_executor_sm90<
            comm::plan::kTmaMultiGpuByValueCapacity64,
            ReduceApply,
            ChunkBytes,
            StageDepth,
            MaxSourceTasks,
            MaxPeers,
            FillDepth,
            LoadFillDepth,
            SmallTaskBytes>(
                window_plan,
                num_blocks,
                threads,
                dynamic_shared_bytes,
                stream,
                device,
                local_ready_signal,
                ready_plan,
                collective_epoch,
                error_prefix,
                cta_barrier_counter,
                cta_barrier_start);
    }

    if (window_plan.total_tasks <=
        comm::plan::kTmaMultiGpuByValueCapacity128) {
        return pack_configure_launch_multi_gpu_window_task_executor_sm90<
            comm::plan::kTmaMultiGpuByValueCapacity128,
            ReduceApply,
            ChunkBytes,
            StageDepth,
            MaxSourceTasks,
            MaxPeers,
            FillDepth,
            LoadFillDepth,
            SmallTaskBytes>(
                window_plan,
                num_blocks,
                threads,
                dynamic_shared_bytes,
                stream,
                device,
                local_ready_signal,
                ready_plan,
                collective_epoch,
                error_prefix,
                cta_barrier_counter,
                cta_barrier_start);
    }

    return cudaErrorInvalidConfiguration;
}

} // namespace kernels
} // namespace comm
} // namespace ooverlap
