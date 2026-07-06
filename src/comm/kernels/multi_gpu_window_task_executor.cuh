#pragma once

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/multi_gpu_ready_signal.cuh"
#include "comm/kernels/window_task_executor.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ooverlap {
namespace comm {
namespace kernels {

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
    comm::plan::WindowTaskExecutorPlan<MaxTasks> plan,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch) {

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

    wait_for_multi_gpu_collective_ready(
        local_ready_signal,
        ready_plan,
        collective_epoch);

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[Variant::barrier_count];

    /*
     * All-gather uses only copy tasks, but the task executor template still
     * needs a ReduceApply type. The reduce path is not used for all-gather.
     */
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
            barriers);
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

    /*
     * Fast path: no map lookup, no lock, no CUDA calls.
     */
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

} // namespace kernels
} // namespace comm
} // namespace ooverlap
