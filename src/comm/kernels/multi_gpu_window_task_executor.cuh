#pragma once

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/multi_gpu_ready_signal.cuh"
#include "comm/kernels/window_task_executor.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"

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
    int LoadFillDepth = FillDepth>
__global__ void multi_gpu_window_task_executor_kernel_sm90(
    comm::plan::WindowTaskExecutorPlan<MaxTasks> plan,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

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
        LoadFillDepth>(
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
    int LoadFillDepth = FillDepth>
void configure_multi_gpu_window_task_executor_once(
    int device,
    const char* error_prefix) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");

    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes = Variant::dynamic_shared_bytes;
    const size_t total_smem_bytes = Variant::total_shared_bytes;

    std::lock_guard<std::mutex> lock(mutex);

    auto it = cache.find(device);

    if (it != cache.end() &&
        it->second.configured &&
        it->second.dynamic_smem_bytes == dynamic_smem_bytes) {
        return;
    }

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
                    LoadFillDepth>,
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
                LoadFillDepth>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");

    cache[device] = {true, dynamic_smem_bytes};
}

} // namespace kernels
} // namespace comm
} // namespace ooverlap
