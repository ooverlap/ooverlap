#pragma once

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/multi_gpu_ready_signal.cuh"
#include "comm/kernels/window_task_executor.cuh"
#include "comm/plan/transfer_plan.h"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/params.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <driver_types.h>
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
    const comm::plan::WindowTaskExecutorPlan<MaxTasks>* plan,
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

    if (plan == nullptr) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[Variant::barrier_count];

    const bool use_entry_ready =
        collective_epoch > 0 &&
        local_ready_signal != nullptr &&
        ready_plan.protocol != MultiGpuReadySignalProtocol::Disabled;

    if (cta_barrier_counter != nullptr) {
        if (threadIdx.x == 0) {
            if (blockIdx.x == 0) {
                if (use_entry_ready) {
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
                }

                advance_cta_barrier_counter(cta_barrier_counter, 1u);
            } else {
                wait_until_cta_barrier_counter_at_least(
                    cta_barrier_counter,
                    cta_barrier_start);
            }
        }

        __syncthreads();
    } else if (use_entry_ready) {
        const int entry_ready_value =
            collective_epoch * comm::plan::kReadySignalPhaseStride;

        if (threadIdx.x == 0) {
            if (blockIdx.x == 0) {
                publish_ready_signal(
                    local_ready_signal,
                    entry_ready_value,
                    ready_plan.protocol);
            }

            for (int peer_idx = 0;
                 peer_idx < ready_plan.peer_count;
                 ++peer_idx) {
                wait_until_ready_signal_at_least(
                    ready_plan.peer_ready_signals[peer_idx],
                    entry_ready_value,
                    ready_plan.poll_sleep_cycles);
            }
        }

        __syncthreads();
    }

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
            plan->tasks,
            plan->total_tasks,
            plan->tasks_per_cta,
            static_cast<int>(blockIdx.x),
            shared_raw,
            barriers,
            cta_barrier_counter);
}


/*
 * OOVERLAP_MAPPED_WINDOW_PLAN_SCRATCH_PATCH
 * OOVERLAP_ROUND_ROBIN_PLAN_SCRATCH_RING_V1
 * OOVERLAP_PLAN_SCRATCH_MINUS_ONE_FAST_PATH_V1
 *
 * scratch_index == -1:
 *   - uses storage entry 0
 *   - never creates, queries, synchronizes, or returns a completion event
 *   - after first initialization, returns through an atomic lock-free fast path
 *
 * scratch_index >= 0:
 *   - preserves the existing event-protected behavior
 *   - the CPU may rewrite mapped host memory only after the prior kernel using
 *     that exact entry has completed
 *
 * The -1 path intentionally removes the lifetime protection for entry 0. It is
 * an experimental path and must only be used when the caller guarantees that
 * the CPU will not rewrite the plan while a prior kernel is still reading it.
 */
constexpr int kOoMappedWindowPlanScratchSlots = 257;

inline cudaError_t allocate_zeroed_cta_barrier_counter(
    int device,
    unsigned int** out) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }

    *out = nullptr;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return err;
    }

    unsigned int* counter = nullptr;
    err = cudaMalloc(
        reinterpret_cast<void**>(&counter),
        sizeof(unsigned int));
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMemset(counter, 0, sizeof(unsigned int));
    if (err != cudaSuccess) {
        cudaFree(counter);
        return err;
    }

    *out = counter;
    return cudaSuccess;
}

template <int MaxTasks>
struct WindowPlanMappedScratch {
    comm::plan::WindowTaskExecutorPlan<MaxTasks>* host_plan = nullptr;
    const comm::plan::WindowTaskExecutorPlan<MaxTasks>* device_plan = nullptr;
    unsigned int* cta_barrier_counter = nullptr;
    unsigned int* cta_barrier_last_value = nullptr;
    cudaEvent_t completion_event = nullptr;
};

template <int MaxTasks>
cudaError_t get_mapped_window_plan_scratch(
    int device,
    int scratch_index,
    WindowPlanMappedScratch<MaxTasks>* out) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }

    *out = WindowPlanMappedScratch<MaxTasks>{};

    if (device < 0 || device >= 32) {
        return cudaErrorInvalidDevice;
    }

    if (scratch_index < -1 ||
        scratch_index >= kOoMappedWindowPlanScratchSlots) {
        return cudaErrorInvalidValue;
    }

    static std::mutex mutex;
    static void* host_buffers[32][kOoMappedWindowPlanScratchSlots] = {};
    static void* device_buffers[32][kOoMappedWindowPlanScratchSlots] = {};
    static unsigned int* cta_barrier_counters
        [32][kOoMappedWindowPlanScratchSlots] = {};
    static unsigned int cta_barrier_last_values
        [32][kOoMappedWindowPlanScratchSlots] = {};
    static cudaEvent_t completion_events
        [32][kOoMappedWindowPlanScratchSlots] = {};

    /*
     * Publish completion of the one-time -1-path initialization. The pointer
     * arrays remain the requested storage; this atomic only makes their
     * lock-free reads well-defined after the initializing thread releases them.
     */
    static std::atomic<bool> no_event_fast_path_ready[32] = {};

    if (scratch_index == -1) {
        constexpr int storage_index = 0;

        if (no_event_fast_path_ready[device].load(
                std::memory_order_acquire)) {
            void* host_buffer = host_buffers[device][storage_index];
            void* device_buffer = device_buffers[device][storage_index];
            unsigned int* cta_barrier_counter =
                cta_barrier_counters[device][storage_index];

            if (host_buffer != nullptr &&
                device_buffer != nullptr &&
                cta_barrier_counter != nullptr) {
                out->host_plan =
                    reinterpret_cast<
                        comm::plan::WindowTaskExecutorPlan<MaxTasks>*>(
                            host_buffer);
                out->device_plan =
                    reinterpret_cast<
                        const comm::plan::WindowTaskExecutorPlan<MaxTasks>*>(
                            device_buffer);
                out->cta_barrier_counter = cta_barrier_counter;
                out->cta_barrier_last_value =
                    &cta_barrier_last_values[device][storage_index];
                out->completion_event = nullptr;
                return cudaSuccess;
            }
        }

        std::lock_guard<std::mutex> lock(mutex);

        void*& host_buffer = host_buffers[device][storage_index];
        void*& device_buffer = device_buffers[device][storage_index];
        unsigned int*& cta_barrier_counter =
            cta_barrier_counters[device][storage_index];

        /*
         * Check again after taking the lock. Another thread may have completed
         * the allocation while this thread was waiting.
         */
        if (host_buffer == nullptr || device_buffer == nullptr) {
            cudaError_t err = cudaSetDevice(device);
            if (err != cudaSuccess) {
                return err;
            }

            void* host_ptr = nullptr;

            err =
                cudaHostAlloc(
                    &host_ptr,
                    sizeof(comm::plan::WindowTaskExecutorPlan<MaxTasks>),
                    cudaHostAllocMapped | cudaHostAllocPortable);

            if (err != cudaSuccess) {
                return err;
            }

            void* device_ptr = nullptr;

            err =
                cudaHostGetDevicePointer(
                    &device_ptr,
                    host_ptr,
                    0);

            if (err != cudaSuccess) {
                cudaFreeHost(host_ptr);
                return err;
            }

            host_buffer = host_ptr;
            device_buffer = device_ptr;
        }

        if (cta_barrier_counter == nullptr) {
            const cudaError_t counter_err =
                allocate_zeroed_cta_barrier_counter(
                    device,
                    &cta_barrier_counter);
            if (counter_err != cudaSuccess) {
                return counter_err;
            }
        }

        no_event_fast_path_ready[device].store(
            true,
            std::memory_order_release);

        out->host_plan =
            reinterpret_cast<comm::plan::WindowTaskExecutorPlan<MaxTasks>*>(
                host_buffer);
        out->device_plan =
            reinterpret_cast<
                const comm::plan::WindowTaskExecutorPlan<MaxTasks>*>(
                    device_buffer);
        out->cta_barrier_counter = cta_barrier_counter;
        out->cta_barrier_last_value =
            &cta_barrier_last_values[device][storage_index];
        out->completion_event = nullptr;

        return cudaSuccess;
    }

    /*
     * Existing event-protected path for every nonnegative scratch index.
     */
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return err;
    }

    std::lock_guard<std::mutex> lock(mutex);

    void*& host_buffer = host_buffers[device][scratch_index];
    void*& device_buffer = device_buffers[device][scratch_index];
    unsigned int*& cta_barrier_counter =
        cta_barrier_counters[device][scratch_index];
    cudaEvent_t& completion_event =
        completion_events[device][scratch_index];

    if (host_buffer == nullptr ||
        device_buffer == nullptr ||
        completion_event == nullptr) {
        void* host_ptr = nullptr;

        err =
            cudaHostAlloc(
                &host_ptr,
                sizeof(comm::plan::WindowTaskExecutorPlan<MaxTasks>),
                cudaHostAllocMapped | cudaHostAllocPortable);

        if (err != cudaSuccess) {
            return err;
        }

        void* device_ptr = nullptr;

        err =
            cudaHostGetDevicePointer(
                &device_ptr,
                host_ptr,
                0);

        if (err != cudaSuccess) {
            cudaFreeHost(host_ptr);
            return err;
        }

        cudaEvent_t event = nullptr;
        err =
            cudaEventCreateWithFlags(
                &event,
                cudaEventDisableTiming);

        if (err != cudaSuccess) {
            cudaFreeHost(host_ptr);
            return err;
        }

        host_buffer = host_ptr;
        device_buffer = device_ptr;
        completion_event = event;
    } else {
        /*
         * The CPU is about to overwrite mapped host memory. A stream wait is not
         * sufficient for that: it orders GPU work but does not stop this CPU
         * write. Query first; block the host only if this exact slot is still
         * being read by its previous kernel.
         */
        err = cudaEventQuery(completion_event);

        if (err == cudaErrorNotReady) {
            err = cudaEventSynchronize(completion_event);
        }

        if (err != cudaSuccess) {
            return err;
        }
    }

    if (cta_barrier_counter == nullptr) {
        const cudaError_t counter_err =
            allocate_zeroed_cta_barrier_counter(
                device,
                &cta_barrier_counter);
        if (counter_err != cudaSuccess) {
            return counter_err;
        }
    }

    out->host_plan =
        reinterpret_cast<comm::plan::WindowTaskExecutorPlan<MaxTasks>*>(
            host_buffer);
    out->device_plan =
        reinterpret_cast<const comm::plan::WindowTaskExecutorPlan<MaxTasks>*>(
            device_buffer);
    out->cta_barrier_counter = cta_barrier_counter;
    out->cta_barrier_last_value =
        &cta_barrier_last_values[device][scratch_index];
    out->completion_event = completion_event;

    return cudaSuccess;
}


/*
 * OOVERLAP_PLAN_SCRATCH_LEGACY_CALLER_OVERLOAD_V1
 *
 * Backward-compatible entry point for all-gather and reduce-scatter. Those
 * paths keep their previous single-outstanding-plan behavior in reserved entry
 * 256. Round-robin all-reduce uses explicit indices 0..255.
 */
template <int MaxTasks>
cudaError_t get_mapped_window_plan_scratch(
    int device,
    WindowPlanMappedScratch<MaxTasks>* out) {
    return get_mapped_window_plan_scratch<MaxTasks>(
        device,
        kOoMappedWindowPlanScratchSlots - 1,
        out);
}


/*
 * OOVERLAP_WINDOW_PLAN_DEVICE_LAUNCH_HELPER_PATCH:
 *
 * WindowTaskExecutorPlan can be too large to pass by value as a CUDA kernel
 * parameter after WindowTask grows fanout metadata. Keep the host-side lowering
 * path unchanged, but pass the mapped device alias to the kernel.
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
cudaError_t launch_multi_gpu_window_task_executor_sm90(
    const comm::plan::WindowTaskExecutorPlan<MaxTasks>* window_plan,
    int num_blocks,
    int threads,
    size_t dynamic_shared_bytes,
    cudaStream_t stream,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch,
    unsigned int* cta_barrier_counter = nullptr,
    unsigned int cta_barrier_start = 0u) {
    if (num_blocks <= 0 || threads <= 0 || window_plan == nullptr) {
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
