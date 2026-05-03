#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/window_task_executor.cuh"
#include "comm/launch_config.h"
#include "comm/params.h"
#include "comm/pipeline/pipeline_tma_reduce.h"
#include "comm/plan/tma_two_gpu_peer_allreduce_plan.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/utils/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ooverlap {
namespace {

struct SignalCacheEntry {
    int* ptr = nullptr;
    size_t capacity = 0;
};

bool reduce_op_supported_for_dtype(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16 ||
               dtype == OO_DTYPE_FLOAT32;
    }

    if (op == OO_REDUCE_MIN || op == OO_REDUCE_MAX) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16;
    }

    return false;
}

SignalCacheEntry& signal_cache_for_device(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, SignalCacheEntry> cache;

    std::lock_guard<std::mutex> lock(mutex);
    return cache[device];
}

int* ensure_signal_capacity(int device, size_t required_count) {
    if (required_count == 0) {
        return nullptr;
    }

    SignalCacheEntry& entry = signal_cache_for_device(device);

    if (entry.ptr != nullptr && entry.capacity >= required_count) {
        return entry.ptr;
    }

    system::runtime::set_device(device);

    if (entry.ptr != nullptr) {
        cudaFree(entry.ptr);
        entry.ptr = nullptr;
        entry.capacity = 0;
    }

    system::runtime::check_cuda(
        cudaMalloc(&entry.ptr, required_count * sizeof(int)),
        "cudaMalloc(overlap window ready flags)");

    entry.capacity = required_count;
    return entry.ptr;
}

template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxTasks>
cudaError_t launch_window_task_executor_rank_sm90(
    comm::plan::WindowTaskExecutorPlan<MaxTasks> plan,
    int num_blocks,
    int threads,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    cudaStream_t stream) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    if (threads <= 0 || threads > 1024 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }

    comm::kernels::window_task_executor_kernel_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxTasks><<<
            num_blocks,
            threads,
            Variant::dynamic_shared_bytes,
            stream>>>(
                plan,
                local_ready_signal,
                peer_ready_signal,
                collective_epoch);

    return cudaGetLastError();
}

template <
    typename ReduceApply,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
void configure_kernel_once_for(int device) {
    (void)ElemBytes;

    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

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
            "tma_two_gpu_peer_allreduce_configure_kernel_once: "
            "requested shared memory exceeds opt-in limit");
    }

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                comm::kernels::window_task_executor_kernel_sm90<
                    ReduceApply,
                    ChunkBytes,
                    StageDepth,
                    comm::plan::kTmaTwoGpuPeerAllreduceMaxWindowTasks>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    }

    system::runtime::check_cuda(
        cudaFuncSetAttribute(
            comm::kernels::window_task_executor_kernel_sm90<
                ReduceApply,
                ChunkBytes,
                StageDepth,
                comm::plan::kTmaTwoGpuPeerAllreduceMaxWindowTasks>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");

    cache[device] = {true, dynamic_smem_bytes};
}

template <
    typename ReduceApply,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
cudaError_t launch_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    if (!comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.chunk_bytes != Variant::chunk_bytes ||
        launch_config.stage_depth != Variant::stage_depth) {
        return cudaErrorInvalidValue;
    }

    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes =
        count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            Variant::chunk_bytes);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            launch_config.window_chunks);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    const bool out_of_place = (local_in != local_buf);

    const int window_ready_flag_count =
        comm::plan::required_window_ready_flags_for_plan(
            launch_config.plan_kind,
            out_of_place,
            rank,
            num_windows);

    system::runtime::set_device(device);

    int* window_ready_flags = nullptr;

    if (window_ready_flag_count > 0) {
        window_ready_flags =
            ensure_signal_capacity(
                device,
                static_cast<size_t>(window_ready_flag_count));

        system::runtime::check_cuda(
            cudaMemsetAsync(
                window_ready_flags,
                0,
                static_cast<size_t>(window_ready_flag_count) * sizeof(int),
                stream),
            "cudaMemsetAsync(window ready flags)");
    }

    comm::plan::WindowTaskExecutorPlan<
        comm::plan::kTmaTwoGpuPeerAllreduceMaxWindowTasks> plan{};

    int num_blocks = 0;

    const bool plan_ok =
        comm::plan::build_tma_two_gpu_peer_allreduce_plan(
            &plan,
            &num_blocks,
            local_in,
            local_buf,
            peer_buf,
            total_bytes,
            rank,
            num_windows,
            launch_config,
            needs_rendezvous,
            window_ready_flags);

    if (!plan_ok) {
        return cudaErrorInvalidValue;
    }

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    configure_kernel_once_for<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(device);

    system::runtime::set_device(device);

    return launch_window_task_executor_rank_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        comm::plan::kTmaTwoGpuPeerAllreduceMaxWindowTasks>(
            plan,
            num_blocks,
            launch_config.threads,
            local_ready_signal,
            peer_ready_signal,
            collective_epoch,
            stream);
}

template <
    typename ReduceOp,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
cudaError_t launch_reduce_op_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    using ReduceApply = comm::pipeline::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::stage_gap,
        ReduceOp>;

    return launch_rank_kernel_sm90<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(
            local_in,
            local_buf,
            peer_buf,
            count,
            rank,
            dev0,
            dev1,
            stream,
            local_ready_signal,
            peer_ready_signal,
            collective_epoch,
            launch_config);
}

template <
    typename ReduceOp,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
void configure_reduce_op_sm90(int device) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    using ReduceApply = comm::pipeline::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::stage_gap,
        ReduceOp>;

    configure_kernel_once_for<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(device);
}

template <int ChunkBytes, int StageDepth>
cudaError_t dispatch_rank_kernel_variant_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::pipeline::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::pipeline::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::pipeline::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::pipeline::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }
    }

    return cudaErrorInvalidValue;
}

cudaError_t dispatch_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
#define OO_TRY_VARIANT(CHUNK_BYTES_VALUE, STAGE_DEPTH_VALUE)                 \
    if (launch_config.chunk_bytes == (CHUNK_BYTES_VALUE) &&                  \
        launch_config.stage_depth == (STAGE_DEPTH_VALUE)) {                  \
        return dispatch_rank_kernel_variant_sm90<                            \
            (CHUNK_BYTES_VALUE),                                             \
            (STAGE_DEPTH_VALUE)>(                                            \
                local_in,                                                    \
                local_buf,                                                   \
                peer_buf,                                                    \
                count,                                                       \
                dtype,                                                       \
                op,                                                          \
                rank,                                                        \
                dev0,                                                        \
                dev1,                                                        \
                stream,                                                      \
                local_ready_signal,                                          \
                peer_ready_signal,                                           \
                collective_epoch,                                            \
                launch_config);                                              \
    }

    OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT(OO_TRY_VARIANT)

#undef OO_TRY_VARIANT

    return cudaErrorInvalidValue;
}

template <int ChunkBytes, int StageDepth>
void configure_dispatch_variant_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_op_sm90<
                comm::pipeline::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_op_sm90<
                comm::pipeline::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_op_sm90<
                comm::pipeline::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_op_sm90<
                comm::pipeline::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    throw std::invalid_argument(
        "tma_two_gpu_peer_allreduce_configure_kernel_once: unsupported dtype/op");
}

void configure_dispatch_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    configure_dispatch_variant_sm90<
        TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES,
        TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH>(
            dtype,
            op,
            device);
}

} // namespace

void tma_two_gpu_peer_allreduce_configure_kernel_once(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    configure_dispatch_sm90(dtype, op, device);
}

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    if (rank != 0 && rank != 1) {
        return cudaErrorInvalidValue;
    }

    if (count == 0) {
        return cudaSuccess;
    }

    if (local_in == nullptr || local_buf == nullptr || peer_buf == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (!reduce_op_supported_for_dtype(dtype, op)) {
        return cudaErrorInvalidValue;
    }

    if (!comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    try {
        return dispatch_rank_kernel_sm90(
            local_in,
            local_buf,
            peer_buf,
            count,
            dtype,
            op,
            rank,
            dev0,
            dev1,
            stream,
            local_ready_signal,
            peer_ready_signal,
            collective_epoch,
            launch_config);
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (const std::exception&) {
        return cudaErrorUnknown;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

} // namespace ooverlap
