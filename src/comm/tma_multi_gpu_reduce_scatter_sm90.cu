#include "comm/tma_multi_gpu_reduce_scatter_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/window_task_executor.cuh"
#include "comm/launch_config.h"
#include "comm/params.h"
#include "comm/pipeline/pipeline_tma_reduce.h"
#include "comm/plan/tma_multi_gpu_reduce_scatter_plan.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/utils/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ooverlap {
namespace {

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

bool rank_partition(
    size_t count,
    int rank,
    int world_size,
    size_t* out_begin,
    size_t* out_count) {
    if (out_begin == nullptr || out_count == nullptr) {
        return false;
    }

    *out_begin = 0;
    *out_count = 0;

    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        return false;
    }

    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);

    const size_t base = count / world;
    const size_t rem = count % world;

    *out_begin = r * base + ((r < rem) ? r : rem);
    *out_count = base + ((r < rem) ? 1 : 0);

    return true;
}

const void* offset_const_ptr(
    const void* ptr,
    size_t byte_offset) {
    return reinterpret_cast<const void*>(
        reinterpret_cast<const unsigned char*>(ptr) + byte_offset);
}

void* offset_ptr(
    void* ptr,
    size_t byte_offset) {
    return reinterpret_cast<void*>(
        reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

template <int MaxPeers>
struct MultiGpuReadySignalPlan {
    int peer_count = 0;
    const int* peer_ready_signals[MaxPeers] = {};
};

template <int MaxPeers>
__device__ __forceinline__ void wait_for_multi_gpu_collective_ready(
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch) {
    if (local_ready_signal == nullptr || collective_epoch <= 0) {
        return;
    }

    if (threadIdx.x == 0) {
        atomicMax(local_ready_signal, collective_epoch);
        __threadfence_system();

        for (int peer_idx = 0; peer_idx < ready_plan.peer_count; ++peer_idx) {
            const int* peer_ready_signal =
                ready_plan.peer_ready_signals[peer_idx];

            if (peer_ready_signal == nullptr) {
                continue;
            }

            const volatile int* peer_ready =
                reinterpret_cast<const volatile int*>(peer_ready_signal);

            while (peer_ready[0] < collective_epoch) {
#if defined(__CUDA_ARCH__)
                __nanosleep(64);
#endif
            }
        }
    }

    __syncthreads();
}

template <
    typename ReduceApply,
    int ChunkBytes,
    int StageDepth,
    int MaxTasks,
    int MaxPeers>
__global__ void reduce_scatter_window_task_executor_kernel_sm90(
    comm::plan::WindowTaskExecutorPlan<MaxTasks> plan,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    wait_for_multi_gpu_collective_ready(
        local_ready_signal,
        ready_plan,
        collective_epoch);

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[Variant::barrier_count];

    comm::kernels::execute_window_task_stripe<
        Variant::stage_depth,
        Variant::stage_gap,
        Variant::chunk_bytes,
        ReduceApply>(
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
    int MaxPeers>
cudaError_t launch_reduce_scatter_window_task_executor_sm90(
    comm::plan::WindowTaskExecutorPlan<MaxTasks> plan,
    int num_blocks,
    int threads,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch,
    cudaStream_t stream) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    if (threads <= 0 || threads > 1024 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }

    reduce_scatter_window_task_executor_kernel_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxTasks,
        MaxPeers><<<
            num_blocks,
            threads,
            Variant::dynamic_shared_bytes,
            stream>>>(
                plan,
                local_ready_signal,
                ready_plan,
                collective_epoch);

    return cudaGetLastError();
}

template <
    typename ReduceApply,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
void configure_reduce_scatter_kernel_once_for(int device) {
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
            "tma_multi_gpu_reduce_scatter_configure_kernel_once: "
            "requested shared memory exceeds opt-in limit");
    }

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                reduce_scatter_window_task_executor_kernel_sm90<
                    ReduceApply,
                    ChunkBytes,
                    StageDepth,
                    comm::plan::kTmaMultiGpuReduceScatterMaxWindowTasks,
                    comm::plan::kTmaMultiGpuReduceScatterMaxPeers>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    }

    system::runtime::check_cuda(
        cudaFuncSetAttribute(
            reduce_scatter_window_task_executor_kernel_sm90<
                ReduceApply,
                ChunkBytes,
                StageDepth,
                comm::plan::kTmaMultiGpuReduceScatterMaxWindowTasks,
                comm::plan::kTmaMultiGpuReduceScatterMaxPeers>,
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
cudaError_t launch_reduce_scatter_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    int peer_count,
    size_t count,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
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

    if (peer_count < 0 ||
        peer_count > comm::plan::kTmaMultiGpuReduceScatterMaxPeers ||
        world_size != peer_count + 1 ||
        rank < 0 ||
        rank >= world_size ||
        local_device < 0) {
        return cudaErrorInvalidValue;
    }

    if (peer_count > 0 && peer_bufs == nullptr) {
        return cudaErrorInvalidValue;
    }

    size_t slice_begin_elems = 0;
    size_t slice_count = 0;

    if (!rank_partition(
            count,
            rank,
            world_size,
            &slice_begin_elems,
            &slice_count)) {
        return cudaErrorInvalidValue;
    }

    const size_t slice_begin_bytes =
        slice_begin_elems * static_cast<size_t>(ElemBytes);

    const size_t slice_bytes =
        slice_count * static_cast<size_t>(ElemBytes);

    const void* local_in_slice =
        offset_const_ptr(local_in, slice_begin_bytes);

    void* local_buf_slice =
        offset_ptr(local_buf, slice_begin_bytes);

    void* peer_slices[comm::plan::kTmaMultiGpuReduceScatterMaxPeers] = {};

    for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
        if (peer_bufs[peer_idx] == nullptr) {
            return cudaErrorInvalidValue;
        }

        peer_slices[peer_idx] =
            offset_ptr(peer_bufs[peer_idx], slice_begin_bytes);
    }

    MultiGpuReadySignalPlan<
        comm::plan::kTmaMultiGpuReduceScatterMaxPeers> ready_plan{};

    ready_plan.peer_count = peer_count;

    for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
        ready_plan.peer_ready_signals[peer_idx] =
            (peer_ready_signals != nullptr)
                ? peer_ready_signals[peer_idx]
                : nullptr;
    }

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            slice_bytes,
            Variant::chunk_bytes);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            launch_config.window_chunks);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        collective_epoch > 0;

    comm::plan::WindowTaskExecutorPlan<
        comm::plan::kTmaMultiGpuReduceScatterMaxWindowTasks> plan{};

    int num_blocks = 0;

    const bool plan_ok =
        comm::plan::build_tma_multi_gpu_reduce_scatter_naive_plan(
            &plan,
            &num_blocks,
            local_in_slice,
            local_buf_slice,
            peer_slices,
            peer_count,
            slice_bytes,
            num_windows,
            launch_config,
            needs_rendezvous);

    if (!plan_ok) {
        return cudaErrorInvalidValue;
    }

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    system::runtime::set_device(local_device);

    configure_reduce_scatter_kernel_once_for<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(local_device);

    system::runtime::set_device(local_device);

    return launch_reduce_scatter_window_task_executor_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        comm::plan::kTmaMultiGpuReduceScatterMaxWindowTasks,
        comm::plan::kTmaMultiGpuReduceScatterMaxPeers>(
            plan,
            num_blocks,
            launch_config.threads,
            local_ready_signal,
            ready_plan,
            collective_epoch,
            stream);
}

template <
    typename ReduceOp,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
cudaError_t launch_reduce_scatter_reduce_op_sm90(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    int peer_count,
    size_t count,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    using ReduceApply = comm::pipeline::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::stage_gap,
        ReduceOp>;

    return launch_reduce_scatter_rank_kernel_sm90<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(
            local_in,
            local_buf,
            peer_bufs,
            peer_count,
            count,
            rank,
            world_size,
            local_device,
            stream,
            local_ready_signal,
            peer_ready_signals,
            collective_epoch,
            launch_config);
}

template <
    typename ReduceOp,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
void configure_reduce_scatter_reduce_op_sm90(int device) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    using ReduceApply = comm::pipeline::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::stage_gap,
        ReduceOp>;

    configure_reduce_scatter_kernel_once_for<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(device);
}

template <int ChunkBytes, int StageDepth>
cudaError_t dispatch_reduce_scatter_rank_kernel_variant_sm90(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config);
        }
    }

    return cudaErrorInvalidValue;
}

cudaError_t dispatch_reduce_scatter_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
#define OO_TRY_VARIANT(CHUNK_BYTES_VALUE, STAGE_DEPTH_VALUE)                 \
    if (launch_config.chunk_bytes == (CHUNK_BYTES_VALUE) &&                  \
        launch_config.stage_depth == (STAGE_DEPTH_VALUE)) {                  \
        return dispatch_reduce_scatter_rank_kernel_variant_sm90<             \
            (CHUNK_BYTES_VALUE),                                             \
            (STAGE_DEPTH_VALUE)>(                                            \
                local_in,                                                    \
                local_buf,                                                   \
                peer_bufs,                                                   \
                peer_count,                                                  \
                count,                                                       \
                dtype,                                                       \
                op,                                                          \
                rank,                                                        \
                world_size,                                                  \
                local_device,                                                \
                stream,                                                      \
                local_ready_signal,                                          \
                peer_ready_signals,                                          \
                collective_epoch,                                            \
                launch_config);                                              \
    }

    OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT(OO_TRY_VARIANT)

#undef OO_TRY_VARIANT

    return cudaErrorInvalidValue;
}

template <int ChunkBytes, int StageDepth>
void configure_reduce_scatter_dispatch_variant_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_scatter_reduce_op_sm90<
                comm::pipeline::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    throw std::invalid_argument(
        "tma_multi_gpu_reduce_scatter_configure_kernel_once: unsupported dtype/op");
}

void configure_reduce_scatter_dispatch_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    configure_reduce_scatter_dispatch_variant_sm90<
        TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES,
        TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH>(
            dtype,
            op,
            device);
}

} // namespace

void tma_multi_gpu_reduce_scatter_configure_kernel_once(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    configure_reduce_scatter_dispatch_sm90(dtype, op, device);
}

cudaError_t enqueue_tma_multi_gpu_reduce_scatter_rank_sm90(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    if (count == 0) {
        return cudaSuccess;
    }

    if (local_in == nullptr || local_buf == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (peer_count < 0 ||
        peer_count > comm::plan::kTmaMultiGpuReduceScatterMaxPeers ||
        world_size != peer_count + 1 ||
        rank < 0 ||
        rank >= world_size ||
        local_device < 0) {
        return cudaErrorInvalidValue;
    }

    if (peer_count > 0 && peer_bufs == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (!reduce_op_supported_for_dtype(dtype, op)) {
        return cudaErrorInvalidValue;
    }

    if (!comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    try {
        return dispatch_reduce_scatter_rank_kernel_sm90(
            local_in,
            local_buf,
            peer_bufs,
            peer_count,
            count,
            dtype,
            op,
            rank,
            world_size,
            local_device,
            stream,
            local_ready_signal,
            peer_ready_signals,
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
