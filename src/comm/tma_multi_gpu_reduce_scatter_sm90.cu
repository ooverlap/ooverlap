#include "comm/tma_multi_gpu_reduce_scatter_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/multi_gpu_ready_signal.cuh"
#include "comm/kernels/multi_gpu_window_task_executor.cuh"
#include "comm/launch_config.h"
#include "comm/params.h"
#include "comm/pipeline/pipeline_tma_reduce.h"
#include "comm/plan/tma_multi_gpu_reduce_scatter_plan.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/utils/collective_utils.h"
#include "comm/utils/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace ooverlap {
namespace {
template <
    typename ReduceOp,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
cudaError_t launch_reduce_scatter_rank_variant_sm90(
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
    using Variant = comm::TmaPipelineVariant<
        ChunkBytes,
        StageDepth,
        FillDepth,
        LoadFillDepth>;

    using ReduceApply = comm::pipeline::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::fill_depth,
        ReduceOp>;

    constexpr int MaxTasks =
        comm::plan::kTmaMultiGpuReduceScatterMaxWindowTasks;
    constexpr int MaxPeers =
        comm::plan::kTmaMultiGpuReduceScatterMaxPeers;


    if (launch_config.plan_for != comm::CollectivePlanFor::ReduceScatter ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.chunk_bytes != Variant::chunk_bytes ||
        launch_config.stage_depth != Variant::stage_depth) {
        return cudaErrorInvalidValue;
    }

    if (peer_count < 0 ||
        peer_count > MaxPeers ||
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

    if (!comm::utils::rank_partition(
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
        comm::utils::offset_const_ptr(local_in, slice_begin_bytes);

    void* local_buf_slice =
        comm::utils::offset_ptr(local_buf, slice_begin_bytes);

    void* peer_slices[MaxPeers] = {};

    for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
        if (peer_bufs[peer_idx] == nullptr) {
            return cudaErrorInvalidValue;
        }

        peer_slices[peer_idx] =
            comm::utils::offset_ptr(peer_bufs[peer_idx], slice_begin_bytes);
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

    comm::plan::WindowTaskExecutorPlan<MaxTasks> plan{};
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

    auto ready_plan =
        comm::kernels::make_multi_gpu_ready_signal_plan<MaxPeers>(
            peer_count,
            peer_ready_signals);

    system::runtime::set_device(local_device);

    comm::kernels::configure_multi_gpu_window_task_executor_once<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxTasks,
        MaxPeers,
        Variant::fill_depth,
        Variant::load_fill_depth>(
            local_device,
            "tma_multi_gpu_reduce_scatter: requested shared memory exceeds opt-in limit");

    system::runtime::set_device(local_device);

    comm::kernels::multi_gpu_window_task_executor_kernel_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxTasks,
        MaxPeers,
        Variant::fill_depth,
        Variant::load_fill_depth><<<
            num_blocks,
            launch_config.threads,
            Variant::dynamic_shared_bytes,
            stream>>>(
                plan,
                local_ready_signal,
                ready_plan,
                collective_epoch);

    return cudaGetLastError();
}

template <
    typename ReduceOp,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
void configure_reduce_scatter_variant_sm90(int device) {
    using Variant = comm::TmaPipelineVariant<
        ChunkBytes,
        StageDepth,
        FillDepth,
        LoadFillDepth>;

    using ReduceApply = comm::pipeline::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::fill_depth,
        ReduceOp>;

    comm::kernels::configure_multi_gpu_window_task_executor_once<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        comm::plan::kTmaMultiGpuReduceScatterMaxWindowTasks,
        comm::plan::kTmaMultiGpuReduceScatterMaxPeers,
        Variant::fill_depth,
        Variant::load_fill_depth>(
            device,
            "tma_multi_gpu_reduce_scatter: requested shared memory exceeds opt-in limit");
}

template <
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
cudaError_t dispatch_reduce_scatter_dtype_op_sm90(
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
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
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
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
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
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
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
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
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
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
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
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
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

    if (dtype == OO_DTYPE_FLOAT32 &&
        (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM)) {
        return launch_reduce_scatter_rank_variant_sm90<
            comm::pipeline::PipelineReduceAddF32,
            static_cast<int>(sizeof(float)),
            ChunkBytes,
            StageDepth,
            FillDepth,
            LoadFillDepth>(
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

    return cudaErrorInvalidValue;
}

cudaError_t dispatch_reduce_scatter_variant_sm90(
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
#define OO_TRY_VARIANT(CHUNK_BYTES_VALUE, STAGE_DEPTH_VALUE, FILL_DEPTH_VALUE, LOAD_FILL_DEPTH_VALUE) \
    if (launch_config.chunk_bytes == (CHUNK_BYTES_VALUE) &&                                      \
        launch_config.stage_depth == (STAGE_DEPTH_VALUE)) {                                      \
        return dispatch_reduce_scatter_dtype_op_sm90<                                                  \
            (CHUNK_BYTES_VALUE),                                                                 \
            (STAGE_DEPTH_VALUE),                                                                 \
            (FILL_DEPTH_VALUE),                                                                  \
            (LOAD_FILL_DEPTH_VALUE)>(                                                           \
                local_in,                                                                        \
                local_buf,                                                                       \
                peer_bufs,                                                                       \
                peer_count,                                                                      \
                count,                                                                           \
                dtype,                                                                           \
                op,                                                                              \
                rank,                                                                            \
                world_size,                                                                      \
                local_device,                                                                    \
                stream,                                                                          \
                local_ready_signal,                                                              \
                peer_ready_signals,                                                              \
                collective_epoch,                                                                \
                launch_config);                                                                  \
    }

    OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT_WITH_DEPTH(OO_TRY_VARIANT)

#undef OO_TRY_VARIANT

    return cudaErrorInvalidValue;
}

template <
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
void configure_reduce_scatter_dtype_op_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_scatter_variant_sm90<
                comm::pipeline::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_scatter_variant_sm90<
                comm::pipeline::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_scatter_variant_sm90<
                comm::pipeline::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_scatter_variant_sm90<
                comm::pipeline::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_scatter_variant_sm90<
                comm::pipeline::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_scatter_variant_sm90<
                comm::pipeline::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_FLOAT32 &&
        (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM)) {
        configure_reduce_scatter_variant_sm90<
            comm::pipeline::PipelineReduceAddF32,
            static_cast<int>(sizeof(float)),
            ChunkBytes,
            StageDepth,
            FillDepth,
            LoadFillDepth>(device);
        return;
    }

    throw std::invalid_argument(
        "tma_multi_gpu_reduce_scatter: unsupported dtype/op");
}

} // namespace

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

    if (!comm::utils::reduce_op_supported_for_dtype(dtype, op)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.plan_for != comm::CollectivePlanFor::ReduceScatter ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    try {
        return dispatch_reduce_scatter_variant_sm90(
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
