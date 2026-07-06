#include "comm/tma_multi_gpu_reduce_scatter_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/multi_gpu_ready_signal.cuh"
#include "comm/kernels/multi_gpu_window_task_executor.cuh"
#include "comm/launch_config.h"
#include "comm/params.h"
#include "comm/pipeline/pipeline_tma_reduce.h"
#include "comm/plan/tma_multi_gpu_reduce_scatter_plan.cuh"
#include "comm/plan/transfer_lowering.cuh"
#include "comm/plan/transfer_plan_distribution.h"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/utils/collective_utils.h"
#include "comm/utils/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>

namespace ooverlap {
namespace {

template <int MaxPeers>
bool validate_peer_inputs(
    void* const* peer_bufs,
    const int* peer_ranks,
    int peer_count,
    int rank,
    int world_size) {
    if (peer_count < 0 ||
        peer_count > MaxPeers ||
        world_size != peer_count + 1 ||
        rank < 0 ||
        rank >= world_size ||
        world_size > kOoMaxLocalDevices) {
        return false;
    }

    if (peer_count > 0 &&
        (peer_bufs == nullptr || peer_ranks == nullptr)) {
        return false;
    }

    bool seen[kOoMaxLocalDevices] = {};
    seen[rank] = true;

    for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
        const int peer_rank = peer_ranks[peer_idx];

        if (peer_bufs[peer_idx] == nullptr ||
            peer_rank < 0 ||
            peer_rank >= world_size ||
            peer_rank == rank ||
            seen[peer_rank]) {
            return false;
        }

        seen[peer_rank] = true;
    }

    for (int r = 0; r < world_size; ++r) {
        if (!seen[r]) {
            return false;
        }
    }

    return true;
}

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
cudaError_t launch_reduce_scatter_rank_variant_sm90_transfer_plan(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    const int* peer_ranks,
    int peer_count,
    size_t count,
    int rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int collective_epoch,
    comm::LaunchConfig launch_config,
    const comm::plan::ReduceScatterTransferPlan& transfer_plan) {
    (void)count;

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
    constexpr int MaxRanks = kOoMaxLocalDevices;
    constexpr int MaxStagingSlots = 0;

    if (launch_config.plan_for != comm::CollectivePlanFor::ReduceScatter ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.chunk_bytes != Variant::chunk_bytes ||
        launch_config.stage_depth != Variant::stage_depth) {
        return cudaErrorInvalidValue;
    }

    if (local_in == nullptr ||
        local_buf == nullptr ||
        local_device < 0 ||
        !validate_peer_inputs<MaxPeers>(
            peer_bufs,
            peer_ranks,
            peer_count,
            rank,
            world_size)) {
        return cudaErrorInvalidValue;
    }

    if (transfer_plan.world_size != world_size ||
        transfer_plan.total_tasks < 0 ||
        transfer_plan.total_tasks >
            comm::plan::kTmaMultiGpuReduceScatterMaxTransferTasks) {
        return cudaErrorInvalidValue;
    }

    /*
     * The transfer plan is logical and pointer-free. Bind logical ranks to
     * this process's current raw pointer view here. Do not pre-slice pointers:
     * LogicalBufferRef::byte_offset is applied by the lowering layer.
     */
    comm::plan::RankPointerBinding<MaxRanks, MaxStagingSlots> binding{};
    binding.current_rank = rank;
    binding.world_size = world_size;

    binding.rank_buffer[rank] = local_buf;
    binding.rank_input[rank] = local_in;
    binding.rank_output[rank] = local_buf;

    for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
        const int peer_rank = peer_ranks[peer_idx];

        binding.rank_buffer[peer_rank] = peer_bufs[peer_idx];
        binding.rank_input[peer_rank] = peer_bufs[peer_idx];
        binding.rank_output[peer_rank] = peer_bufs[peer_idx];
    }

    comm::plan::WindowTaskExecutorPlan<MaxTasks> plan{};
    int num_blocks = 0;

    const bool plan_ok =
        comm::plan::lower_transfer_plan_for_rank<
            comm::plan::kTmaMultiGpuReduceScatterMaxTransferTasks,
            MaxTasks,
            MaxRanks,
            MaxStagingSlots>(
                transfer_plan,
                binding,
                launch_config,
                &plan,
                &num_blocks);

    if (!plan_ok) {
        return cudaErrorInvalidValue;
    }

    if (num_blocks <= 0 || plan.total_tasks <= 0) {
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

template <
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
cudaError_t dispatch_reduce_scatter_dtype_op_sm90_transfer_plan(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    const int* peer_ranks,
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
    comm::LaunchConfig launch_config,
    const comm::plan::ReduceScatterTransferPlan& transfer_plan) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_scatter_rank_variant_sm90_transfer_plan<
                comm::pipeline::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_ranks,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config,
                    transfer_plan);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_scatter_rank_variant_sm90_transfer_plan<
                comm::pipeline::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_ranks,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config,
                    transfer_plan);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_scatter_rank_variant_sm90_transfer_plan<
                comm::pipeline::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_ranks,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config,
                    transfer_plan);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_scatter_rank_variant_sm90_transfer_plan<
                comm::pipeline::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_ranks,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config,
                    transfer_plan);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_scatter_rank_variant_sm90_transfer_plan<
                comm::pipeline::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_ranks,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config,
                    transfer_plan);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_scatter_rank_variant_sm90_transfer_plan<
                comm::pipeline::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    local_in,
                    local_buf,
                    peer_bufs,
                    peer_ranks,
                    peer_count,
                    count,
                    rank,
                    world_size,
                    local_device,
                    stream,
                    local_ready_signal,
                    peer_ready_signals,
                    collective_epoch,
                    launch_config,
                    transfer_plan);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32 &&
        (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM)) {
        return launch_reduce_scatter_rank_variant_sm90_transfer_plan<
            comm::pipeline::PipelineReduceAddF32,
            static_cast<int>(sizeof(float)),
            ChunkBytes,
            StageDepth,
            FillDepth,
            LoadFillDepth>(
                local_in,
                local_buf,
                peer_bufs,
                peer_ranks,
                peer_count,
                count,
                rank,
                world_size,
                local_device,
                stream,
                local_ready_signal,
                peer_ready_signals,
                collective_epoch,
                launch_config,
                transfer_plan);
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
        return dispatch_reduce_scatter_dtype_op_sm90<                                            \
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

cudaError_t dispatch_reduce_scatter_variant_sm90_transfer_plan(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    const int* peer_ranks,
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
    comm::LaunchConfig launch_config,
    const comm::plan::ReduceScatterTransferPlan& transfer_plan) {
#define OO_TRY_VARIANT(CHUNK_BYTES_VALUE, STAGE_DEPTH_VALUE, FILL_DEPTH_VALUE, LOAD_FILL_DEPTH_VALUE) \
    if (launch_config.chunk_bytes == (CHUNK_BYTES_VALUE) &&                                      \
        launch_config.stage_depth == (STAGE_DEPTH_VALUE)) {                                      \
        return dispatch_reduce_scatter_dtype_op_sm90_transfer_plan<                              \
            (CHUNK_BYTES_VALUE),                                                                 \
            (STAGE_DEPTH_VALUE),                                                                 \
            (FILL_DEPTH_VALUE),                                                                  \
            (LOAD_FILL_DEPTH_VALUE)>(                                                           \
                local_in,                                                                        \
                local_buf,                                                                       \
                peer_bufs,                                                                       \
                peer_ranks,                                                                      \
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
                launch_config,                                                                   \
                transfer_plan);                                                                  \
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

cudaError_t enqueue_tma_multi_gpu_reduce_scatter_rank_sm90_transfer_plan(
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    const int* peer_ranks,
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
    comm::LaunchConfig launch_config,
    const comm::plan::ReduceScatterTransferPlan& transfer_plan) {
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

    if (peer_count > 0 &&
        (peer_bufs == nullptr || peer_ranks == nullptr)) {
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
        return dispatch_reduce_scatter_variant_sm90_transfer_plan(
            local_in,
            local_buf,
            peer_bufs,
            peer_ranks,
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
            launch_config,
            transfer_plan);
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (const std::exception&) {
        return cudaErrorUnknown;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

} // namespace ooverlap
