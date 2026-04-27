#include "comm/tma_two_gpu_peer_allreduce_fast_gmem_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/params.h"
#include "comm/pipeline_tma_reduce.h"
#include "comm/utils.h"
#include "comm/window_pipeline_sm90.cuh"

#include <cooperative_groups.h>
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

namespace cg = cooperative_groups;

__host__ __device__ __forceinline__ size_t dtype_size_bytes(
    oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
            return sizeof(half);
        case OO_DTYPE_BFLOAT16:
            return sizeof(__nv_bfloat16);
        case OO_DTYPE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

__host__ __device__ __forceinline__ int overlap_pair_count_for_windows(
    int owned_windows) {
    if (owned_windows <= 0) {
        return 0;
    }

    return comm::utils::min_int(
        owned_windows,
        TMA_TWO_GPU_PEER_MAX_CTAS / 2);
}

template <typename ReduceApply, int ElemBytes>
__global__ void tma_then_fastcopy_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int ctas_per_rank) {
    comm::window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const int cta_idx = static_cast<int>(blockIdx.x);

    if (ctas_per_rank <= 0 || cta_idx >= ctas_per_rank) {
        return;
    }

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            TMA_TWO_GPU_PEER_WINDOW_CHUNKS);

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    const comm::utils::WindowRange cta_range =
        comm::utils::cta_window_range(
            cta_idx,
            ctas_per_rank,
            rank_range);

    if (cta_range.begin >= cta_range.end) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[TMA_TWO_GPU_PEER_BARRIER_COUNT];

    comm::window_pipeline::run_window_range<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        TMA_TWO_GPU_PEER_WINDOW_CHUNKS,
        ReduceApply>(
            local_in,
            peer_buf,
            total_bytes,
            cta_range.begin,
            cta_range.end,
            shared_raw,
            barriers);

    comm::window_pipeline::copy_window_range_gmem<
        uint4,
        TMA_TWO_GPU_PEER_FAST_COPY_UNROLL,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        TMA_TWO_GPU_PEER_WINDOW_CHUNKS>(
            peer_buf,
            local_buf,
            total_bytes,
            cta_range.begin,
            cta_range.end);
}

template <typename ReduceApply, int ElemBytes>
__global__ void tma_overlap_fastcopy_dsm_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int pair_count) {
    cg::cluster_group cluster = cg::this_cluster();

    const int cluster_block_rank = cluster.block_rank();
    const int role = cluster_block_rank & 1;

    /*
     * Cluster size is exactly 2:
     *
     *   cluster block rank 0 = producer
     *   cluster block rank 1 = consumer
     *
     * One cluster owns one producer/consumer pair.
     */
    const int pair_idx = static_cast<int>(blockIdx.x) / 2;

    __shared__ int dsm_ready_window_exclusive;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            TMA_TWO_GPU_PEER_WINDOW_CHUNKS);

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    const comm::utils::WindowRange pair_range =
        comm::utils::cta_window_range(
            pair_idx,
            pair_count,
            rank_range);

    if (threadIdx.x == 0) {
        dsm_ready_window_exclusive = pair_range.begin;
    }

    cluster.sync();

    comm::window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const bool active_pair =
        pair_count > 0 &&
        pair_idx < pair_count &&
        pair_range.begin < pair_range.end;

    if (active_pair) {
        int* producer_ready_window_exclusive =
            cluster.map_shared_rank(
                &dsm_ready_window_exclusive,
                0);

        if (role == 1) {
            comm::window_pipeline::copy_window_range_gmem_after_counter<
                uint4,
                TMA_TWO_GPU_PEER_FAST_COPY_UNROLL,
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                TMA_TWO_GPU_PEER_WINDOW_CHUNKS>(
                    peer_buf,
                    local_buf,
                    total_bytes,
                    pair_range.begin,
                    pair_range.end,
                    producer_ready_window_exclusive);
        } else {
            extern __shared__ uint4 shared_storage_u4[];

            unsigned char* shared_raw =
                reinterpret_cast<unsigned char*>(shared_storage_u4);

            __shared__ sync::semaphore barriers[
                TMA_TWO_GPU_PEER_BARRIER_COUNT];

            comm::window_pipeline::run_window_range_signal_counter<
                TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
                TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                TMA_TWO_GPU_PEER_WINDOW_CHUNKS,
                ReduceApply>(
                    local_in,
                    peer_buf,
                    total_bytes,
                    pair_range.begin,
                    pair_range.end,
                    producer_ready_window_exclusive,
                    shared_raw,
                    barriers);
        }
    }

    /*
     * DSM lifetime rule: producer block owns the DSM signal, so it must not exit
     * until the consumer is done polling/copying.
     */
    cluster.sync();
}

template <typename ReduceApply, int ElemBytes>
void configure_seq_kernel_once_for(int device) {
    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes =
        TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES;

    const size_t total_smem_bytes =
        dynamic_smem_bytes + TMA_TWO_GPU_PEER_STATIC_SHARED_BYTES;

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
            "tma_fastcopy_allreduce_configure_kernel_once: requested shared memory exceeds opt-in limit");
    }

    if (dynamic_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_then_fastcopy_rank_kernel_sm90<
                    ReduceApply,
                    ElemBytes>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize seq)");
    }

    system::runtime::check_cuda(
        cudaFuncSetAttribute(
            tma_then_fastcopy_rank_kernel_sm90<
                ReduceApply,
                ElemBytes>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout seq)");

    cache[device] = {true, dynamic_smem_bytes};
}

template <typename ReduceApply, int ElemBytes>
void configure_overlap_dsm_kernel_once_for(int device) {
    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes =
        TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES;

    const size_t total_smem_bytes =
        dynamic_smem_bytes + TMA_TWO_GPU_PEER_STATIC_SHARED_BYTES;

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
            "tma_fastcopy_allreduce_configure_dsm_kernel_once: requested shared memory exceeds opt-in limit");
    }

    auto kernel =
        tma_overlap_fastcopy_dsm_rank_kernel_sm90<
            ReduceApply,
            ElemBytes>;

    if (dynamic_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize overlap dsm)");
    }

    system::runtime::check_cuda(
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout overlap dsm)");

    cache[device] = {true, dynamic_smem_bytes};
}

template <typename ReduceApply, int ElemBytes>
cudaError_t launch_seq_rank_kernel_sm90(
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
    int collective_epoch) {
    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            TMA_TWO_GPU_PEER_WINDOW_CHUNKS);

    const int owned_windows =
        comm::utils::rank_window_count(num_windows, rank);

    const int ctas_per_rank =
        comm::utils::cta_count_for_windows(
            owned_windows,
            TMA_TWO_GPU_PEER_MAX_CTAS);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    const int num_blocks =
        needs_rendezvous ? std::max(1, ctas_per_rank) : ctas_per_rank;

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    configure_seq_kernel_once_for<ReduceApply, ElemBytes>(device);

    system::runtime::set_device(device);

    tma_then_fastcopy_rank_kernel_sm90<
        ReduceApply,
        ElemBytes><<<
            num_blocks,
            TMA_TWO_GPU_PEER_THREADS,
            TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES,
            stream>>>(
                local_in,
                local_buf,
                peer_buf,
                count,
                rank,
                local_ready_signal,
                peer_ready_signal,
                collective_epoch,
                ctas_per_rank);

    return cudaGetLastError();
}

template <typename ReduceApply, int ElemBytes>
cudaError_t launch_overlap_dsm_rank_kernel_sm90(
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
    int collective_epoch) {
    constexpr int ClusterBlocks = 2;

    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            TMA_TWO_GPU_PEER_WINDOW_CHUNKS);

    const int owned_windows =
        comm::utils::rank_window_count(num_windows, rank);

    const int pair_count =
        overlap_pair_count_for_windows(owned_windows);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    if (pair_count <= 0 && !needs_rendezvous) {
        return cudaSuccess;
    }

    const int num_clusters =
        needs_rendezvous
            ? std::max(1, pair_count)
            : pair_count;

    const int num_blocks = num_clusters * ClusterBlocks;

    configure_overlap_dsm_kernel_once_for<ReduceApply, ElemBytes>(device);

    system::runtime::set_device(device);

    auto kernel =
        tma_overlap_fastcopy_dsm_rank_kernel_sm90<
            ReduceApply,
            ElemBytes>;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_blocks, 1, 1);
    config.blockDim = dim3(TMA_TWO_GPU_PEER_THREADS, 1, 1);
    config.dynamicSmemBytes = TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES;
    config.stream = stream;

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = ClusterBlocks;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    config.attrs = attrs;
    config.numAttrs = 1;

    return cudaLaunchKernelEx(
        &config,
        kernel,
        local_in,
        local_buf,
        peer_buf,
        count,
        rank,
        local_ready_signal,
        peer_ready_signal,
        collective_epoch,
        pair_count);
}

template <typename ReduceApply, int ElemBytes, bool Overlap>
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
    int collective_epoch) {
    if constexpr (Overlap) {
        return launch_overlap_dsm_rank_kernel_sm90<
            ReduceApply,
            ElemBytes>(
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
                collective_epoch);
    } else {
        return launch_seq_rank_kernel_sm90<
            ReduceApply,
            ElemBytes>(
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
                collective_epoch);
    }
}

template <typename ReduceOp, int ElemBytes, bool Overlap>
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
    int collective_epoch) {
    using ReduceApply = comm::PipelineTMAReduce<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        ReduceOp>;

    return launch_rank_kernel_sm90<ReduceApply, ElemBytes, Overlap>(
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
        collective_epoch);
}

template <bool Overlap>
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
    int collective_epoch) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
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
                    collective_epoch);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
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
                    collective_epoch);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
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
                    collective_epoch);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
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
                    collective_epoch);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
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
                    collective_epoch);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
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
                    collective_epoch);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                Overlap>(
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
                    collective_epoch);
        }
    }

    return cudaErrorInvalidValue;
}

cudaError_t validate_args(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    int rank,
    int dev0,
    int dev1) {
    if (local_in == nullptr || local_buf == nullptr || peer_buf == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (count == 0) {
        return cudaErrorInvalidValue;
    }

    if (rank != 0 && rank != 1) {
        return cudaErrorInvalidValue;
    }

    if (dev0 == dev1) {
        return cudaErrorInvalidValue;
    }

    if (dtype_size_bytes(dtype) == 0) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

template <bool Overlap>
cudaError_t enqueue_rank_kernel_sm90(
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
    int collective_epoch) {
    cudaError_t status = validate_args(
        local_in,
        local_buf,
        peer_buf,
        count,
        dtype,
        rank,
        dev0,
        dev1);

    if (status != cudaSuccess) {
        return status;
    }

    return dispatch_rank_kernel_sm90<Overlap>(
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
        collective_epoch);
}

} // namespace

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_seq_fastcopy_sm90(
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
    int collective_epoch) {
    return enqueue_rank_kernel_sm90<false>(
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
        collective_epoch);
}

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_overlap_fastcopy_sm90(
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
    int collective_epoch) {
    return enqueue_rank_kernel_sm90<true>(
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
        collective_epoch);
}

} // namespace ooverlap
