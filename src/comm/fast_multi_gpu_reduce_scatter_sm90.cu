#include "comm/fast_multi_gpu_reduce_scatter_sm90.h"

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

constexpr int kFastReduceScatterThreads = 32;
constexpr int kFastReduceScatterScratchSlots = 257;
constexpr int kReadySignalPhaseStride = 1024;
constexpr int kLoadedReadyPhase = 1;
constexpr int kReduceDoneReadyPhase = 2;
constexpr int kFastReduceScatterMaxPeers =
    TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS;

struct FastReduceScatterKernelParams {
    const void* local_src_ptrs[kFastReduceScatterMaxPeers] = {};
    void* peer_dst_ptrs[kFastReduceScatterMaxPeers] = {};
    std::uint32_t shard_bytes[kFastReduceScatterMaxPeers] = {};
    int first_cta[kFastReduceScatterMaxPeers] = {};
    int cta_count[kFastReduceScatterMaxPeers] = {};

    comm::kernels::MultiGpuReadySignalPlan<kFastReduceScatterMaxPeers>
        ready_plan{};

    int peer_count = 0;
    int collective_epoch = 0;
};

struct FastReduceScatterBarrierScratch {
    unsigned int* counter = nullptr;
    unsigned int* last_value = nullptr;
};

cudaError_t allocate_zeroed_fast_reduce_scatter_counter(
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

cudaError_t get_fast_reduce_scatter_barrier_scratch(
    int device,
    int scratch_index,
    FastReduceScatterBarrierScratch* out) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }

    *out = FastReduceScatterBarrierScratch{};

    if (device < 0 || device >= 32) {
        return cudaErrorInvalidDevice;
    }

    const int storage_index = scratch_index == -1 ? 0 : scratch_index;

    if (storage_index < 0 ||
        storage_index >= kFastReduceScatterScratchSlots) {
        return cudaErrorInvalidValue;
    }

    static std::mutex mutex;
    static unsigned int*
        counters[32][kFastReduceScatterScratchSlots] = {};
    static unsigned int
        last_values[32][kFastReduceScatterScratchSlots] = {};
    static std::atomic<bool>
        ready[32][kFastReduceScatterScratchSlots] = {};

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
            allocate_zeroed_fast_reduce_scatter_counter(
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

__device__ __forceinline__ void
arrive_and_wait_fast_reduce_scatter_counter(
    unsigned int* counter,
    unsigned int target) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> state(*counter);
    state.fetch_add(1u, cuda::memory_order_acq_rel);

    while (state.load(cuda::memory_order_acquire) < target) {
        __nanosleep(16);
    }
}

__device__ __forceinline__ int fast_reduce_scatter_ready_value(
    int collective_epoch,
    int phase) {
    return collective_epoch * kReadySignalPhaseStride + phase;
}

bool compute_rank_partition(
    size_t count,
    int rank,
    int world_size,
    size_t* out_offset,
    size_t* out_count) {
    if (out_offset == nullptr ||
        out_count == nullptr ||
        rank < 0 ||
        rank >= world_size ||
        world_size <= 0) {
        return false;
    }

    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = count / world;
    const size_t rem = count - base * world;

    *out_offset = r * base + ((r < rem) ? r : rem);
    *out_count = base + ((r < rem) ? 1u : 0u);
    return true;
}

template <typename ReduceOp>
__global__ void fast_multi_gpu_reduce_scatter_kernel_sm90(
    FastReduceScatterKernelParams params,
    unsigned int* grid_counter,
    unsigned int counter_base) {
    if (threadIdx.x != 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_chunk =
        reinterpret_cast<unsigned char*>(shared_storage_u4);
    __shared__ sync::semaphore load_barrier;

    int peer_idx = -1;
    int peer_chunk_idx = -1;
    const int cta_idx = static_cast<int>(blockIdx.x);

    for (int candidate = 0;
         candidate < params.peer_count;
         ++candidate) {
        const int begin = params.first_cta[candidate];
        const int end = begin + params.cta_count[candidate];

        if (cta_idx >= begin && cta_idx < end) {
            peer_idx = candidate;
            peer_chunk_idx = cta_idx - begin;
            break;
        }
    }

    /*
     * The host creates exactly one CTA for every destination-shard chunk, so
     * each CTA performs exactly one local load and later one peer reduction.
     */
    if (peer_idx < 0 || peer_chunk_idx < 0) {
        return;
    }

    const size_t chunk_offset =
        static_cast<size_t>(peer_chunk_idx) *
        static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES);
    const size_t total_shard_bytes =
        static_cast<size_t>(params.shard_bytes[peer_idx]);

    if (chunk_offset >= total_shard_bytes) {
        return;
    }

    const size_t remaining =
        total_shard_bytes - chunk_offset;
    const std::uint32_t chunk_bytes =
        static_cast<std::uint32_t>(
            remaining <
                    static_cast<size_t>(
                        TMA_TWO_GPU_PEER_SMALL_TASK_BYTES)
                ? remaining
                : static_cast<size_t>(
                    TMA_TWO_GPU_PEER_SMALL_TASK_BYTES));

    const unsigned char* local_src =
        reinterpret_cast<const unsigned char*>(
            params.local_src_ptrs[peer_idx]) +
        chunk_offset;
    unsigned char* peer_dst =
        reinterpret_cast<unsigned char*>(
            params.peer_dst_ptrs[peer_idx]) +
        chunk_offset;

    /* Phase 1: every CTA loads exactly one local shard chunk into its SMEM. */
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

    /* Keep the loaded SMEM chunk live while all local CTAs finish loading. */
    arrive_and_wait_fast_reduce_scatter_counter(
        grid_counter,
        loaded_target);

    /*
     * Every rank reaches this rendezvous only after all of its source chunks
     * are resident in CTA-local SMEM. It also proves that each destination
     * rank's preceding stream work has completed before remote reductions.
     */
    const int loaded_ready_value =
        fast_reduce_scatter_ready_value(
            params.collective_epoch,
            kLoadedReadyPhase);

    comm::kernels::distributed_ready_rendezvous_for_cta(
        params.ready_plan,
        loaded_ready_value);

    const unsigned int peers_loaded_target =
        loaded_target + cta_count;

    /*
     * The peer rendezvous is distributed over CTAs. Do not let any CTA reduce
     * until the rendezvous-owning CTAs have all completed it.
     */
    arrive_and_wait_fast_reduce_scatter_counter(
        grid_counter,
        peers_loaded_target);

    /* Phase 2: reduce the preserved local chunk into its owner peer's shard. */
    tma::reduce_fence_proxy_async_shared_cta();
    ReduceOp::template issue_bulk_op_nofence<
        tma::TmaReduceScope::Default>(
            peer_dst,
            shared_chunk,
            chunk_bytes);
    tma::reduce_commit_group();
    tma::reduce_async_wait<0>();

    const unsigned int reduce_done_target =
        peers_loaded_target + cta_count;

    arrive_and_wait_fast_reduce_scatter_counter(
        grid_counter,
        reduce_done_target);

    const int reduce_done_ready_value =
        fast_reduce_scatter_ready_value(
            params.collective_epoch,
            kReduceDoneReadyPhase);

    /*
     * Every local CTA has completed its remote reduction before any rank
     * publishes completion. Kernel completion therefore means this rank's
     * owner shard has received every peer contribution.
     */
    comm::kernels::distributed_ready_rendezvous_for_cta(
        params.ready_plan,
        reduce_done_ready_value);
}

template <typename ReduceOp>
cudaError_t configure_fast_reduce_scatter_kernel_once(int device) {
    if (device < 0 || device >= 32) {
        return cudaErrorInvalidDevice;
    }

    static std::mutex mutex;
    static std::atomic<unsigned int> configured_mask{0u};

    const unsigned int bit =
        1u << static_cast<unsigned int>(device);

    if ((configured_mask.load(std::memory_order_acquire) & bit) != 0u) {
        return cudaSuccess;
    }

    std::lock_guard<std::mutex> lock(mutex);

    if ((configured_mask.load(std::memory_order_relaxed) & bit) != 0u) {
        return cudaSuccess;
    }

    cudaDeviceProp prop{};
    cudaError_t error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaSuccess) {
        return error;
    }

    constexpr size_t dynamic_smem_bytes =
        static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES);
    constexpr size_t total_smem_bytes =
        dynamic_smem_bytes + sizeof(sync::semaphore);

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        return cudaErrorInvalidConfiguration;
    }

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        error = cudaFuncSetAttribute(
            fast_multi_gpu_reduce_scatter_kernel_sm90<ReduceOp>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dynamic_smem_bytes));

        if (error != cudaSuccess) {
            return error;
        }
    }

    error = cudaFuncSetAttribute(
        fast_multi_gpu_reduce_scatter_kernel_sm90<ReduceOp>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    if (error != cudaSuccess) {
        return error;
    }

    configured_mask.fetch_or(bit, std::memory_order_release);
    return cudaSuccess;
}

template <typename ReduceOp>
cudaError_t launch_fast_multi_gpu_reduce_scatter_typed(
    const comm::api::CollectiveLaunchState& launch,
    size_t count,
    cudaStream_t stream,
    int scratch_index) {
    if (launch.local_ptr == nullptr ||
        launch.out_of_place ||
        launch.rank < 0 ||
        launch.rank >= launch.world_size ||
        launch.world_size < 2 ||
        launch.peer_count != launch.world_size - 1 ||
        launch.peer_count <= 0 ||
        launch.peer_count > kFastReduceScatterMaxPeers ||
        launch.dtype_size == 0 ||
        launch.bytes == 0 ||
        count == 0 ||
        count > static_cast<size_t>(-1) / launch.dtype_size ||
        count * launch.dtype_size != launch.bytes ||
        launch.collective_epoch <= 0) {
        return cudaErrorInvalidValue;
    }

    FastReduceScatterKernelParams params{};
    params.peer_count = launch.peer_count;
    params.collective_epoch = launch.collective_epoch;
    params.ready_plan =
        comm::kernels::make_multi_gpu_ready_signal_plan<
            kFastReduceScatterMaxPeers>(
                launch.peer_count,
                nullptr,
                launch.peer_publish_signals,
                launch.local_wait_signals,
                comm::kernels::MultiGpuReadySignalProtocol::
                    DeviceMemoryStoreRelease,
                64);

    bool seen_rank[kOoMaxLocalDevices] = {};
    int num_ctas = 0;

    for (int peer_idx = 0;
         peer_idx < launch.peer_count;
         ++peer_idx) {
        const int peer_rank = launch.peer_ranks[peer_idx];

        if (peer_rank < 0 ||
            peer_rank >= launch.world_size ||
            peer_rank == launch.rank ||
            peer_rank >= kOoMaxLocalDevices ||
            seen_rank[peer_rank] ||
            launch.peer_ptrs[peer_idx] == nullptr ||
            launch.peer_publish_signals[peer_idx] == nullptr ||
            launch.local_wait_signals[peer_idx] == nullptr) {
            return cudaErrorInvalidValue;
        }

        seen_rank[peer_rank] = true;

        size_t shard_offset_elements = 0;
        size_t shard_count = 0;

        if (!compute_rank_partition(
                count,
                peer_rank,
                launch.world_size,
                &shard_offset_elements,
                &shard_count) ||
            shard_count == 0 ||
            shard_offset_elements >
                static_cast<size_t>(-1) / launch.dtype_size ||
            shard_count >
                static_cast<size_t>(-1) / launch.dtype_size) {
            return cudaErrorInvalidValue;
        }

        const size_t shard_offset_bytes =
            shard_offset_elements * launch.dtype_size;
        const size_t shard_bytes =
            shard_count * launch.dtype_size;

        if (shard_offset_bytes > launch.bytes ||
            shard_bytes > launch.bytes - shard_offset_bytes ||
            shard_bytes > static_cast<size_t>(UINT32_MAX) ||
            (shard_offset_bytes % 16) != 0 ||
            (shard_bytes % 16) != 0) {
            return cudaErrorInvalidValue;
        }

        const unsigned char* local_src =
            reinterpret_cast<const unsigned char*>(launch.local_ptr) +
            shard_offset_bytes;
        unsigned char* peer_dst =
            reinterpret_cast<unsigned char*>(launch.peer_ptrs[peer_idx]) +
            shard_offset_bytes;

        if ((reinterpret_cast<std::uintptr_t>(local_src) &
             static_cast<std::uintptr_t>(15)) != 0 ||
            (reinterpret_cast<std::uintptr_t>(peer_dst) &
             static_cast<std::uintptr_t>(15)) != 0) {
            return cudaErrorInvalidValue;
        }

        const int peer_ctas =
            static_cast<int>(
                (shard_bytes +
                 static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES) - 1) /
                static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES));

        if (peer_ctas <= 0 ||
            peer_ctas > kFastReduceScatterMaxCtas ||
            num_ctas > kFastReduceScatterMaxCtas - peer_ctas) {
            return cudaErrorInvalidConfiguration;
        }

        params.local_src_ptrs[peer_idx] = local_src;
        params.peer_dst_ptrs[peer_idx] = peer_dst;
        params.shard_bytes[peer_idx] =
            static_cast<std::uint32_t>(shard_bytes);
        params.first_cta[peer_idx] = num_ctas;
        params.cta_count[peer_idx] = peer_ctas;

        num_ctas += peer_ctas;
    }

    if (num_ctas <= 0 || num_ctas > kFastReduceScatterMaxCtas) {
        return cudaErrorInvalidConfiguration;
    }

    cudaError_t error =
        configure_fast_reduce_scatter_kernel_once<ReduceOp>(
            launch.local_device);
    if (error != cudaSuccess) {
        return error;
    }

    FastReduceScatterBarrierScratch scratch{};
    error =
        get_fast_reduce_scatter_barrier_scratch(
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
        static_cast<unsigned int>(3 * num_ctas);

    fast_multi_gpu_reduce_scatter_kernel_sm90<ReduceOp><<<
        num_ctas,
        kFastReduceScatterThreads,
        TMA_TWO_GPU_PEER_SMALL_TASK_BYTES,
        stream>>>(
            params,
            scratch.counter,
            counter_base);

#if !defined(NDEBUG) || defined(OOVERLAP_DEBUG_CUDA_LAUNCH_CHECKS)
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
#endif

    *scratch.last_value = counter_final;
    return cudaSuccess;
}

} // namespace

cudaError_t enqueue_fast_multi_gpu_reduce_scatter_rank_sm90(
    const comm::api::CollectiveLaunchState& launch,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    int scratch_index) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_fast_multi_gpu_reduce_scatter_typed<
                comm::pipeline::PipelineReduceAddNoFtzF16>(
                    launch,
                    count,
                    stream,
                    scratch_index);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_fast_multi_gpu_reduce_scatter_typed<
                comm::pipeline::PipelineReduceMinF16>(
                    launch,
                    count,
                    stream,
                    scratch_index);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_fast_multi_gpu_reduce_scatter_typed<
                comm::pipeline::PipelineReduceMaxF16>(
                    launch,
                    count,
                    stream,
                    scratch_index);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_fast_multi_gpu_reduce_scatter_typed<
                comm::pipeline::PipelineReduceAddBF16>(
                    launch,
                    count,
                    stream,
                    scratch_index);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_fast_multi_gpu_reduce_scatter_typed<
                comm::pipeline::PipelineReduceMinBF16>(
                    launch,
                    count,
                    stream,
                    scratch_index);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_fast_multi_gpu_reduce_scatter_typed<
                comm::pipeline::PipelineReduceMaxBF16>(
                    launch,
                    count,
                    stream,
                    scratch_index);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32 &&
        (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM)) {
        return launch_fast_multi_gpu_reduce_scatter_typed<
            comm::pipeline::PipelineReduceAddF32>(
                launch,
                count,
                stream,
                scratch_index);
    }

    return cudaErrorInvalidValue;
}

} // namespace ooverlap
