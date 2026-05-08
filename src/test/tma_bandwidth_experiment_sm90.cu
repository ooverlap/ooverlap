#include "test/tma_bandwidth_experiment_sm90.h"

#include "comm/kernels/fast_gmem_copy.cuh"
#include "comm/params.h"
#include "comm/pipeline/pipeline_stage.h"
#include "comm/pipeline/pipeline_tma_copy.h"
#include "comm/pipeline/pipeline_tma_load.h"
#include "comm/pipeline/pipeline_tma_reduce.h"

#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/two_gpu_test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ooverlap {
namespace {

constexpr int kTmaThreads = TMA_TWO_GPU_PEER_DEFAULT_THREADS;
constexpr int kFastThreads = 1024;

constexpr int kChunkBytes = TMA_TWO_GPU_PEER_CHUNK_BYTES;
constexpr int kStageDepth = TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH;
constexpr int kFillDepth = TMA_TWO_GPU_PEER_DEFAULT_STAGE_GAP;
constexpr int kBarrierCount = TMA_TWO_GPU_PEER_BARRIER_COUNT;

constexpr size_t kTmaSmemBytes =
    static_cast<size_t>(kStageDepth) * static_cast<size_t>(kChunkBytes);

static_assert(kStageDepth > 0, "stage depth must be positive");
static_assert(kFillDepth > 0, "fill depth must be positive");
static_assert(kFillDepth <= kStageDepth, "fill depth must be <= stage depth");
static_assert((kChunkBytes % sizeof(uint4)) == 0, "chunk bytes must be 16B aligned");

enum ExperimentId {
    kExperimentCopy = 0,
    kExperimentReduceAddF16 = 1,
};

enum ScenarioId {
    kScenarioLocalToPeer = 0,
    kScenarioPeerToLocal = 1,
};

enum MethodId {
    kMethodTmaCopy = 0,
    kMethodFastCopyU128 = 1,
    kMethodNcclSendRecv = 2,
    kMethodTmaReduceAddF16 = 3,
    kMethodFastAddF16U128 = 4,
};

struct ChunkRange {
    int start_chunk = 0;
    int chunk_count = 0;
};

struct DeviceBuffers {
    system::mapped_peer_buffer local{};
    system::mapped_peer_buffer peer{};
};

using KernelLaunchFn =
    cudaError_t (*)(
        const void*,
        void*,
        size_t,
        int,
        cudaStream_t);

__host__ __device__ __forceinline__ size_t min_size(
    size_t a,
    size_t b) {
    return a < b ? a : b;
}

__host__ __device__ __forceinline__ int ceil_div_size_to_int(
    size_t x,
    size_t y) {
    return static_cast<int>((x + y - 1) / y);
}

__host__ __device__ __forceinline__ ChunkRange make_block_chunk_range(
    int block_idx,
    int num_blocks,
    int num_chunks) {
    const int chunks_per_block =
        (num_chunks + num_blocks - 1) / num_blocks;

    ChunkRange range{};
    range.start_chunk = block_idx * chunks_per_block;
    range.chunk_count = chunks_per_block;

    if (range.start_chunk >= num_chunks) {
        range.chunk_count = 0;
        return range;
    }

    const int remaining = num_chunks - range.start_chunk;

    if (range.chunk_count > remaining) {
        range.chunk_count = remaining;
    }

    return range;
}

__host__ __forceinline__ bool is_aligned_16_host(
    const void* ptr) {
    return (
        (reinterpret_cast<uintptr_t>(ptr) &
         static_cast<uintptr_t>(sizeof(uint4) - 1)) == 0);
}

__host__ __forceinline__ bool is_aligned_16_size_host(
    size_t x) {
    return ((x & static_cast<size_t>(sizeof(uint4) - 1)) == 0);
}

__host__ __forceinline__ cudaError_t validate_tma_thread0_args(
    const void* src,
    const void* dst,
    size_t bytes) {
    if (bytes == 0) {
        return cudaSuccess;
    }

    if (src == nullptr || dst == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (!is_aligned_16_host(src) ||
        !is_aligned_16_host(dst) ||
        !is_aligned_16_size_host(bytes)) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

template <size_t ChunkBytes>
__device__ __forceinline__ size_t chunk_offset_bytes(
    int chunk) {
    return static_cast<size_t>(chunk) * ChunkBytes;
}

template <size_t ChunkBytes>
__device__ __forceinline__ comm::pipeline::PipelineStage make_stage_for_abs_chunk(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    size_t total_bytes,
    int abs_chunk,
    int slot,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const size_t offset =
        chunk_offset_bytes<ChunkBytes>(abs_chunk);

    const size_t remaining =
        total_bytes > offset ? total_bytes - offset : 0;

    const size_t bytes =
        remaining < ChunkBytes ? remaining : ChunkBytes;

    return comm::pipeline::make_pipeline_stage(
        comm::pipeline::make_pipeline_chunk(
            src_bytes + offset,
            dst_bytes + offset,
            bytes),
        shared_raw + static_cast<size_t>(slot) * ChunkBytes,
        &barriers[slot]);
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply>
__device__ __forceinline__ void run_tma_range_thread0_only(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    ChunkRange range,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");
    static_assert((ChunkBytes % sizeof(uint4)) == 0, "ChunkBytes must be 16B aligned");

    if (threadIdx.x != 0) {
        return;
    }

    if (range.chunk_count <= 0 || total_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    comm::pipeline::PipelineTMALoad load{};
    Apply apply{};

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= range.chunk_count) {
            break;
        }

        const int abs_chunk =
            range.start_chunk + warm;

        comm::pipeline::PipelineStage stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    for (int iter = 0; iter < range.chunk_count; ++iter) {
        const int abs_chunk =
            range.start_chunk + iter;

        const int cur_slot =
            iter % StageDepth;

        comm::pipeline::PipelineStage cur_stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + FillDepth;

        if (future_iter < range.chunk_count) {
            const int future_abs_chunk =
                range.start_chunk + future_iter;

            const int future_slot =
                future_iter % StageDepth;

            comm::pipeline::PipelineStage future_stage =
                make_stage_for_abs_chunk<ChunkBytes>(
                    src_bytes,
                    dst_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (iter >= FillDepth) {
                apply.wait_before_stage_reuse();
            }

            load.issue(&future_stage);
        }

        apply.issue_bulk(&cur_stage);
    }

    apply.wait_complete();
    __threadfence_system();
}

__global__ void tma_copy_kernel(
    const void* src,
    void* dst,
    size_t total_bytes,
    int num_chunks) {
    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kBarrierCount];

    using CopyApply =
        comm::pipeline::PipelineTMACopy<kStageDepth, kFillDepth>;

    run_tma_range_thread0_only<
        kStageDepth,
        kFillDepth,
        static_cast<size_t>(kChunkBytes),
        CopyApply>(
            src,
            dst,
            total_bytes,
            range,
            shared_raw,
            barriers);
}

__global__ void tma_reduce_add_f16_kernel(
    const void* src,
    void* dst,
    size_t total_bytes,
    int num_chunks) {
    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kBarrierCount];

    using ReduceApply =
        comm::pipeline::PipelineTMAReduce<
            kStageDepth,
            kFillDepth,
            comm::pipeline::PipelineReduceAddNoFtzF16>;

    run_tma_range_thread0_only<
        kStageDepth,
        kFillDepth,
        static_cast<size_t>(kChunkBytes),
        ReduceApply>(
            src,
            dst,
            total_bytes,
            range,
            shared_raw,
            barriers);
}

void configure_one_kernel(
    const void* kernel,
    size_t dynamic_smem_bytes,
    int device,
    const char* name) {
    system::runtime::set_device(device);

    cudaDeviceProp prop{};

    testing::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties");

    const size_t static_smem_bytes =
        static_cast<size_t>(kBarrierCount) * sizeof(sync::semaphore);

    const size_t total_smem_bytes =
        dynamic_smem_bytes + static_smem_bytes;

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            std::string(name) +
            ": requested shared memory exceeds opt-in limit");
    }

    if (dynamic_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        testing::check_cuda(
            cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    }

    testing::check_cuda(
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
}

void configure_kernels_once(int device) {
    static bool configured[32] = {};

    if (device >= 0 && device < 32 && configured[device]) {
        return;
    }

    configure_one_kernel(
        reinterpret_cast<const void*>(tma_copy_kernel),
        kTmaSmemBytes,
        device,
        "tma_copy_kernel");

    configure_one_kernel(
        reinterpret_cast<const void*>(tma_reduce_add_f16_kernel),
        kTmaSmemBytes,
        device,
        "tma_reduce_add_f16_kernel");

    configure_one_kernel(
        reinterpret_cast<const void*>(
            comm::kernels::fast_copy::gmem_copy_coalesced_kernel<uint4>),
        0,
        device,
        "gmem_copy_coalesced_kernel<uint4>");

    configure_one_kernel(
        reinterpret_cast<const void*>(
            comm::kernels::fast_copy::gmem_add_f16_u128_kernel<4>),
        0,
        device,
        "gmem_add_f16_u128_kernel<4>");

    if (device >= 0 && device < 32) {
        configured[device] = true;
    }
}

cudaError_t launch_tma_copy(
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_thread0_args(
            src,
            dst,
            bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    const int num_chunks =
        ceil_div_size_to_int(bytes, static_cast<size_t>(kChunkBytes));

    tma_copy_kernel<<<
        num_blocks,
        kTmaThreads,
        kTmaSmemBytes,
        stream>>>(
            src,
            dst,
            bytes,
            num_chunks);

    return cudaGetLastError();
}

cudaError_t launch_tma_reduce_add_f16(
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_thread0_args(
            src,
            dst,
            bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    const int num_chunks =
        ceil_div_size_to_int(bytes, static_cast<size_t>(kChunkBytes));

    tma_reduce_add_f16_kernel<<<
        num_blocks,
        kTmaThreads,
        kTmaSmemBytes,
        stream>>>(
            src,
            dst,
            bytes,
            num_chunks);

    return cudaGetLastError();
}

cudaError_t launch_fast_copy_u128(
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    comm::kernels::fast_copy::gmem_copy_coalesced_kernel<uint4><<<
        num_blocks,
        kFastThreads,
        0,
        stream>>>(
            src,
            dst,
            bytes);

    return cudaGetLastError();
}

cudaError_t launch_fast_add_f16_u128(
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    comm::kernels::fast_copy::gmem_add_f16_u128_kernel<4><<<
        num_blocks,
        kFastThreads,
        0,
        stream>>>(
            src,
            dst,
            bytes);

    return cudaGetLastError();
}

double benchmark_kernel_ms(
    KernelLaunchFn launch,
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    int kernel_device,
    int iters,
    int warmup) {
    system::runtime::set_device(kernel_device);

    cudaStream_t stream =
        system::runtime::create_stream_on_device(kernel_device);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try {
        for (int i = 0; i < warmup; ++i) {
            testing::check_cuda(
                launch(src, dst, bytes, num_blocks, stream),
                "launch warmup");
        }

        testing::check_cuda(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(warmup)");

        testing::check_cuda(
            cudaEventCreate(&start),
            "cudaEventCreate(start)");

        testing::check_cuda(
            cudaEventCreate(&stop),
            "cudaEventCreate(stop)");

        testing::check_cuda(
            cudaEventRecord(start, stream),
            "cudaEventRecord(start)");

        for (int i = 0; i < iters; ++i) {
            testing::check_cuda(
                launch(src, dst, bytes, num_blocks, stream),
                "launch timed");
        }

        testing::check_cuda(
            cudaEventRecord(stop, stream),
            "cudaEventRecord(stop)");

        testing::check_cuda(
            cudaEventSynchronize(stop),
            "cudaEventSynchronize(stop)");

        float total_ms = 0.0f;

        testing::check_cuda(
            cudaEventElapsedTime(&total_ms, start, stop),
            "cudaEventElapsedTime");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        system::runtime::destroy_stream_on_device(kernel_device, stream);

        return static_cast<double>(total_ms) / static_cast<double>(iters);
    } catch (...) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }

        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }

        system::runtime::destroy_stream_on_device(kernel_device, stream);
        throw;
    }
}

void fill_buffer(
    int device,
    void* ptr,
    int byte_value,
    size_t bytes) {
    system::runtime::set_device(device);

    cudaStream_t stream =
        system::runtime::create_stream_on_device(device);

    try {
        testing::check_cuda(
            cudaMemsetAsync(ptr, byte_value, bytes, stream),
            "cudaMemsetAsync");

        system::runtime::sync_stream_on_device(
            device,
            stream,
            "sync memset");

        system::runtime::destroy_stream_on_device(device, stream);
    } catch (...) {
        system::runtime::destroy_stream_on_device(device, stream);
        throw;
    }
}

double benchmark_nccl_sendrecv_ms(
    const void* src,
    void* dst,
    size_t bytes,
    int src_device,
    int dst_device,
    int src_rank,
    int dst_rank,
    ncclComm_t src_comm,
    ncclComm_t dst_comm,
    int iters,
    int warmup) {
    cudaStream_t src_stream =
        system::runtime::create_stream_on_device(src_device);

    cudaStream_t dst_stream =
        system::runtime::create_stream_on_device(dst_device);

    auto launch_once = [&]() {
        OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());

        system::runtime::set_device(src_device);

        OOVERLAP_TEST_NCCL_CHECK(
            ncclSend(
                src,
                bytes,
                ncclUint8,
                dst_rank,
                src_comm,
                src_stream));

        system::runtime::set_device(dst_device);

        OOVERLAP_TEST_NCCL_CHECK(
            ncclRecv(
                dst,
                bytes,
                ncclUint8,
                src_rank,
                dst_comm,
                dst_stream));

        OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
    };

    try {
        for (int i = 0; i < warmup; ++i) {
            launch_once();
        }

        testing::sync_two_streams(
            src_device,
            src_stream,
            dst_device,
            dst_stream,
            "sync NCCL sendrecv warmup");

        const double total_ms =
            testing::elapsed_ms_two_stream_max(
                src_device,
                src_stream,
                dst_device,
                dst_stream,
                iters,
                [&](int) {
                    launch_once();
                });

        system::runtime::destroy_stream_on_device(src_device, src_stream);
        system::runtime::destroy_stream_on_device(dst_device, dst_stream);

        return total_ms / static_cast<double>(iters);
    } catch (...) {
        system::runtime::destroy_stream_on_device(src_device, src_stream);
        system::runtime::destroy_stream_on_device(dst_device, dst_stream);
        throw;
    }
}

void add_result(
    std::vector<std::map<std::string, double>>& results,
    int experiment,
    int scenario,
    int method,
    int src_device,
    int dst_device,
    int kernel_device,
    size_t bytes,
    int num_blocks,
    double latency_ms) {
    const double seconds = latency_ms * 1.0e-3;

    const double gbps =
        seconds > 0.0
            ? static_cast<double>(bytes) / seconds / 1.0e9
            : 0.0;

    results.push_back({
        {"experiment", static_cast<double>(experiment)},
        {"scenario", static_cast<double>(scenario)},
        {"method", static_cast<double>(method)},
        {"src_device", static_cast<double>(src_device)},
        {"dst_device", static_cast<double>(dst_device)},
        {"kernel_device", static_cast<double>(kernel_device)},
        {"bytes", static_cast<double>(bytes)},
        {"num_blocks", static_cast<double>(num_blocks)},
        {"latency_ms", latency_ms},
        {"gbps", gbps},
    });
}

DeviceBuffers alloc_buffers(
    size_t bytes,
    int local_device,
    int peer_device) {
    std::vector<int> access_devices = {
        local_device,
        peer_device,
    };

    DeviceBuffers bufs{};

    bufs.local =
        system::alloc_peer_visible_buffer(
            bytes,
            local_device,
            access_devices);

    bufs.peer =
        system::alloc_peer_visible_buffer(
            bytes,
            peer_device,
            access_devices);

    return bufs;
}

void free_buffers(DeviceBuffers& bufs) {
    system::free_peer_visible_buffer(bufs.local);
    system::free_peer_visible_buffer(bufs.peer);
}

void run_copy_case(
    std::vector<std::map<std::string, double>>& results,
    int scenario,
    const void* src,
    void* dst,
    int src_device,
    int dst_device,
    int kernel_device,
    size_t bytes,
    int num_blocks,
    int iters,
    int warmup,
    bool include_nccl,
    ncclComm_t* comms) {
    configure_kernels_once(kernel_device);

    double ms =
        benchmark_kernel_ms(
            launch_tma_copy,
            src,
            dst,
            bytes,
            num_blocks,
            kernel_device,
            iters,
            warmup);

    add_result(
        results,
        kExperimentCopy,
        scenario,
        kMethodTmaCopy,
        src_device,
        dst_device,
        kernel_device,
        bytes,
        num_blocks,
        ms);

    ms =
        benchmark_kernel_ms(
            launch_fast_copy_u128,
            src,
            dst,
            bytes,
            num_blocks,
            kernel_device,
            iters,
            warmup);

    add_result(
        results,
        kExperimentCopy,
        scenario,
        kMethodFastCopyU128,
        src_device,
        dst_device,
        kernel_device,
        bytes,
        num_blocks,
        ms);

    if (!include_nccl || comms == nullptr || src_device == dst_device) {
        return;
    }

    int send_rank = -1;
    int recv_rank = -1;
    ncclComm_t send_comm = nullptr;
    ncclComm_t recv_comm = nullptr;

    if (src_device < dst_device) {
        send_rank = 0;
        recv_rank = 1;
        send_comm = comms[0];
        recv_comm = comms[1];
    } else {
        send_rank = 1;
        recv_rank = 0;
        send_comm = comms[1];
        recv_comm = comms[0];
    }

    ms =
        benchmark_nccl_sendrecv_ms(
            src,
            dst,
            bytes,
            src_device,
            dst_device,
            send_rank,
            recv_rank,
            send_comm,
            recv_comm,
            iters,
            warmup);

    add_result(
        results,
        kExperimentCopy,
        scenario,
        kMethodNcclSendRecv,
        src_device,
        dst_device,
        -1,
        bytes,
        num_blocks,
        ms);
}

void run_reduce_case(
    std::vector<std::map<std::string, double>>& results,
    int scenario,
    const void* src,
    void* dst,
    int src_device,
    int dst_device,
    int kernel_device,
    size_t bytes,
    int num_blocks,
    int iters,
    int warmup) {
    if ((bytes % sizeof(half)) != 0) {
        throw std::invalid_argument(
            "reduce experiment requires byte size divisible by sizeof(half)");
    }

    configure_kernels_once(kernel_device);

    double ms =
        benchmark_kernel_ms(
            launch_tma_reduce_add_f16,
            src,
            dst,
            bytes,
            num_blocks,
            kernel_device,
            iters,
            warmup);

    add_result(
        results,
        kExperimentReduceAddF16,
        scenario,
        kMethodTmaReduceAddF16,
        src_device,
        dst_device,
        kernel_device,
        bytes,
        num_blocks,
        ms);

    ms =
        benchmark_kernel_ms(
            launch_fast_add_f16_u128,
            src,
            dst,
            bytes,
            num_blocks,
            kernel_device,
            iters,
            warmup);

    add_result(
        results,
        kExperimentReduceAddF16,
        scenario,
        kMethodFastAddF16U128,
        src_device,
        dst_device,
        kernel_device,
        bytes,
        num_blocks,
        ms);
}

void run_one_size_and_block_count(
    std::vector<std::map<std::string, double>>& results,
    size_t bytes,
    int num_blocks,
    int iters,
    int warmup,
    int local_device,
    int peer_device,
    bool include_nccl,
    ncclComm_t* comms) {
    DeviceBuffers bufs =
        alloc_buffers(
            bytes,
            local_device,
            peer_device);

    try {
        fill_buffer(local_device, bufs.local.ptr, 1, bytes);
        fill_buffer(peer_device, bufs.peer.ptr, 2, bytes);

        run_copy_case(
            results,
            kScenarioLocalToPeer,
            bufs.local.ptr,
            bufs.peer.ptr,
            local_device,
            peer_device,
            local_device,
            bytes,
            num_blocks,
            iters,
            warmup,
            include_nccl,
            comms);

        run_reduce_case(
            results,
            kScenarioLocalToPeer,
            bufs.local.ptr,
            bufs.peer.ptr,
            local_device,
            peer_device,
            local_device,
            bytes,
            num_blocks,
            iters,
            warmup);

        run_copy_case(
            results,
            kScenarioPeerToLocal,
            bufs.peer.ptr,
            bufs.local.ptr,
            peer_device,
            local_device,
            local_device,
            bytes,
            num_blocks,
            iters,
            warmup,
            include_nccl,
            comms);

        run_reduce_case(
            results,
            kScenarioPeerToLocal,
            bufs.peer.ptr,
            bufs.local.ptr,
            peer_device,
            local_device,
            local_device,
            bytes,
            num_blocks,
            iters,
            warmup);

        free_buffers(bufs);
    } catch (...) {
        free_buffers(bufs);
        throw;
    }
}

std::vector<int64_t> make_power_of_two_sizes(
    int64_t min_bytes,
    int64_t max_bytes) {
    std::vector<int64_t> sizes;

    for (int64_t b = min_bytes; b <= max_bytes;) {
        sizes.push_back(b);

        if (b > max_bytes / 2) {
            break;
        }

        b *= 2;
    }

    return sizes;
}

void validate_sweep_args(
    const std::vector<int64_t>& sizes_bytes,
    const std::vector<int>& num_blocks_list,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    if (sizes_bytes.empty()) {
        throw std::invalid_argument("sizes_bytes must be non-empty");
    }

    if (num_blocks_list.empty()) {
        throw std::invalid_argument("num_blocks_list must be non-empty");
    }

    if (iters <= 0 || warmup < 0) {
        throw std::invalid_argument("invalid iteration counts");
    }

    if (dev0 < 0 || dev1 < 0 || dev0 == dev1) {
        throw std::invalid_argument("invalid devices");
    }

    for (int64_t bytes : sizes_bytes) {
        if (bytes <= 0) {
            throw std::invalid_argument("all sizes must be positive");
        }

        if (!is_aligned_16_size_host(static_cast<size_t>(bytes))) {
            throw std::invalid_argument("all sizes must be 16-byte aligned");
        }
    }

    for (int blocks : num_blocks_list) {
        if (blocks <= 0) {
            throw std::invalid_argument("all num_blocks values must be positive");
        }
    }
}

} // namespace

std::vector<std::map<std::string, double>>
benchmark_tma_bandwidth_experiment_sweep_sm90(
    const std::vector<int64_t>& sizes_bytes,
    const std::vector<int>& num_blocks_list,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    bool include_nccl) {
    validate_sweep_args(
        sizes_bytes,
        num_blocks_list,
        iters,
        warmup,
        dev0,
        dev1);

    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    std::vector<std::map<std::string, double>> results;

    ncclComm_t comms[2] = {
        nullptr,
        nullptr,
    };

    try {
        if (include_nccl) {
            int devices[2] = {
                dev0,
                dev1,
            };

            OOVERLAP_TEST_NCCL_CHECK(
                ncclCommInitAll(comms, 2, devices));
        }

        for (int64_t bytes_i : sizes_bytes) {
            const size_t bytes =
                static_cast<size_t>(bytes_i);

            for (int num_blocks : num_blocks_list) {
                run_one_size_and_block_count(
                    results,
                    bytes,
                    num_blocks,
                    iters,
                    warmup,
                    dev0,
                    dev1,
                    include_nccl,
                    include_nccl ? comms : nullptr);
            }
        }

        testing::destroy_nccl_comms(comms, 2);
        return results;
    } catch (...) {
        testing::destroy_nccl_comms(comms, 2);
        throw;
    }
}

std::vector<std::map<std::string, double>>
benchmark_tma_bandwidth_experiment_sm90(
    int64_t min_bytes,
    int64_t max_bytes,
    int iters,
    int warmup,
    int num_blocks,
    int dev0,
    int dev1,
    bool include_mem_async,
    bool include_nccl) {
    (void)include_mem_async;

    if (min_bytes <= 0 || max_bytes <= 0 || min_bytes > max_bytes) {
        throw std::invalid_argument("invalid byte range");
    }

    std::vector<int64_t> sizes =
        make_power_of_two_sizes(
            min_bytes,
            max_bytes);

    std::vector<int> blocks = {
        num_blocks,
    };

    return benchmark_tma_bandwidth_experiment_sweep_sm90(
        sizes,
        blocks,
        iters,
        warmup,
        dev0,
        dev1,
        include_nccl);
}

} // namespace ooverlap
