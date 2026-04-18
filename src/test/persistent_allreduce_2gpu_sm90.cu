#include "test/persistent_allreduce_2gpu_sm90.h"

#include "test/tma_basic_collective_sm90.h"
#include "comm/chunk_pipeline.h"
#include "comm/range_chunk_scheduler.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define OOVERLAP_PERSIST_NCCL_CHECK(cmd)                                                        \
    do {                                                                                        \
        ncclResult_t result__ = (cmd);                                                          \
        if (result__ != ncclSuccess) {                                                          \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result__)); \
        }                                                                                       \
    } while (0)

namespace ooverlap {

static constexpr int kPersistentThreads = 16;
static constexpr size_t kPersistentChunkBytes = 16 * 1024;
static constexpr int kPersistentMaxBlocks = 16;
static constexpr int kPersistentStageDepth = 3;
static constexpr size_t kPersistentStaticSharedBytes =
    static_cast<size_t>(kPersistentStageDepth) * sizeof(sync::semaphore);

using PersistentScheduler = comm::RangeChunkScheduler;
using PersistentLoadOp = comm::PipelineTMALoad;
using PersistentReduceOp = comm::PipelineTMAReduceAddNoFtzF16;
using PersistentPipeline =
    comm::ChunkPipeline<
        kPersistentStageDepth,
        PersistentScheduler,
        PersistentLoadOp,
        PersistentReduceOp>;

namespace {

struct PersistentPeerBuffers {
    system::mapped_peer_buffer out0;  // owned by dev0, visible to dev0/dev1
    system::mapped_peer_buffer out1;  // owned by dev1, visible to dev0/dev1
};

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

inline int compute_num_chunks(size_t numel) {
    const size_t total_bytes = numel * sizeof(half);
    return static_cast<int>((total_bytes + kPersistentChunkBytes - 1) / kPersistentChunkBytes);
}

__device__ __forceinline__ void run_reduce_pipeline_for_block(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* load_barriers) {

    PersistentScheduler scheduler{};
    comm::chunk_scheduler_init(
        &scheduler,
        src_bytes,
        dst_bytes,
        total_bytes,
        kPersistentChunkBytes,
        static_cast<int>(blockIdx.x),
        static_cast<int>(gridDim.x));

    PersistentPipeline pipe{};
    comm::chunk_pipeline_bind_stage_storage<
        kPersistentStageDepth,
        kPersistentChunkBytes>(&pipe, shared_raw, load_barriers);
    comm::chunk_pipeline_init(&pipe, &scheduler);

    if (!comm::chunk_pipeline_try_prime(&pipe)) {
        return;
    }
    __syncthreads();

    while (comm::chunk_pipeline_has_current(&pipe)) {
        comm::chunk_pipeline_wait_current_stage(&pipe);
        __syncthreads();

        comm::chunk_pipeline_issue_current_reduce(&pipe);
        comm::chunk_pipeline_schedule_next_load(&pipe);
        __syncthreads();

        comm::chunk_pipeline_finish_current_tail(&pipe);
        __syncthreads();

        comm::chunk_pipeline_advance(&pipe);
    }
}

__global__ void persistent_two_gpu_reduce_to_peer_kernel_sm90(
    const half* local_in,
    half* peer_out,
    size_t numel,
    int num_chunks) {

    if (static_cast<int>(blockIdx.x) >= num_chunks) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw = reinterpret_cast<unsigned char*>(shared_storage_u4);
    __shared__ sync::semaphore load_barriers[kPersistentStageDepth];

    run_reduce_pipeline_for_block(
        reinterpret_cast<const unsigned char*>(local_in),
        reinterpret_cast<unsigned char*>(peer_out),
        numel * sizeof(half),
        shared_raw,
        load_barriers);
}

__global__ void persistent_two_gpu_reduce_to_peer_loop_kernel_sm90(
    const half* local_in,
    half* peer_out,
    size_t numel,
    int num_chunks,
    int iters) {

    if (static_cast<int>(blockIdx.x) >= num_chunks) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw = reinterpret_cast<unsigned char*>(shared_storage_u4);
    __shared__ sync::semaphore load_barriers[kPersistentStageDepth];

    const size_t total_bytes = numel * sizeof(half);

    for (int iter = 0; iter < iters; ++iter) {
        run_reduce_pipeline_for_block(
            reinterpret_cast<const unsigned char*>(local_in),
            reinterpret_cast<unsigned char*>(peer_out),
            total_bytes,
            shared_raw,
            load_barriers);
        __syncthreads();
    }
}

inline void configure_persistent_kernel_smem_once(int device, size_t dynamic_smem_bytes) {
    struct KernelConfigCacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, KernelConfigCacheEntry> cache;

    std::lock_guard<std::mutex> lock(mutex);

    auto it = cache.find(device);
    if (it != cache.end() &&
        it->second.configured &&
        it->second.dynamic_smem_bytes == dynamic_smem_bytes) {
        return;
    }

    system::runtime::set_device(device);

    cudaDeviceProp prop{};
    system::runtime::check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    const size_t total_smem_bytes = dynamic_smem_bytes + kPersistentStaticSharedBytes;
    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "persistent kernel requested total shared memory exceeds device opt-in limit");
    }

    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                persistent_two_gpu_reduce_to_peer_kernel_sm90,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize one-iter)");
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                persistent_two_gpu_reduce_to_peer_kernel_sm90,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout one-iter)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                persistent_two_gpu_reduce_to_peer_loop_kernel_sm90,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize loop)");
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                persistent_two_gpu_reduce_to_peer_loop_kernel_sm90,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout loop)");
    }

    cache[device] = {true, dynamic_smem_bytes};
}

inline void sync_two_streams(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    const char* what) {
    system::runtime::sync_stream_on_device(dev0, stream0, what);
    system::runtime::sync_stream_on_device(dev1, stream1, what);
}

inline double elapsed_ms_two_stream_max(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    int iters,
    const std::function<void(int)>& launch_once) {

    cudaEvent_t start0 = nullptr, stop0 = nullptr;
    cudaEvent_t start1 = nullptr, stop1 = nullptr;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaEventCreate(&start0), "cudaEventCreate(start0)");
    system::runtime::check_cuda(cudaEventCreate(&stop0), "cudaEventCreate(stop0)");
    system::runtime::check_cuda(cudaEventRecord(start0, stream0), "cudaEventRecord(start0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaEventCreate(&start1), "cudaEventCreate(start1)");
    system::runtime::check_cuda(cudaEventCreate(&stop1), "cudaEventCreate(stop1)");
    system::runtime::check_cuda(cudaEventRecord(start1, stream1), "cudaEventRecord(start1)");

    for (int i = 0; i < iters; ++i) {
        launch_once(i);
    }

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaEventRecord(stop0, stream0), "cudaEventRecord(stop0)");
    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaEventRecord(stop1, stream1), "cudaEventRecord(stop1)");

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaEventSynchronize(stop0), "cudaEventSynchronize(stop0)");
    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaEventSynchronize(stop1), "cudaEventSynchronize(stop1)");

    float ms0 = 0.0f, ms1 = 0.0f;
    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaEventElapsedTime(&ms0, start0, stop0), "cudaEventElapsedTime(ms0)");
    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaEventElapsedTime(&ms1, start1, stop1), "cudaEventElapsedTime(ms1)");

    system::runtime::set_device(dev0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);
    system::runtime::set_device(dev1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    return static_cast<double>(std::max(ms0, ms1));
}

inline std::vector<float> reference_two_gpu_sum_fp16(int64_t numel) {
    auto ref0 = testing::host_reference_pattern_fp16(numel, 0.25f, 1.0f);
    auto ref1 = testing::host_reference_pattern_fp16(numel, 0.50f, 2.0f);

    std::vector<float> ref(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        float acc = ref0[static_cast<size_t>(i)];
        acc = testing::round_to_half(acc + ref1[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = acc;
    }
    return ref;
}

inline void alloc_peer_buffers(
    PersistentPeerBuffers* peer,
    int dev0,
    int dev1,
    size_t bytes) {
    if (peer == nullptr) {
        throw std::invalid_argument("alloc_peer_buffers: peer is null");
    }

    std::vector<int> access_devices = {dev0, dev1};
    peer->out0 = system::alloc_peer_visible_buffer(bytes, dev0, access_devices);
    peer->out1 = system::alloc_peer_visible_buffer(bytes, dev1, access_devices);
}

inline void free_peer_buffers(PersistentPeerBuffers* peer) {
    if (peer == nullptr) {
        return;
    }
    system::free_peer_visible_buffer(peer->out0);
    system::free_peer_visible_buffer(peer->out1);
}

inline void seed_peer_outputs_from_local_inputs(
    const comm::Communicator* comm,
    const PersistentPeerBuffers* peer,
    const half* rank0_in,
    const half* rank1_in,
    size_t bytes) {

    system::runtime::set_device(comm->devices[0]);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            peer->out0.ptr,
            rank0_in,
            bytes,
            cudaMemcpyDeviceToDevice,
            comm->streams[0]),
        "cudaMemcpyAsync(rank0_in -> peer.out0)");

    system::runtime::set_device(comm->devices[1]);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            peer->out1.ptr,
            rank1_in,
            bytes,
            cudaMemcpyDeviceToDevice,
            comm->streams[1]),
        "cudaMemcpyAsync(rank1_in -> peer.out1)");

    sync_two_streams(
        comm->devices[0], comm->streams[0],
        comm->devices[1], comm->streams[1],
        "sync seed_peer_outputs_from_local_inputs");
}

inline cudaError_t launch_persistent_kernel_one_iteration(
    const comm::Communicator* comm,
    const PersistentPeerBuffers* peer,
    const half* rank0_in,
    const half* rank1_in,
    size_t numel) {

    const int num_chunks = compute_num_chunks(numel);
    const int num_blocks = std::min(kPersistentMaxBlocks, num_chunks);
    const size_t smem_bytes = static_cast<size_t>(kPersistentStageDepth) * kPersistentChunkBytes;

    system::runtime::set_device(comm->devices[0]);
    persistent_two_gpu_reduce_to_peer_kernel_sm90<<<
        num_blocks,
        kPersistentThreads,
        smem_bytes,
        comm->streams[0]>>>(
        rank0_in,
        reinterpret_cast<half*>(peer->out1.ptr),
        numel,
        num_chunks);
    cudaError_t err0 = cudaGetLastError();

    system::runtime::set_device(comm->devices[1]);
    persistent_two_gpu_reduce_to_peer_kernel_sm90<<<
        num_blocks,
        kPersistentThreads,
        smem_bytes,
        comm->streams[1]>>>(
        rank1_in,
        reinterpret_cast<half*>(peer->out0.ptr),
        numel,
        num_chunks);
    cudaError_t err1 = cudaGetLastError();

    return (err0 != cudaSuccess) ? err0 : err1;
}

inline cudaError_t launch_persistent_kernel_loop(
    const comm::Communicator* comm,
    const PersistentPeerBuffers* peer,
    const half* rank0_in,
    const half* rank1_in,
    size_t numel,
    int iters) {

    const int num_chunks = compute_num_chunks(numel);
    const int num_blocks = std::min(kPersistentMaxBlocks, num_chunks);
    const size_t smem_bytes = static_cast<size_t>(kPersistentStageDepth) * kPersistentChunkBytes;

    system::runtime::set_device(comm->devices[0]);
    persistent_two_gpu_reduce_to_peer_loop_kernel_sm90<<<
        num_blocks,
        kPersistentThreads,
        smem_bytes,
        comm->streams[0]>>>(
        rank0_in,
        reinterpret_cast<half*>(peer->out1.ptr),
        numel,
        num_chunks,
        iters);
    cudaError_t err0 = cudaGetLastError();

    system::runtime::set_device(comm->devices[1]);
    persistent_two_gpu_reduce_to_peer_loop_kernel_sm90<<<
        num_blocks,
        kPersistentThreads,
        smem_bytes,
        comm->streams[1]>>>(
        rank1_in,
        reinterpret_cast<half*>(peer->out0.ptr),
        numel,
        num_chunks,
        iters);
    cudaError_t err1 = cudaGetLastError();

    return (err0 != cudaSuccess) ? err0 : err1;
}

} // namespace

cudaError_t enqueue_persistent_two_gpu_allreduce_sm90(
    comm::Communicator* comm,
    half* rank0_in,
    half* rank1_in,
    half* rank0_out,
    half* rank1_out,
    size_t numel) {

    if (comm == nullptr || comm->world_size != 2) {
        return cudaErrorInvalidValue;
    }
    if (rank0_in == nullptr || rank1_in == nullptr ||
        rank0_out == nullptr || rank1_out == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    const size_t smem_bytes = static_cast<size_t>(kPersistentStageDepth) * kPersistentChunkBytes;
    configure_persistent_kernel_smem_once(comm->devices[0], smem_bytes);
    configure_persistent_kernel_smem_once(comm->devices[1], smem_bytes);

    PersistentPeerBuffers peer{};
    alloc_peer_buffers(&peer, comm->devices[0], comm->devices[1], numel * sizeof(half));
    seed_peer_outputs_from_local_inputs(
        comm,
        &peer,
        rank0_in,
        rank1_in,
        numel * sizeof(half));

    cudaError_t err = launch_persistent_kernel_one_iteration(
        comm,
        &peer,
        rank0_in,
        rank1_in,
        numel);

    if (err == cudaSuccess) {
        system::runtime::set_device(comm->devices[0]);
        system::runtime::check_cuda(
            cudaMemcpyAsync(
                rank0_out,
                peer.out0.ptr,
                numel * sizeof(half),
                cudaMemcpyDeviceToDevice,
                comm->streams[0]),
            "cudaMemcpyAsync(peer.out0 -> rank0_out)");

        system::runtime::set_device(comm->devices[1]);
        system::runtime::check_cuda(
            cudaMemcpyAsync(
                rank1_out,
                peer.out1.ptr,
                numel * sizeof(half),
                cudaMemcpyDeviceToDevice,
                comm->streams[1]),
            "cudaMemcpyAsync(peer.out1 -> rank1_out)");

        sync_two_streams(
            comm->devices[0], comm->streams[0],
            comm->devices[1], comm->streams[1],
            "sync enqueue_persistent_two_gpu_allreduce_sm90");
    }

    free_peer_buffers(&peer);
    return err;
}

bool tma_persistent_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {

    if (numel <= 0) {
        throw std::invalid_argument("tma_persistent_two_gpu_allreduce_smoke_test: numel must be > 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("tma_persistent_two_gpu_allreduce_smoke_test: dev0 and dev1 must differ");
    }

    comm::Communicator comm{};
    communicator_init(&comm, {dev0, dev1}, static_cast<size_t>(numel), 1);

    const size_t smem_bytes = static_cast<size_t>(kPersistentStageDepth) * kPersistentChunkBytes;
    configure_persistent_kernel_smem_once(dev0, smem_bytes);
    configure_persistent_kernel_smem_once(dev1, smem_bytes);

    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
    half* rank0_in = nullptr;
    half* rank1_in = nullptr;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&rank0_in, bytes), "cudaMalloc(rank0_in)");
    testing::fill_pattern(rank0_in, numel, 0.25f, 1.0f, comm.streams[0]);

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&rank1_in, bytes), "cudaMalloc(rank1_in)");
    testing::fill_pattern(rank1_in, numel, 0.50f, 2.0f, comm.streams[1]);

    sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync fill persistent smoke");

    PersistentPeerBuffers peer{};
    alloc_peer_buffers(&peer, dev0, dev1, bytes);
    seed_peer_outputs_from_local_inputs(&comm, &peer, rank0_in, rank1_in, bytes);

    system::runtime::check_cuda(
        launch_persistent_kernel_one_iteration(
            &comm,
            &peer,
            rank0_in,
            rank1_in,
            static_cast<size_t>(numel)),
        "launch_persistent_kernel_one_iteration");

    sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync persistent smoke kernel");

    auto got0 = testing::copy_half_device_to_host_float(
        reinterpret_cast<half*>(peer.out0.ptr), numel, dev0);
    auto got1 = testing::copy_half_device_to_host_float(
        reinterpret_cast<half*>(peer.out1.ptr), numel, dev1);
    auto ref = reference_two_gpu_sum_fp16(numel);

    testing::expect_allclose(got0, ref, "persistent allreduce rank0");
    testing::expect_allclose(got1, ref, "persistent allreduce rank1");

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaFree(rank0_in), "cudaFree(rank0_in)");
    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaFree(rank1_in), "cudaFree(rank1_in)");

    free_peer_buffers(&peer);
    communicator_destroy(&comm);
    return true;
}

std::map<std::string, double> benchmark_persistent_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {

    if (numel <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument("benchmark_persistent_two_gpu_allreduce_sm90: invalid args");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("benchmark_persistent_two_gpu_allreduce_sm90: dev0 and dev1 must differ");
    }

    comm::Communicator comm{};
    communicator_init(&comm, {dev0, dev1}, static_cast<size_t>(numel), 1);

    const size_t smem_bytes = static_cast<size_t>(kPersistentStageDepth) * kPersistentChunkBytes;
    configure_persistent_kernel_smem_once(dev0, smem_bytes);
    configure_persistent_kernel_smem_once(dev1, smem_bytes);

    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
    half* rank0_in = nullptr;
    half* rank1_in = nullptr;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&rank0_in, bytes), "cudaMalloc(rank0_in)");
    testing::fill_pattern(rank0_in, numel, 0.25f, 1.0f, comm.streams[0]);

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&rank1_in, bytes), "cudaMalloc(rank1_in)");
    testing::fill_pattern(rank1_in, numel, 0.50f, 2.0f, comm.streams[1]);

    sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync fill persistent bench");

    PersistentPeerBuffers peer{};
    alloc_peer_buffers(&peer, dev0, dev1, bytes);

    half* basic_rank0_out = buffer_as_half(communicator_get_local_full_buffer(&comm, 0));
    half* basic_rank1_out = buffer_as_half(communicator_get_local_full_buffer(&comm, 1));
    std::vector<half*> basic_inputs = {rank0_in, rank1_in};

    for (int i = 0; i < warmup; ++i) {
        system::runtime::check_cuda(
            enqueue_basic_all_reduce_tma_sm90(&comm, basic_inputs, static_cast<size_t>(numel)),
            "basic warmup");
        sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync basic warmup");
    }

    const double basic_total_ms = elapsed_ms_two_stream_max(
        dev0, comm.streams[0], dev1, comm.streams[1], iters,
        [&](int) {
            system::runtime::check_cuda(
                enqueue_basic_all_reduce_tma_sm90(&comm, basic_inputs, static_cast<size_t>(numel)),
                "enqueue_basic_all_reduce_tma_sm90");
        });

    seed_peer_outputs_from_local_inputs(&comm, &peer, rank0_in, rank1_in, bytes);
    for (int i = 0; i < warmup; ++i) {
        system::runtime::check_cuda(
            launch_persistent_kernel_one_iteration(
                &comm,
                &peer,
                rank0_in,
                rank1_in,
                static_cast<size_t>(numel)),
            "persistent warmup relaunch");
        sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync persistent warmup relaunch");
    }

    seed_peer_outputs_from_local_inputs(&comm, &peer, rank0_in, rank1_in, bytes);
    const double persistent_relaunch_total_ms = elapsed_ms_two_stream_max(
        dev0, comm.streams[0], dev1, comm.streams[1], iters,
        [&](int) {
            system::runtime::check_cuda(
                launch_persistent_kernel_one_iteration(
                    &comm,
                    &peer,
                    rank0_in,
                    rank1_in,
                    static_cast<size_t>(numel)),
                "launch_persistent_kernel_one_iteration");
        });

    seed_peer_outputs_from_local_inputs(&comm, &peer, rank0_in, rank1_in, bytes);
    if (warmup > 0) {
        system::runtime::check_cuda(
            launch_persistent_kernel_loop(
                &comm,
                &peer,
                rank0_in,
                rank1_in,
                static_cast<size_t>(numel),
                warmup),
            "persistent warmup single-launch");
        sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync persistent warmup single-launch");
    }

    seed_peer_outputs_from_local_inputs(&comm, &peer, rank0_in, rank1_in, bytes);
    const double persistent_single_launch_total_ms = elapsed_ms_two_stream_max(
        dev0, comm.streams[0], dev1, comm.streams[1], 1,
        [&](int) {
            system::runtime::check_cuda(
                launch_persistent_kernel_loop(
                    &comm,
                    &peer,
                    rank0_in,
                    rank1_in,
                    static_cast<size_t>(numel),
                    iters),
                "launch_persistent_kernel_loop");
        });

    ncclComm_t comms[2] = {nullptr, nullptr};
    int devices[2] = {dev0, dev1};
    OOVERLAP_PERSIST_NCCL_CHECK(ncclCommInitAll(comms, 2, devices));

    for (int i = 0; i < warmup; ++i) {
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(rank0_in, basic_rank0_out, static_cast<size_t>(numel),
                          ncclFloat16, ncclSum, comms[0], comm.streams[0]));
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(rank1_in, basic_rank1_out, static_cast<size_t>(numel),
                          ncclFloat16, ncclSum, comms[1], comm.streams[1]));
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
        sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync nccl warmup");
    }

    const double nccl_total_ms = elapsed_ms_two_stream_max(
        dev0, comm.streams[0], dev1, comm.streams[1], iters,
        [&](int) {
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(rank0_in, basic_rank0_out, static_cast<size_t>(numel),
                              ncclFloat16, ncclSum, comms[0], comm.streams[0]));
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(rank1_in, basic_rank1_out, static_cast<size_t>(numel),
                              ncclFloat16, ncclSum, comms[1], comm.streams[1]));
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
        });

    ncclCommDestroy(comms[0]);
    ncclCommDestroy(comms[1]);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaFree(rank0_in), "cudaFree(rank0_in)");
    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaFree(rank1_in), "cudaFree(rank1_in)");

    free_peer_buffers(&peer);
    communicator_destroy(&comm);

    const double avg_basic_ms = basic_total_ms / static_cast<double>(iters);
    const double avg_persistent_relaunch_ms =
        persistent_relaunch_total_ms / static_cast<double>(iters);
    const double avg_persistent_single_launch_ms =
        persistent_single_launch_total_ms / static_cast<double>(iters);
    const double avg_nccl_ms = nccl_total_ms / static_cast<double>(iters);

    return {
        {"numel", static_cast<double>(numel)},
        {"avg_ms_basic", avg_basic_ms},
        {"avg_ms_nccl", avg_nccl_ms},
        {"avg_ms_persistent", avg_persistent_single_launch_ms},
        {"avg_ms_persistent_relaunch", avg_persistent_relaunch_ms},
        {"avg_ms_persistent_single_launch", avg_persistent_single_launch_ms},
        {"speedup_basic_over_persistent", avg_basic_ms / avg_persistent_single_launch_ms},
        {"speedup_basic_over_persistent_relaunch", avg_basic_ms / avg_persistent_relaunch_ms},
        {"speedup_basic_over_persistent_single_launch", avg_basic_ms / avg_persistent_single_launch_ms},
        {"speedup_nccl_over_persistent", avg_nccl_ms / avg_persistent_single_launch_ms},
        {"speedup_nccl_over_persistent_relaunch", avg_nccl_ms / avg_persistent_relaunch_ms},
        {"speedup_nccl_over_persistent_single_launch", avg_nccl_ms / avg_persistent_single_launch_ms},
        {"speedup_persistent_relaunch_over_single_launch",
            avg_persistent_relaunch_ms / avg_persistent_single_launch_ms}
    };
}

} // namespace ooverlap
