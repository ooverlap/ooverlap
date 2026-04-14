#include "overlap/persistent_allreduce_2gpu_sm90.h"

#include "overlap/tma_basic_collective_sm90.h"
#include "ooverlap/tma/tma.cuh"
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
#include <stdexcept>
#include <string>
#include <vector>

#define OOVERLAP_PERSIST_NCCL_CHECK(cmd)                                                        \
    do {                                                                                        \
        ncclResult_t result__ = (cmd);                                                          \
        if (result__ != ncclSuccess) {                                                          \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result__)); \
        }                                                                                       \
    } while (0)

namespace ooverlap {

// Keep these OUTSIDE the anonymous namespace so device code can see them.
static constexpr int kPersistentThreads = 256;
static constexpr size_t kPersistentChunkBytes = 16 * 1024;

namespace {

struct PersistentTwoGpuPeerState {
    system::mapped_peer_buffer send01;   // owned by dev1, written by dev0, read by dev1
    system::mapped_peer_buffer send10;   // owned by dev0, written by dev1, read by dev0
    system::mapped_peer_buffer sig01;    // owned by dev1, written by dev0, polled by dev1
    system::mapped_peer_buffer sig10;    // owned by dev0, written by dev1, polled by dev0
    uint64_t seq = 0;
};

__device__ __forceinline__ void load_chunk_to_smem(
    unsigned char* dst_smem,
    const unsigned char* src_gmem,
    size_t bytes) {
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        dst_smem[i] = src_gmem[i];
    }
}

__global__ void persistent_two_gpu_allreduce_kernel_sm90(
    const half* local_in,
    half* local_out,
    half* send_slot,
    const half* recv_slot,
    volatile unsigned long long* send_signal,
    const volatile unsigned long long* recv_signal,
    size_t numel,
    unsigned long long seq) {

    extern __shared__ unsigned char shared_raw[];
    unsigned char* smem0 = shared_raw;
    unsigned char* smem1 = shared_raw + kPersistentChunkBytes;

    const unsigned char* local_bytes = reinterpret_cast<const unsigned char*>(local_in);
    unsigned char* send_bytes = reinterpret_cast<unsigned char*>(send_slot);

    const size_t total_bytes = numel * sizeof(half);
    const size_t nchunks = (total_bytes + kPersistentChunkBytes - 1) / kPersistentChunkBytes;

    if (nchunks == 0) {
        return;
    }

    {
        const size_t offset = 0;
        const size_t bytes = min(kPersistentChunkBytes, total_bytes - offset);
        load_chunk_to_smem(smem0, local_bytes + offset, bytes);
    }
    __syncthreads();

    int cur = 0;
    for (size_t chunk = 0; chunk < nchunks; ++chunk) {
        unsigned char* cur_smem = (cur == 0) ? smem0 : smem1;
        unsigned char* next_smem = (cur == 0) ? smem1 : smem0;

        const size_t offset = chunk * kPersistentChunkBytes;
        const size_t bytes = min(kPersistentChunkBytes, total_bytes - offset);

        if (threadIdx.x == 0) {
            tma::store_async(send_bytes + offset, cur_smem, static_cast<uint32_t>(bytes));
        }

        if (chunk + 1 < nchunks) {
            const size_t next_offset = (chunk + 1) * kPersistentChunkBytes;
            const size_t next_bytes = min(kPersistentChunkBytes, total_bytes - next_offset);
            load_chunk_to_smem(next_smem, local_bytes + next_offset, next_bytes);
        }

        if (threadIdx.x == 0) {
            tma::store_async_read_wait<0>();
        }
        __syncthreads();
        cur ^= 1;
    }

    if (threadIdx.x == 0) {
        __threadfence_system();
        *send_signal = seq;
        __threadfence_system();
        while (*recv_signal < seq) {
        }
        __threadfence_system();
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < numel; idx += blockDim.x) {
        float a = __half2float(local_in[idx]);
        float b = __half2float(recv_slot[idx]);
        local_out[idx] = __float2half_rn(a + b);
    }
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

inline void alloc_persistent_peer_state(
    PersistentTwoGpuPeerState* st,
    int dev0,
    int dev1,
    size_t bytes) {

    if (st == nullptr) {
        throw std::invalid_argument("alloc_persistent_peer_state: state is null");
    }

    std::vector<int> access_devices = {dev0, dev1};

    st->send01 = system::alloc_peer_visible_buffer(bytes, dev1, access_devices);
    st->send10 = system::alloc_peer_visible_buffer(bytes, dev0, access_devices);
    st->sig01  = system::alloc_peer_visible_buffer(sizeof(uint64_t), dev1, access_devices);
    st->sig10  = system::alloc_peer_visible_buffer(sizeof(uint64_t), dev0, access_devices);
    st->seq = 0;

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMemset(st->sig01.ptr, 0, st->sig01.mapped_size), "cudaMemset(sig01)");
    system::runtime::check_cuda(cudaMemset(st->send01.ptr, 0, st->send01.mapped_size), "cudaMemset(send01)");

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMemset(st->sig10.ptr, 0, st->sig10.mapped_size), "cudaMemset(sig10)");
    system::runtime::check_cuda(cudaMemset(st->send10.ptr, 0, st->send10.mapped_size), "cudaMemset(send10)");
}

inline void free_persistent_peer_state(PersistentTwoGpuPeerState* st) {
    if (st == nullptr) {
        return;
    }
    system::free_peer_visible_buffer(st->send01);
    system::free_peer_visible_buffer(st->send10);
    system::free_peer_visible_buffer(st->sig01);
    system::free_peer_visible_buffer(st->sig10);
    st->seq = 0;
}

inline cudaError_t enqueue_persistent_two_gpu_allreduce_with_state(
    TmaCommunicator* comm,
    PersistentTwoGpuPeerState* st,
    half* rank0_in,
    half* rank1_in,
    half* rank0_out,
    half* rank1_out,
    size_t numel) {

    if (comm == nullptr || comm->world_size != 2) {
        return cudaErrorInvalidValue;
    }
    if (st == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (rank0_in == nullptr || rank1_in == nullptr ||
        rank0_out == nullptr || rank1_out == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (numel == 0 || numel > comm->max_full_numel) {
        return cudaErrorInvalidValue;
    }

    const size_t bytes = numel * sizeof(half);
    if (bytes > st->send01.mapped_size || bytes > st->send10.mapped_size) {
        return cudaErrorInvalidValue;
    }

    st->seq += 1;
    const uint64_t seq = st->seq;
    const size_t smem_bytes = 2 * kPersistentChunkBytes;

    system::runtime::set_device(comm->devices[0]);
    persistent_two_gpu_allreduce_kernel_sm90<<<1, kPersistentThreads, smem_bytes, comm->streams[0]>>>(
        rank0_in,
        rank0_out,
        reinterpret_cast<half*>(st->send01.ptr),
        reinterpret_cast<const half*>(st->send10.ptr),
        reinterpret_cast<volatile unsigned long long*>(st->sig01.ptr),
        reinterpret_cast<const volatile unsigned long long*>(st->sig10.ptr),
        numel,
        static_cast<unsigned long long>(seq));
    cudaError_t err0 = cudaGetLastError();

    system::runtime::set_device(comm->devices[1]);
    persistent_two_gpu_allreduce_kernel_sm90<<<1, kPersistentThreads, smem_bytes, comm->streams[1]>>>(
        rank1_in,
        rank1_out,
        reinterpret_cast<half*>(st->send10.ptr),
        reinterpret_cast<const half*>(st->send01.ptr),
        reinterpret_cast<volatile unsigned long long*>(st->sig10.ptr),
        reinterpret_cast<const volatile unsigned long long*>(st->sig01.ptr),
        numel,
        static_cast<unsigned long long>(seq));
    cudaError_t err1 = cudaGetLastError();

    return (err0 != cudaSuccess) ? err0 : err1;
}

} // namespace

cudaError_t enqueue_persistent_two_gpu_allreduce_sm90(
    TmaCommunicator* comm,
    half* rank0_in,
    half* rank1_in,
    half* rank0_out,
    half* rank1_out,
    size_t numel) {

    if (comm == nullptr || comm->world_size != 2) {
        return cudaErrorInvalidValue;
    }

    PersistentTwoGpuPeerState st{};
    alloc_persistent_peer_state(&st, comm->devices[0], comm->devices[1], numel * sizeof(half));

    cudaError_t err = enqueue_persistent_two_gpu_allreduce_with_state(
        comm,
        &st,
        rank0_in,
        rank1_in,
        rank0_out,
        rank1_out,
        numel);

    if (err == cudaSuccess) {
        sync_two_streams(comm->devices[0], comm->streams[0],
                         comm->devices[1], comm->streams[1],
                         "sync enqueue_persistent_two_gpu_allreduce_sm90");
    }

    free_persistent_peer_state(&st);
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

    TmaCommunicator comm{};
    communicator_init(&comm, {dev0, dev1}, static_cast<size_t>(numel), 1);

    PersistentTwoGpuPeerState st{};
    alloc_persistent_peer_state(&st, dev0, dev1, static_cast<size_t>(numel) * sizeof(half));

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

    half* rank0_out = buffer_as_half(communicator_get_local_full_buffer(&comm, 0));
    half* rank1_out = buffer_as_half(communicator_get_local_full_buffer(&comm, 1));

    system::runtime::check_cuda(
        enqueue_persistent_two_gpu_allreduce_with_state(
            &comm,
            &st,
            rank0_in,
            rank1_in,
            rank0_out,
            rank1_out,
            static_cast<size_t>(numel)),
        "enqueue_persistent_two_gpu_allreduce_with_state");

    sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync persistent allreduce");

    auto got0 = testing::copy_half_device_to_host_float(rank0_out, numel, dev0);
    auto got1 = testing::copy_half_device_to_host_float(rank1_out, numel, dev1);
    auto ref = reference_two_gpu_sum_fp16(numel);

    testing::expect_allclose(got0, ref, "persistent allreduce rank0");
    testing::expect_allclose(got1, ref, "persistent allreduce rank1");

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaFree(rank0_in), "cudaFree(rank0_in)");
    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaFree(rank1_in), "cudaFree(rank1_in)");

    free_persistent_peer_state(&st);
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

    TmaCommunicator comm{};
    communicator_init(&comm, {dev0, dev1}, static_cast<size_t>(numel), 1);

    PersistentTwoGpuPeerState st{};
    alloc_persistent_peer_state(&st, dev0, dev1, static_cast<size_t>(numel) * sizeof(half));

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

    half* rank0_out = buffer_as_half(communicator_get_local_full_buffer(&comm, 0));
    half* rank1_out = buffer_as_half(communicator_get_local_full_buffer(&comm, 1));

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

    for (int i = 0; i < warmup; ++i) {
        system::runtime::check_cuda(
            enqueue_persistent_two_gpu_allreduce_with_state(
                &comm,
                &st,
                rank0_in,
                rank1_in,
                rank0_out,
                rank1_out,
                static_cast<size_t>(numel)),
            "persistent warmup");
        sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync persistent warmup");
    }

    const double persistent_total_ms = elapsed_ms_two_stream_max(
        dev0, comm.streams[0], dev1, comm.streams[1], iters,
        [&](int) {
            system::runtime::check_cuda(
                enqueue_persistent_two_gpu_allreduce_with_state(
                    &comm,
                    &st,
                    rank0_in,
                    rank1_in,
                    rank0_out,
                    rank1_out,
                    static_cast<size_t>(numel)),
                "enqueue_persistent_two_gpu_allreduce_with_state");
        });

    ncclComm_t comms[2] = {nullptr, nullptr};
    int devices[2] = {dev0, dev1};
    OOVERLAP_PERSIST_NCCL_CHECK(ncclCommInitAll(comms, 2, devices));

    for (int i = 0; i < warmup; ++i) {
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(rank0_in, rank0_out, static_cast<size_t>(numel),
                          ncclFloat16, ncclSum, comms[0], comm.streams[0]));
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(rank1_in, rank1_out, static_cast<size_t>(numel),
                          ncclFloat16, ncclSum, comms[1], comm.streams[1]));
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
        sync_two_streams(dev0, comm.streams[0], dev1, comm.streams[1], "sync nccl warmup");
    }

    const double nccl_total_ms = elapsed_ms_two_stream_max(
        dev0, comm.streams[0], dev1, comm.streams[1], iters,
        [&](int) {
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(rank0_in, rank0_out, static_cast<size_t>(numel),
                              ncclFloat16, ncclSum, comms[0], comm.streams[0]));
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(rank1_in, rank1_out, static_cast<size_t>(numel),
                              ncclFloat16, ncclSum, comms[1], comm.streams[1]));
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
        });

    ncclCommDestroy(comms[0]);
    ncclCommDestroy(comms[1]);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaFree(rank0_in), "cudaFree(rank0_in)");
    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaFree(rank1_in), "cudaFree(rank1_in)");

    free_persistent_peer_state(&st);
    communicator_destroy(&comm);

    const double avg_basic_ms = basic_total_ms / static_cast<double>(iters);
    const double avg_persistent_ms = persistent_total_ms / static_cast<double>(iters);
    const double avg_nccl_ms = nccl_total_ms / static_cast<double>(iters);

    return {
        {"numel", static_cast<double>(numel)},
        {"avg_ms_basic", avg_basic_ms},
        {"avg_ms_persistent", avg_persistent_ms},
        {"avg_ms_nccl", avg_nccl_ms},
        {"speedup_basic_over_persistent", avg_basic_ms / avg_persistent_ms},
        {"speedup_nccl_over_persistent", avg_nccl_ms / avg_persistent_ms}
    };
}

} // namespace ooverlap
