#include "test/persistent_allreduce_2gpu_sm90.h"

#include "comm/group.h"
#include "comm/tma_two_gpu_peer_allreduce_sm90.h"
#include "test/tma_collective_sm90.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
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
namespace {

void sync_two_streams(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    const char* what) {
    system::runtime::sync_stream_on_device(dev0, stream0, what);
    system::runtime::sync_stream_on_device(dev1, stream1, what);
}

double elapsed_ms_two_stream_max(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    int iters,
    const std::function<void(int)>& launch_once) {

    cudaEvent_t start0 = nullptr;
    cudaEvent_t stop0 = nullptr;
    cudaEvent_t start1 = nullptr;
    cudaEvent_t stop1 = nullptr;

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

    float ms0 = 0.0f;
    float ms1 = 0.0f;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        cudaEventElapsedTime(&ms0, start0, stop0),
        "cudaEventElapsedTime(ms0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaEventElapsedTime(&ms1, start1, stop1),
        "cudaEventElapsedTime(ms1)");

    system::runtime::set_device(dev0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);

    system::runtime::set_device(dev1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    return static_cast<double>(std::max(ms0, ms1));
}

double elapsed_ms_host_avg(
    int iters,
    const std::function<void(int)>& launch_once) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        launch_once(i);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    return total_ms / static_cast<double>(iters);
}

double elapsed_ms_peer_persistent(
    TmaTwoGpuPeerAllreduceState* st,
    const half* rank0_in,
    const half* rank1_in,
    half* rank0_out_peer,
    half* rank1_out_peer,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {

    cudaEvent_t start0 = nullptr;
    cudaEvent_t stop0 = nullptr;
    cudaEvent_t start1 = nullptr;
    cudaEvent_t stop1 = nullptr;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaEventCreate(&start0), "cudaEventCreate(start0)");
    system::runtime::check_cuda(cudaEventCreate(&stop0), "cudaEventCreate(stop0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaEventCreate(&start1), "cudaEventCreate(start1)");
    system::runtime::check_cuda(cudaEventCreate(&stop1), "cudaEventCreate(stop1)");

    double total_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        system::runtime::check_cuda(
            prime_tma_two_gpu_peer_allreduce_outputs_sm90(
                st,
                rank0_in,
                rank1_in,
                rank0_out_peer,
                rank1_out_peer,
                numel,
                stream0,
                stream1),
            "prime_tma_two_gpu_peer_allreduce_outputs_sm90");

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaEventRecord(start0, stream0),
            "cudaEventRecord(start0)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaEventRecord(start1, stream1),
            "cudaEventRecord(start1)");

        system::runtime::check_cuda(
            enqueue_tma_two_gpu_peer_allreduce_kernel_only_sm90(
                st,
                rank0_in,
                rank1_in,
                rank0_out_peer,
                rank1_out_peer,
                numel,
                stream0,
                stream1),
            "enqueue_tma_two_gpu_peer_allreduce_kernel_only_sm90");

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaEventRecord(stop0, stream0),
            "cudaEventRecord(stop0)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaEventRecord(stop1, stream1),
            "cudaEventRecord(stop1)");

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaEventSynchronize(stop0),
            "cudaEventSynchronize(stop0)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaEventSynchronize(stop1),
            "cudaEventSynchronize(stop1)");

        float ms0 = 0.0f;
        float ms1 = 0.0f;

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaEventElapsedTime(&ms0, start0, stop0),
            "cudaEventElapsedTime(ms0)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaEventElapsedTime(&ms1, start1, stop1),
            "cudaEventElapsedTime(ms1)");

        total_ms += static_cast<double>(std::max(ms0, ms1));
    }

    system::runtime::set_device(dev0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);

    system::runtime::set_device(dev1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    return total_ms;
}

std::vector<float> reference_two_gpu_sum_fp16(int64_t numel) {
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

void fill_inputs(
    half* rank0,
    half* rank1,
    int64_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    system::runtime::set_device(dev0);
    testing::fill_pattern(rank0, numel, 0.25f, 1.0f, stream0);

    system::runtime::set_device(dev1);
    testing::fill_pattern(rank1, numel, 0.50f, 2.0f, stream1);

    sync_two_streams(dev0, stream0, dev1, stream1, "sync fill inputs");
}

} // namespace

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

    comm::Group group{};
    TmaTwoGpuPeerAllreduceState st{};

    try {
        comm::group_init(&group, {dev0, dev1});

        comm::Node* node0 = comm::group_get_node(&group, 0);
        comm::Node* node1 = comm::group_get_node(&group, 1);

        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

        const int in0_idx = comm::node_add_buffer(node0, bytes, group.devices);
        const int in1_idx = comm::node_add_buffer(node1, bytes, group.devices);
        const int out0_idx = comm::node_add_buffer(node0, bytes, group.devices);
        const int out1_idx = comm::node_add_buffer(node1, bytes, group.devices);

        comm::Buffer* in0 = comm::node_get_buffer(node0, in0_idx);
        comm::Buffer* in1 = comm::node_get_buffer(node1, in1_idx);
        comm::Buffer* out0 = comm::node_get_buffer(node0, out0_idx);
        comm::Buffer* out1 = comm::node_get_buffer(node1, out1_idx);

        fill_inputs(
            reinterpret_cast<half*>(comm::buffer_ptr(in0)),
            reinterpret_cast<half*>(comm::buffer_ptr(in1)),
            numel,
            node0->device,
            node1->device,
            node0->stream,
            node1->stream);

        tma_two_gpu_peer_allreduce_state_init(
            &st,
            node0->device,
            node1->device,
            static_cast<size_t>(numel));

        system::runtime::check_cuda(
            enqueue_tma_two_gpu_peer_allreduce_sm90(
                &st,
                reinterpret_cast<const half*>(comm::buffer_ptr(in0)),
                reinterpret_cast<const half*>(comm::buffer_ptr(in1)),
                reinterpret_cast<half*>(comm::buffer_ptr(out0)),
                reinterpret_cast<half*>(comm::buffer_ptr(out1)),
                static_cast<size_t>(numel),
                node0->stream,
                node1->stream),
            "enqueue_tma_two_gpu_peer_allreduce_sm90");

        sync_two_streams(
            node0->device,
            node0->stream,
            node1->device,
            node1->stream,
            "sync peer allreduce smoke");

        auto got0 = testing::copy_half_device_to_host_float(
            reinterpret_cast<half*>(comm::buffer_ptr(out0)), numel, node0->device);
        auto got1 = testing::copy_half_device_to_host_float(
            reinterpret_cast<half*>(comm::buffer_ptr(out1)), numel, node1->device);
        auto ref = reference_two_gpu_sum_fp16(numel);

        testing::expect_allclose(got0, ref, "tma persistent peer allreduce rank0");
        testing::expect_allclose(got1, ref, "tma persistent peer allreduce rank1");

        tma_two_gpu_peer_allreduce_state_destroy(&st);
        comm::group_destroy(&group);

        return true;
    } catch (...) {
        tma_two_gpu_peer_allreduce_state_destroy(&st);
        comm::group_destroy(&group);
        throw;
    }
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

    comm::Group group{};
    half* basic_rank0 = nullptr;
    half* basic_rank1 = nullptr;
    half* nccl_rank0_out = nullptr;
    half* nccl_rank1_out = nullptr;
    TmaTwoGpuPeerAllreduceState st{};
    ncclComm_t comms[2] = {nullptr, nullptr};

    try {
        comm::group_init(&group, {dev0, dev1});

        comm::Node* node0 = comm::group_get_node(&group, 0);
        comm::Node* node1 = comm::group_get_node(&group, 1);

        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

        system::runtime::set_device(node0->device);
        system::runtime::check_cuda(cudaMalloc(&basic_rank0, bytes), "cudaMalloc(basic_rank0)");
        system::runtime::check_cuda(cudaMalloc(&nccl_rank0_out, bytes), "cudaMalloc(nccl_rank0_out)");

        system::runtime::set_device(node1->device);
        system::runtime::check_cuda(cudaMalloc(&basic_rank1, bytes), "cudaMalloc(basic_rank1)");
        system::runtime::check_cuda(cudaMalloc(&nccl_rank1_out, bytes), "cudaMalloc(nccl_rank1_out)");

        const int in0_idx = comm::node_add_buffer(node0, bytes, group.devices);
        const int in1_idx = comm::node_add_buffer(node1, bytes, group.devices);
        const int basic_inbox0_idx = comm::node_add_buffer(node0, bytes, group.devices);
        const int basic_inbox1_idx = comm::node_add_buffer(node1, bytes, group.devices);
        const int out0_idx = comm::node_add_buffer(node0, bytes, group.devices);
        const int out1_idx = comm::node_add_buffer(node1, bytes, group.devices);

        comm::Buffer* in0 = comm::node_get_buffer(node0, in0_idx);
        comm::Buffer* in1 = comm::node_get_buffer(node1, in1_idx);
        comm::Buffer* basic_inbox0 = comm::node_get_buffer(node0, basic_inbox0_idx);
        comm::Buffer* basic_inbox1 = comm::node_get_buffer(node1, basic_inbox1_idx);
        comm::Buffer* out0 = comm::node_get_buffer(node0, out0_idx);
        comm::Buffer* out1 = comm::node_get_buffer(node1, out1_idx);

        const half* rank0_in =
            reinterpret_cast<const half*>(comm::buffer_ptr(in0));
        const half* rank1_in =
            reinterpret_cast<const half*>(comm::buffer_ptr(in1));

        fill_inputs(
            reinterpret_cast<half*>(comm::buffer_ptr(in0)),
            reinterpret_cast<half*>(comm::buffer_ptr(in1)),
            numel,
            node0->device,
            node1->device,
            node0->stream,
            node1->stream);

        tma_two_gpu_peer_allreduce_state_init(
            &st,
            node0->device,
            node1->device,
            static_cast<size_t>(numel));

        int devices[2] = {node0->device, node1->device};
        OOVERLAP_PERSIST_NCCL_CHECK(ncclCommInitAll(comms, 2, devices));

        // Warm up basic.
        system::runtime::set_device(node0->device);
        system::runtime::check_cuda(
            cudaMemcpyAsync(basic_rank0, rank0_in, bytes, cudaMemcpyDeviceToDevice, node0->stream),
            "cudaMemcpyAsync(rank0_in -> basic_rank0)");
        system::runtime::set_device(node1->device);
        system::runtime::check_cuda(
            cudaMemcpyAsync(basic_rank1, rank1_in, bytes, cudaMemcpyDeviceToDevice, node1->stream),
            "cudaMemcpyAsync(rank1_in -> basic_rank1)");
        sync_two_streams(node0->device, node0->stream, node1->device, node1->stream, "sync basic reset");

        for (int i = 0; i < warmup; ++i) {
            system::runtime::check_cuda(
                enqueue_two_gpu_all_reduce_tma_sm90(
                    basic_rank0,
                    basic_rank1,
                    reinterpret_cast<half*>(comm::buffer_ptr(basic_inbox0)),
                    reinterpret_cast<half*>(comm::buffer_ptr(basic_inbox1)),
                    static_cast<size_t>(numel),
                    node0->device,
                    node1->device,
                    node0->stream,
                    node1->stream),
                "enqueue_two_gpu_all_reduce_tma_sm90 warmup");
        }

        system::runtime::set_device(node0->device);
        system::runtime::check_cuda(
            cudaMemcpyAsync(basic_rank0, rank0_in, bytes, cudaMemcpyDeviceToDevice, node0->stream),
            "cudaMemcpyAsync(rank0_in -> basic_rank0 reset)");
        system::runtime::set_device(node1->device);
        system::runtime::check_cuda(
            cudaMemcpyAsync(basic_rank1, rank1_in, bytes, cudaMemcpyDeviceToDevice, node1->stream),
            "cudaMemcpyAsync(rank1_in -> basic_rank1 reset)");
        sync_two_streams(node0->device, node0->stream, node1->device, node1->stream, "sync basic reset timed");

        const double avg_basic_ms = elapsed_ms_host_avg(
            iters,
            [&](int) {
                system::runtime::check_cuda(
                    enqueue_two_gpu_all_reduce_tma_sm90(
                        basic_rank0,
                        basic_rank1,
                        reinterpret_cast<half*>(comm::buffer_ptr(basic_inbox0)),
                        reinterpret_cast<half*>(comm::buffer_ptr(basic_inbox1)),
                        static_cast<size_t>(numel),
                        node0->device,
                        node1->device,
                        node0->stream,
                        node1->stream),
                    "enqueue_two_gpu_all_reduce_tma_sm90");
            });

        // Warm up persistent path.
        for (int i = 0; i < warmup; ++i) {
            system::runtime::check_cuda(
                enqueue_tma_two_gpu_peer_allreduce_sm90(
                    &st,
                    rank0_in,
                    rank1_in,
                    reinterpret_cast<half*>(comm::buffer_ptr(out0)),
                    reinterpret_cast<half*>(comm::buffer_ptr(out1)),
                    static_cast<size_t>(numel),
                    node0->stream,
                    node1->stream),
                "enqueue_tma_two_gpu_peer_allreduce_sm90 warmup");
            sync_two_streams(node0->device, node0->stream, node1->device, node1->stream, "sync peer allreduce warmup");
        }

        const double persistent_total_ms = elapsed_ms_peer_persistent(
            &st,
            rank0_in,
            rank1_in,
            reinterpret_cast<half*>(comm::buffer_ptr(out0)),
            reinterpret_cast<half*>(comm::buffer_ptr(out1)),
            static_cast<size_t>(numel),
            node0->device,
            node1->device,
            node0->stream,
            node1->stream,
            iters);

        // Warm up NCCL.
        for (int i = 0; i < warmup; ++i) {
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank0_in,
                    nccl_rank0_out,
                    static_cast<size_t>(numel),
                    ncclFloat16,
                    ncclSum,
                    comms[0],
                    node0->stream));
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank1_in,
                    nccl_rank1_out,
                    static_cast<size_t>(numel),
                    ncclFloat16,
                    ncclSum,
                    comms[1],
                    node1->stream));
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
            sync_two_streams(node0->device, node0->stream, node1->device, node1->stream, "sync NCCL warmup");
        }

        const double nccl_total_ms = elapsed_ms_two_stream_max(
            node0->device, node0->stream, node1->device, node1->stream, iters,
            [&](int) {
                OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
                OOVERLAP_PERSIST_NCCL_CHECK(
                    ncclAllReduce(
                        rank0_in,
                        nccl_rank0_out,
                        static_cast<size_t>(numel),
                        ncclFloat16,
                        ncclSum,
                        comms[0],
                        node0->stream));
                OOVERLAP_PERSIST_NCCL_CHECK(
                    ncclAllReduce(
                        rank1_in,
                        nccl_rank1_out,
                        static_cast<size_t>(numel),
                        ncclFloat16,
                        ncclSum,
                        comms[1],
                        node1->stream));
                OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
            });

        ncclCommDestroy(comms[0]);
        ncclCommDestroy(comms[1]);
        comms[0] = nullptr;
        comms[1] = nullptr;

        system::runtime::set_device(node0->device);
        system::runtime::check_cuda(cudaFree(basic_rank0), "cudaFree(basic_rank0)");
        system::runtime::check_cuda(cudaFree(nccl_rank0_out), "cudaFree(nccl_rank0_out)");
        basic_rank0 = nullptr;
        nccl_rank0_out = nullptr;

        system::runtime::set_device(node1->device);
        system::runtime::check_cuda(cudaFree(basic_rank1), "cudaFree(basic_rank1)");
        system::runtime::check_cuda(cudaFree(nccl_rank1_out), "cudaFree(nccl_rank1_out)");
        basic_rank1 = nullptr;
        nccl_rank1_out = nullptr;

        tma_two_gpu_peer_allreduce_state_destroy(&st);
        comm::group_destroy(&group);

        const double avg_persistent_ms =
            persistent_total_ms / static_cast<double>(iters);
        const double avg_nccl_ms =
            nccl_total_ms / static_cast<double>(iters);

        return {
            {"numel", static_cast<double>(numel)},
            {"avg_ms_basic", avg_basic_ms},
            {"avg_ms_persistent", avg_persistent_ms},
            {"avg_ms_nccl", avg_nccl_ms},
            {"speedup_basic_over_persistent", avg_basic_ms / avg_persistent_ms},
            {"speedup_nccl_over_persistent", avg_nccl_ms / avg_persistent_ms}
        };
    } catch (...) {
        if (comms[0] != nullptr) {
            ncclCommDestroy(comms[0]);
        }
        if (comms[1] != nullptr) {
            ncclCommDestroy(comms[1]);
        }

        if (basic_rank0 != nullptr) {
            system::runtime::set_device(dev0);
            cudaFree(basic_rank0);
        }
        if (nccl_rank0_out != nullptr) {
            system::runtime::set_device(dev0);
            cudaFree(nccl_rank0_out);
        }

        if (basic_rank1 != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(basic_rank1);
        }
        if (nccl_rank1_out != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(nccl_rank1_out);
        }

        tma_two_gpu_peer_allreduce_state_destroy(&st);
        comm::group_destroy(&group);
        throw;
    }
}

} // namespace ooverlap
