#include "test/persistent_allreduce_2gpu_sm90.h"

#include "comm/ooverlap_comm.h"
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

#ifndef OOVERLAP_BENCH_VERIFY_RESULTS
#define OOVERLAP_BENCH_VERIFY_RESULTS 1
#endif

namespace ooverlap {
namespace {

const char* oo_status_string(oo_status_t status) {
    switch (status) {
        case OO_SUCCESS:
            return "OO_SUCCESS";
        case OO_ERROR_INVALID_ARGUMENT:
            return "OO_ERROR_INVALID_ARGUMENT";
        case OO_ERROR_INVALID_DEVICE:
            return "OO_ERROR_INVALID_DEVICE";
        case OO_ERROR_UNSUPPORTED:
            return "OO_ERROR_UNSUPPORTED";
        case OO_ERROR_CUDA:
            return "OO_ERROR_CUDA";
        case OO_ERROR_INTERNAL:
            return "OO_ERROR_INTERNAL";
        default:
            return "OO_ERROR_UNKNOWN";
    }
}

void check_oo(oo_status_t status, const char* what) {
    if (status != OO_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + " failed: " + oo_status_string(status));
    }
}

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

void reset_working_inputs_async(
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_work,
    half* rank1_work,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        cudaMemcpyAsync(rank0_work, rank0_src, bytes, cudaMemcpyDeviceToDevice, stream0),
        "cudaMemcpyAsync(rank0_src -> rank0_work)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaMemcpyAsync(rank1_work, rank1_src, bytes, cudaMemcpyDeviceToDevice, stream1),
        "cudaMemcpyAsync(rank1_src -> rank1_work)");
}

double elapsed_ms_oo_allreduce(
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    const half* rank0_src,
    const half* rank1_src,
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

    const size_t bytes = numel * sizeof(half);
    double total_ms = 0.0;

    half* rank0_work = reinterpret_cast<half*>(oo_buffer_ptr(rank0_buf));
    half* rank1_work = reinterpret_cast<half*>(oo_buffer_ptr(rank1_buf));

    for (int i = 0; i < iters; ++i) {
        reset_working_inputs_async(
            rank0_src,
            rank1_src,
            rank0_work,
            rank1_work,
            bytes,
            dev0,
            dev1,
            stream0,
            stream1);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaEventRecord(start0, stream0),
            "cudaEventRecord(start0)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaEventRecord(start1, stream1),
            "cudaEventRecord(start1)");

        check_oo(
            oo_allreduce(
                node0,
                rank0_buf,
                rank1_buf,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream0),
            "oo_allreduce(rank0)");

        check_oo(
            oo_allreduce(
                node1,
                rank1_buf,
                rank0_buf,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream1),
            "oo_allreduce(rank1)");

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

#if OOVERLAP_BENCH_VERIFY_RESULTS
void verify_two_gpu_allreduce_result(
    const char* label,
    half* rank0,
    half* rank1,
    int64_t numel,
    int dev0,
    int dev1) {
    auto got0 = testing::copy_half_device_to_host_float(rank0, numel, dev0);
    auto got1 = testing::copy_half_device_to_host_float(rank1, numel, dev1);
    auto ref = reference_two_gpu_sum_fp16(numel);

    testing::expect_allclose(got0, ref, (std::string(label) + " rank0").c_str());
    testing::expect_allclose(got1, ref, (std::string(label) + " rank1").c_str());
}
#endif

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

    oo_group_t* group = nullptr;
    oo_node_t* node0 = nullptr;
    oo_node_t* node1 = nullptr;
    oo_buffer_t* buf0 = nullptr;
    oo_buffer_t* buf1 = nullptr;
    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    try {
        int devices[2] = {dev0, dev1};

        check_oo(
            oo_group_create(devices, 2, &group),
            "oo_group_create");
        check_oo(
            oo_node_create(group, 0, &node0),
            "oo_node_create(rank0)");
        check_oo(
            oo_node_create(group, 1, &node1),
            "oo_node_create(rank1)");

        const int node0_dev = oo_node_device(node0);
        const int node1_dev = oo_node_device(node1);
        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

        stream0 = system::runtime::create_stream_on_device(node0_dev);
        stream1 = system::runtime::create_stream_on_device(node1_dev);

        check_oo(
            oo_buffer_alloc(node0, bytes, &buf0),
            "oo_buffer_alloc(rank0)");
        check_oo(
            oo_buffer_alloc(node1, bytes, &buf1),
            "oo_buffer_alloc(rank1)");

        half* rank0_buf = reinterpret_cast<half*>(oo_buffer_ptr(buf0));
        half* rank1_buf = reinterpret_cast<half*>(oo_buffer_ptr(buf1));

        fill_inputs(
            rank0_buf,
            rank1_buf,
            numel,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        check_oo(
            oo_allreduce(
                node0,
                buf0,
                buf1,
                static_cast<size_t>(numel),
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream0),
            "oo_allreduce(rank0 smoke)");

        check_oo(
            oo_allreduce(
                node1,
                buf1,
                buf0,
                static_cast<size_t>(numel),
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream1),
            "oo_allreduce(rank1 smoke)");

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync oo allreduce smoke");

        auto got0 = testing::copy_half_device_to_host_float(
            rank0_buf, numel, node0_dev);
        auto got1 = testing::copy_half_device_to_host_float(
            rank1_buf, numel, node1_dev);
        auto ref = reference_two_gpu_sum_fp16(numel);

        testing::expect_allclose(got0, ref, "oo allreduce rank0");
        testing::expect_allclose(got1, ref, "oo allreduce rank1");

        oo_buffer_destroy(buf0);
        oo_buffer_destroy(buf1);
        oo_node_destroy(node0);
        oo_node_destroy(node1);
        oo_group_destroy(group);
        system::runtime::destroy_stream_on_device(node0_dev, stream0);
        system::runtime::destroy_stream_on_device(node1_dev, stream1);

        return true;
    } catch (...) {
        int node0_dev = (node0 != nullptr) ? oo_node_device(node0) : dev0;
        int node1_dev = (node1 != nullptr) ? oo_node_device(node1) : dev1;

        if (buf0 != nullptr) {
            oo_buffer_destroy(buf0);
        }
        if (buf1 != nullptr) {
            oo_buffer_destroy(buf1);
        }
        if (node0 != nullptr) {
            oo_node_destroy(node0);
        }
        if (node1 != nullptr) {
            oo_node_destroy(node1);
        }
        if (group != nullptr) {
            oo_group_destroy(group);
        }
        if (stream0 != nullptr) {
            system::runtime::destroy_stream_on_device(node0_dev, stream0);
        }
        if (stream1 != nullptr) {
            system::runtime::destroy_stream_on_device(node1_dev, stream1);
        }
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

    oo_group_t* group = nullptr;
    oo_node_t* node0 = nullptr;
    oo_node_t* node1 = nullptr;
    oo_buffer_t* oo_rank0_buf = nullptr;
    oo_buffer_t* oo_rank1_buf = nullptr;
    oo_buffer_t* basic_inbox0 = nullptr;
    oo_buffer_t* basic_inbox1 = nullptr;

    half* basic_rank0 = nullptr;
    half* basic_rank1 = nullptr;
    half* nccl_rank0_out = nullptr;
    half* nccl_rank1_out = nullptr;
    half* rank0_src = nullptr;
    half* rank1_src = nullptr;

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    ncclComm_t comms[2] = {nullptr, nullptr};

    try {
        int devices[2] = {dev0, dev1};

        check_oo(
            oo_group_create(devices, 2, &group),
            "oo_group_create");
        check_oo(
            oo_node_create(group, 0, &node0),
            "oo_node_create(rank0)");
        check_oo(
            oo_node_create(group, 1, &node1),
            "oo_node_create(rank1)");

        const int node0_dev = oo_node_device(node0);
        const int node1_dev = oo_node_device(node1);
        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

        stream0 = system::runtime::create_stream_on_device(node0_dev);
        stream1 = system::runtime::create_stream_on_device(node1_dev);

        system::runtime::set_device(node0_dev);
        system::runtime::check_cuda(cudaMalloc(&basic_rank0, bytes), "cudaMalloc(basic_rank0)");
        system::runtime::check_cuda(cudaMalloc(&nccl_rank0_out, bytes), "cudaMalloc(nccl_rank0_out)");
        system::runtime::check_cuda(cudaMalloc(&rank0_src, bytes), "cudaMalloc(rank0_src)");

        system::runtime::set_device(node1_dev);
        system::runtime::check_cuda(cudaMalloc(&basic_rank1, bytes), "cudaMalloc(basic_rank1)");
        system::runtime::check_cuda(cudaMalloc(&nccl_rank1_out, bytes), "cudaMalloc(nccl_rank1_out)");
        system::runtime::check_cuda(cudaMalloc(&rank1_src, bytes), "cudaMalloc(rank1_src)");

        check_oo(
            oo_buffer_alloc(node0, bytes, &oo_rank0_buf),
            "oo_buffer_alloc(rank0 work)");
        check_oo(
            oo_buffer_alloc(node1, bytes, &oo_rank1_buf),
            "oo_buffer_alloc(rank1 work)");
        check_oo(
            oo_buffer_alloc(node0, bytes, &basic_inbox0),
            "oo_buffer_alloc(basic inbox0)");
        check_oo(
            oo_buffer_alloc(node1, bytes, &basic_inbox1),
            "oo_buffer_alloc(basic inbox1)");

        half* rank0_work = reinterpret_cast<half*>(oo_buffer_ptr(oo_rank0_buf));
        half* rank1_work = reinterpret_cast<half*>(oo_buffer_ptr(oo_rank1_buf));

        fill_inputs(
            rank0_src,
            rank1_src,
            numel,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        int nccl_devices[2] = {node0_dev, node1_dev};
        OOVERLAP_PERSIST_NCCL_CHECK(ncclCommInitAll(comms, 2, nccl_devices));

        // Warm up basic.
        system::runtime::set_device(node0_dev);
        system::runtime::check_cuda(
            cudaMemcpyAsync(basic_rank0, rank0_src, bytes, cudaMemcpyDeviceToDevice, stream0),
            "cudaMemcpyAsync(rank0_src -> basic_rank0)");
        system::runtime::set_device(node1_dev);
        system::runtime::check_cuda(
            cudaMemcpyAsync(basic_rank1, rank1_src, bytes, cudaMemcpyDeviceToDevice, stream1),
            "cudaMemcpyAsync(rank1_src -> basic_rank1)");
        sync_two_streams(node0_dev, stream0, node1_dev, stream1, "sync basic reset");

        for (int i = 0; i < warmup; ++i) {
            system::runtime::check_cuda(
                enqueue_two_gpu_all_reduce_tma_sm90(
                    basic_rank0,
                    basic_rank1,
                    reinterpret_cast<half*>(oo_buffer_ptr(basic_inbox0)),
                    reinterpret_cast<half*>(oo_buffer_ptr(basic_inbox1)),
                    static_cast<size_t>(numel),
                    node0_dev,
                    node1_dev,
                    stream0,
                    stream1),
                "enqueue_two_gpu_all_reduce_tma_sm90 warmup");
        }

        system::runtime::set_device(node0_dev);
        system::runtime::check_cuda(
            cudaMemcpyAsync(basic_rank0, rank0_src, bytes, cudaMemcpyDeviceToDevice, stream0),
            "cudaMemcpyAsync(rank0_src -> basic_rank0 reset)");
        system::runtime::set_device(node1_dev);
        system::runtime::check_cuda(
            cudaMemcpyAsync(basic_rank1, rank1_src, bytes, cudaMemcpyDeviceToDevice, stream1),
            "cudaMemcpyAsync(rank1_src -> basic_rank1 reset)");
        sync_two_streams(node0_dev, stream0, node1_dev, stream1, "sync basic reset timed");

        const double avg_basic_ms = elapsed_ms_host_avg(
            iters,
            [&](int) {
                system::runtime::check_cuda(
                    enqueue_two_gpu_all_reduce_tma_sm90(
                        basic_rank0,
                        basic_rank1,
                        reinterpret_cast<half*>(oo_buffer_ptr(basic_inbox0)),
                        reinterpret_cast<half*>(oo_buffer_ptr(basic_inbox1)),
                        static_cast<size_t>(numel),
                        node0_dev,
                        node1_dev,
                        stream0,
                        stream1),
                    "enqueue_two_gpu_all_reduce_tma_sm90");
            });

        // Warm up new public API path.
        for (int i = 0; i < warmup; ++i) {
            reset_working_inputs_async(
                rank0_src,
                rank1_src,
                rank0_work,
                rank1_work,
                bytes,
                node0_dev,
                node1_dev,
                stream0,
                stream1);

            check_oo(
                oo_allreduce(
                    node0,
                    oo_rank0_buf,
                    oo_rank1_buf,
                    static_cast<size_t>(numel),
                    OO_DTYPE_FLOAT16,
                    OO_REDUCE_SUM,
                    stream0),
                "oo_allreduce(rank0 warmup)");

            check_oo(
                oo_allreduce(
                    node1,
                    oo_rank1_buf,
                    oo_rank0_buf,
                    static_cast<size_t>(numel),
                    OO_DTYPE_FLOAT16,
                    OO_REDUCE_SUM,
                    stream1),
                "oo_allreduce(rank1 warmup)");

            sync_two_streams(node0_dev, stream0, node1_dev, stream1, "sync oo allreduce warmup");
        }

        const double oo_total_ms = elapsed_ms_oo_allreduce(
            node0,
            node1,
            oo_rank0_buf,
            oo_rank1_buf,
            rank0_src,
            rank1_src,
            static_cast<size_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            iters);

        // Warm up NCCL.
        for (int i = 0; i < warmup; ++i) {
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank0_src,
                    nccl_rank0_out,
                    static_cast<size_t>(numel),
                    ncclFloat16,
                    ncclSum,
                    comms[0],
                    stream0));
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank1_src,
                    nccl_rank1_out,
                    static_cast<size_t>(numel),
                    ncclFloat16,
                    ncclSum,
                    comms[1],
                    stream1));
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
            sync_two_streams(node0_dev, stream0, node1_dev, stream1, "sync NCCL warmup");
        }

        const double nccl_total_ms = elapsed_ms_two_stream_max(
            node0_dev, stream0, node1_dev, stream1, iters,
            [&](int) {
                OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
                OOVERLAP_PERSIST_NCCL_CHECK(
                    ncclAllReduce(
                        rank0_src,
                        nccl_rank0_out,
                        static_cast<size_t>(numel),
                        ncclFloat16,
                        ncclSum,
                        comms[0],
                        stream0));
                OOVERLAP_PERSIST_NCCL_CHECK(
                    ncclAllReduce(
                        rank1_src,
                        nccl_rank1_out,
                        static_cast<size_t>(numel),
                        ncclFloat16,
                        ncclSum,
                        comms[1],
                        stream1));
                OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
            });

#if OOVERLAP_BENCH_VERIFY_RESULTS
        // Verify basic TMA with a clean one-shot run after timing.
        reset_working_inputs_async(
            rank0_src,
            rank1_src,
            basic_rank0,
            basic_rank1,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        system::runtime::set_device(node0_dev);
        system::runtime::check_cuda(
            cudaMemsetAsync(oo_buffer_ptr(basic_inbox0), 0, bytes, stream0),
            "cudaMemsetAsync(basic_inbox0 verify)");
        system::runtime::set_device(node1_dev);
        system::runtime::check_cuda(
            cudaMemsetAsync(oo_buffer_ptr(basic_inbox1), 0, bytes, stream1),
            "cudaMemsetAsync(basic_inbox1 verify)");

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync basic verify reset");

        system::runtime::check_cuda(
            enqueue_two_gpu_all_reduce_tma_sm90(
                basic_rank0,
                basic_rank1,
                reinterpret_cast<half*>(oo_buffer_ptr(basic_inbox0)),
                reinterpret_cast<half*>(oo_buffer_ptr(basic_inbox1)),
                static_cast<size_t>(numel),
                node0_dev,
                node1_dev,
                stream0,
                stream1),
            "enqueue_two_gpu_all_reduce_tma_sm90 verify");

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync basic verify");

        verify_two_gpu_allreduce_result(
            "basic TMA benchmark verify",
            basic_rank0,
            basic_rank1,
            numel,
            node0_dev,
            node1_dev);

        // Verify public oo_allreduce with a clean one-shot run after timing.
        reset_working_inputs_async(
            rank0_src,
            rank1_src,
            rank0_work,
            rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync oo verify reset");

        check_oo(
            oo_allreduce(
                node0,
                oo_rank0_buf,
                oo_rank1_buf,
                static_cast<size_t>(numel),
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream0),
            "oo_allreduce(rank0 verify)");

        check_oo(
            oo_allreduce(
                node1,
                oo_rank1_buf,
                oo_rank0_buf,
                static_cast<size_t>(numel),
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream1),
            "oo_allreduce(rank1 verify)");

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync oo verify");

        verify_two_gpu_allreduce_result(
            "oo_allreduce benchmark verify",
            rank0_work,
            rank1_work,
            numel,
            node0_dev,
            node1_dev);

        // Verify NCCL with a clean one-shot run after timing.
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank0_src,
                nccl_rank0_out,
                static_cast<size_t>(numel),
                ncclFloat16,
                ncclSum,
                comms[0],
                stream0));
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank1_src,
                nccl_rank1_out,
                static_cast<size_t>(numel),
                ncclFloat16,
                ncclSum,
                comms[1],
                stream1));
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync NCCL verify");

        verify_two_gpu_allreduce_result(
            "NCCL benchmark verify",
            nccl_rank0_out,
            nccl_rank1_out,
            numel,
            node0_dev,
            node1_dev);
#endif

        ncclCommDestroy(comms[0]);
        ncclCommDestroy(comms[1]);
        comms[0] = nullptr;
        comms[1] = nullptr;

        system::runtime::set_device(node0_dev);
        system::runtime::check_cuda(cudaFree(basic_rank0), "cudaFree(basic_rank0)");
        system::runtime::check_cuda(cudaFree(nccl_rank0_out), "cudaFree(nccl_rank0_out)");
        system::runtime::check_cuda(cudaFree(rank0_src), "cudaFree(rank0_src)");
        basic_rank0 = nullptr;
        nccl_rank0_out = nullptr;
        rank0_src = nullptr;

        system::runtime::set_device(node1_dev);
        system::runtime::check_cuda(cudaFree(basic_rank1), "cudaFree(basic_rank1)");
        system::runtime::check_cuda(cudaFree(nccl_rank1_out), "cudaFree(nccl_rank1_out)");
        system::runtime::check_cuda(cudaFree(rank1_src), "cudaFree(rank1_src)");
        basic_rank1 = nullptr;
        nccl_rank1_out = nullptr;
        rank1_src = nullptr;

        oo_buffer_destroy(oo_rank0_buf);
        oo_buffer_destroy(oo_rank1_buf);
        oo_buffer_destroy(basic_inbox0);
        oo_buffer_destroy(basic_inbox1);
        oo_rank0_buf = nullptr;
        oo_rank1_buf = nullptr;
        basic_inbox0 = nullptr;
        basic_inbox1 = nullptr;

        oo_node_destroy(node0);
        oo_node_destroy(node1);
        oo_group_destroy(group);
        node0 = nullptr;
        node1 = nullptr;
        group = nullptr;

        system::runtime::destroy_stream_on_device(node0_dev, stream0);
        system::runtime::destroy_stream_on_device(node1_dev, stream1);
        stream0 = nullptr;
        stream1 = nullptr;

        const double avg_oo_ms =
            oo_total_ms / static_cast<double>(iters);
        const double avg_nccl_ms =
            nccl_total_ms / static_cast<double>(iters);

        return {
            {"numel", static_cast<double>(numel)},
            {"avg_ms_basic", avg_basic_ms},
            {"avg_ms_persistent", avg_oo_ms},
            {"avg_ms_oo_allreduce", avg_oo_ms},
            {"avg_ms_nccl", avg_nccl_ms},
            {"speedup_basic_over_persistent", avg_basic_ms / avg_oo_ms},
            {"speedup_nccl_over_persistent", avg_nccl_ms / avg_oo_ms},
            {"speedup_basic_over_oo_allreduce", avg_basic_ms / avg_oo_ms},
            {"speedup_nccl_over_oo_allreduce", avg_nccl_ms / avg_oo_ms},
            {"verify_results", static_cast<double>(OOVERLAP_BENCH_VERIFY_RESULTS)}
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
        if (rank0_src != nullptr) {
            system::runtime::set_device(dev0);
            cudaFree(rank0_src);
        }

        if (basic_rank1 != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(basic_rank1);
        }
        if (nccl_rank1_out != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(nccl_rank1_out);
        }
        if (rank1_src != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(rank1_src);
        }

        if (oo_rank0_buf != nullptr) {
            oo_buffer_destroy(oo_rank0_buf);
        }
        if (oo_rank1_buf != nullptr) {
            oo_buffer_destroy(oo_rank1_buf);
        }
        if (basic_inbox0 != nullptr) {
            oo_buffer_destroy(basic_inbox0);
        }
        if (basic_inbox1 != nullptr) {
            oo_buffer_destroy(basic_inbox1);
        }

        if (node0 != nullptr) {
            oo_node_destroy(node0);
        }
        if (node1 != nullptr) {
            oo_node_destroy(node1);
        }
        if (group != nullptr) {
            oo_group_destroy(group);
        }

        if (stream0 != nullptr) {
            system::runtime::destroy_stream_on_device(dev0, stream0);
        }
        if (stream1 != nullptr) {
            system::runtime::destroy_stream_on_device(dev1, stream1);
        }

        throw;
    }
}

} // namespace ooverlap
