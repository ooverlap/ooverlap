#include "test/persistent_external_p2p_collective_2gpu_sm90.h"

#include "ooverlap/comm.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/collective_test_utils.cuh"
#include "ooverlap/testing/two_gpu_test_utils.cuh"

#include "test/internal_comm_test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

#ifndef OOVERLAP_BENCH_VERIFY_RESULTS
#define OOVERLAP_BENCH_VERIFY_RESULTS 1
#endif

namespace ooverlap {
namespace {

using testing::TestCollective;

enum class BenchMode {
    BestPerformance = 0,
    BestEfficiency = 1,
};

oo_tuning_mode_t tuning_mode_for(BenchMode mode) {
    switch (mode) {
        case BenchMode::BestPerformance:
            return OO_TUNING_BEST_PERFORMANCE;
        case BenchMode::BestEfficiency:
            return OO_TUNING_BEST_EFFICIENCY;
        default:
            throw std::invalid_argument("unsupported benchmark mode");
    }
}

void launch_ooverlap_public_once_for_rank(
    TestCollective collective,
    BenchMode mode,
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t numel,
    cudaStream_t stream,
    const char* label) {
    oo_buffer_t* peers[] = {
        peer,
    };

    const oo_tuning_mode_t tuning_mode =
        tuning_mode_for(mode);

    if (collective == TestCollective::AllReduce) {
        testing::check_oo(
            oo_allreduce_tuned(
                node,
                local,
                peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                tuning_mode,
                stream),
            label);
        return;
    }

    if (collective == TestCollective::ReduceScatter) {
        oo_tensor_slice_t slice{};

        testing::check_oo(
            oo_reduce_scatter_tuned(
                node,
                local,
                peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                tuning_mode,
                &slice,
                stream),
            label);
        return;
    }

    if (collective == TestCollective::AllGather) {
        testing::check_oo(
            oo_all_gather_tuned(
                node,
                local,
                peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                tuning_mode,
                stream),
            label);
        return;
    }

    throw std::invalid_argument("unsupported collective");
}

void launch_ooverlap_public_once(
    TestCollective collective,
    BenchMode mode,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    launch_ooverlap_public_once_for_rank(
        collective,
        mode,
        node0,
        rank0_buf,
        rank1_buf,
        numel,
        stream0,
        "ooverlap rank0");

    launch_ooverlap_public_once_for_rank(
        collective,
        mode,
        node1,
        rank1_buf,
        rank0_buf,
        numel,
        stream1,
        "ooverlap rank1");
}

void run_ooverlap_public_iters(
    TestCollective collective,
    BenchMode mode,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    if (iters <= 0) {
        return;
    }

    testing::reset_ready_signals(group);

    for (int i = 0; i < iters; ++i) {
        launch_ooverlap_public_once(
            collective,
            mode,
            node0,
            node1,
            rank0_buf,
            rank1_buf,
            numel,
            stream0,
            stream1);
    }

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync ooverlap external p2p warmup");
}

double elapsed_ms_ooverlap_public(
    TestCollective collective,
    BenchMode mode,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    testing::reset_ready_signals(group);

    return testing::elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int) {
            launch_ooverlap_public_once(
                collective,
                mode,
                node0,
                node1,
                rank0_buf,
                rank1_buf,
                numel,
                stream0,
                stream1);
        });
}

void launch_nccl_once(
    TestCollective collective,
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms) {
    OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());

    testing::launch_nccl_collective_fp16(
        collective,
        comms[0],
        rank0_buf,
        numel,
        0,
        2,
        stream0);

    testing::launch_nccl_collective_fp16(
        collective,
        comms[1],
        rank1_buf,
        numel,
        1,
        2,
        stream1);

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
}

void run_nccl_iters(
    TestCollective collective,
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters) {
    if (iters <= 0) {
        return;
    }

    for (int i = 0; i < iters; ++i) {
        launch_nccl_once(
            collective,
            rank0_buf,
            rank1_buf,
            numel,
            stream0,
            stream1,
            comms);
    }
}

double elapsed_ms_nccl(
    TestCollective collective,
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters) {
    return testing::elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int) {
            launch_nccl_once(
                collective,
                rank0_buf,
                rank1_buf,
                numel,
                stream0,
                stream1,
                comms);
        });
}

void verify_collective_result(
    TestCollective collective,
    const char* label,
    half* rank0,
    half* rank1,
    int64_t numel,
    int dev0,
    int dev1) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    testing::verify_collective_fp16(
        collective,
        label,
        rank0,
        numel,
        0,
        2,
        dev0);

    testing::verify_collective_fp16(
        collective,
        label,
        rank1,
        numel,
        1,
        2,
        dev1);
#else
    (void)collective;
    (void)label;
    (void)rank0;
    (void)rank1;
    (void)numel;
    (void)dev0;
    (void)dev1;
#endif
}

void bench_ooverlap_external_variant(
    std::map<std::string, double>& results,
    const char* result_key,
    TestCollective collective,
    BenchMode mode,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters,
    int warmup) {
    testing::prepare_two_work_buffers(
        rank0_src,
        rank1_src,
        rank0_work,
        rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    run_ooverlap_public_iters(
        collective,
        mode,
        group,
        node0,
        node1,
        rank0_buf,
        rank1_buf,
        numel,
        dev0,
        dev1,
        stream0,
        stream1,
        warmup);

    testing::prepare_two_work_buffers(
        rank0_src,
        rank1_src,
        rank0_work,
        rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    const double total_ms =
        elapsed_ms_ooverlap_public(
            collective,
            mode,
            group,
            node0,
            node1,
            rank0_buf,
            rank1_buf,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            iters);

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync ooverlap external p2p measured");

    verify_collective_result(
        collective,
        result_key,
        rank0_work,
        rank1_work,
        static_cast<int64_t>(numel),
        dev0,
        dev1);

    results[result_key] =
        total_ms / static_cast<double>(iters);
}

void cleanup(
    int dev0,
    int dev1,
    half*& rank0_src,
    half*& rank1_src,
    half*& normal_rank0,
    half*& normal_rank1,
    half*& efficiency_rank0,
    half*& efficiency_rank1,
    half*& nccl_rank0,
    half*& nccl_rank1,
    oo_buffer_t*& normal_rank0_buf,
    oo_buffer_t*& normal_rank1_buf,
    oo_buffer_t*& efficiency_rank0_buf,
    oo_buffer_t*& efficiency_rank1_buf,
    oo_node_t*& node0,
    oo_node_t*& node1,
    oo_group_t*& group,
    cudaStream_t& stream0,
    cudaStream_t& stream1,
    ncclComm_t* comms) {
    testing::destroy_nccl_comms(comms, 2);

    /*
     * These are non-owning wrappers around external cudaMalloc buffers.
     * Destroy wrappers before freeing the external memory.
     */
    testing::destroy_oo_buffer(normal_rank0_buf);
    testing::destroy_oo_buffer(normal_rank1_buf);
    testing::destroy_oo_buffer(efficiency_rank0_buf);
    testing::destroy_oo_buffer(efficiency_rank1_buf);

    testing::cuda_free_on_device(dev0, rank0_src);
    testing::cuda_free_on_device(dev1, rank1_src);

    testing::cuda_free_on_device(dev0, normal_rank0);
    testing::cuda_free_on_device(dev1, normal_rank1);

    testing::cuda_free_on_device(dev0, efficiency_rank0);
    testing::cuda_free_on_device(dev1, efficiency_rank1);

    testing::cuda_free_on_device(dev0, nccl_rank0);
    testing::cuda_free_on_device(dev1, nccl_rank1);

    testing::destroy_oo_node(node0);
    testing::destroy_oo_node(node1);
    testing::destroy_oo_group(group);

    testing::destroy_stream_on_device(dev0, stream0);
    testing::destroy_stream_on_device(dev1, stream1);
}

} // namespace

bool external_p2p_two_gpu_collective_smoke_test(
    const std::string& collective_name_arg,
    int64_t numel,
    int dev0,
    int dev1) {
    std::map<std::string, double> result =
        benchmark_external_p2p_two_gpu_collective_sm90(
            collective_name_arg,
            numel,
            1,
            0,
            dev0,
            dev1);

    return !result.empty();
}

bool external_p2p_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    return external_p2p_two_gpu_collective_smoke_test(
        "allreduce",
        numel,
        dev0,
        dev1);
}

std::map<std::string, double> benchmark_external_p2p_two_gpu_collective_sm90(
    const std::string& collective_name_arg,
    int64_t numel_arg,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    if (numel_arg <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument(
            "benchmark_external_p2p_two_gpu_collective_sm90: invalid args");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument(
            "benchmark_external_p2p_two_gpu_collective_sm90: dev0 and dev1 must differ");
    }

    const TestCollective collective =
        testing::parse_collective(collective_name_arg);

    testing::validate_numel_for_collective(
        collective,
        numel_arg,
        2);

    const size_t numel =
        static_cast<size_t>(numel_arg);

    const size_t bytes =
        numel * sizeof(half);

    oo_group_t* group = nullptr;
    oo_node_t* node0 = nullptr;
    oo_node_t* node1 = nullptr;

    oo_buffer_t* normal_rank0_buf = nullptr;
    oo_buffer_t* normal_rank1_buf = nullptr;
    oo_buffer_t* efficiency_rank0_buf = nullptr;
    oo_buffer_t* efficiency_rank1_buf = nullptr;

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;

    /*
     * External/user-owned work buffers.
     * ooverlap only wraps these pointers; it does not allocate or free them.
     */
    half* normal_rank0 = nullptr;
    half* normal_rank1 = nullptr;
    half* efficiency_rank0 = nullptr;
    half* efficiency_rank1 = nullptr;

    /*
     * Standard NCCL comparison buffers.
     * These use plain cudaMalloc, not ncclMemAlloc/ncclCommWindowRegister.
     */
    half* nccl_rank0 = nullptr;
    half* nccl_rank1 = nullptr;

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    ncclComm_t comms[2] = {
        nullptr,
        nullptr,
    };

    try {
        int devices[2] = {dev0, dev1};

        testing::check_oo(
            oo_group_create_p2p(
                devices,
                2,
                &group),
            "oo_group_create_p2p");

        testing::check_oo(
            oo_node_create(
                group,
                0,
                &node0),
            "oo_node_create(rank0)");

        testing::check_oo(
            oo_node_create(
                group,
                1,
                &node1),
            "oo_node_create(rank1)");

        const int node0_dev =
            oo_node_device(node0);

        const int node1_dev =
            oo_node_device(node1);

        stream0 =
            system::runtime::create_stream_on_device(node0_dev);

        stream1 =
            system::runtime::create_stream_on_device(node1_dev);

        testing::cuda_malloc_half_on_device(
            node0_dev,
            &rank0_src,
            bytes,
            "cudaMalloc(rank0_src)");

        testing::cuda_malloc_half_on_device(
            node1_dev,
            &rank1_src,
            bytes,
            "cudaMalloc(rank1_src)");

        testing::cuda_malloc_half_on_device(
            node0_dev,
            &normal_rank0,
            bytes,
            "cudaMalloc(normal_rank0 external)");

        testing::cuda_malloc_half_on_device(
            node1_dev,
            &normal_rank1,
            bytes,
            "cudaMalloc(normal_rank1 external)");

        testing::cuda_malloc_half_on_device(
            node0_dev,
            &efficiency_rank0,
            bytes,
            "cudaMalloc(efficiency_rank0 external)");

        testing::cuda_malloc_half_on_device(
            node1_dev,
            &efficiency_rank1,
            bytes,
            "cudaMalloc(efficiency_rank1 external)");

        testing::cuda_malloc_half_on_device(
            node0_dev,
            &nccl_rank0,
            bytes,
            "cudaMalloc(nccl_rank0)");

        testing::cuda_malloc_half_on_device(
            node1_dev,
            &nccl_rank1,
            bytes,
            "cudaMalloc(nccl_rank1)");

        /*
         * Wrap externally allocated CUDA memory.
         * The wrappers are non-owning.
         */
        testing::check_oo(
            oo_buffer_wrap(
                node0,
                normal_rank0,
                bytes,
                &normal_rank0_buf),
            "oo_buffer_wrap(normal rank0)");

        testing::check_oo(
            oo_buffer_wrap(
                node1,
                normal_rank1,
                bytes,
                &normal_rank1_buf),
            "oo_buffer_wrap(normal rank1)");

        testing::check_oo(
            oo_buffer_wrap(
                node0,
                efficiency_rank0,
                bytes,
                &efficiency_rank0_buf),
            "oo_buffer_wrap(efficiency rank0)");

        testing::check_oo(
            oo_buffer_wrap(
                node1,
                efficiency_rank1,
                bytes,
                &efficiency_rank1_buf),
            "oo_buffer_wrap(efficiency rank1)");

        testing::fill_two_rank_sources_fp16(
            rank0_src,
            rank1_src,
            numel_arg,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommInitAll(
                comms,
                2,
                devices));

        std::map<std::string, double> results;

        bench_ooverlap_external_variant(
            results,
            "normal_ms",
            collective,
            BenchMode::BestPerformance,
            group,
            node0,
            node1,
            normal_rank0_buf,
            normal_rank1_buf,
            rank0_src,
            rank1_src,
            normal_rank0,
            normal_rank1,
            numel,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            iters,
            warmup);

        bench_ooverlap_external_variant(
            results,
            "efficiency_ms",
            collective,
            BenchMode::BestEfficiency,
            group,
            node0,
            node1,
            efficiency_rank0_buf,
            efficiency_rank1_buf,
            rank0_src,
            rank1_src,
            efficiency_rank0,
            efficiency_rank1,
            numel,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            iters,
            warmup);

        testing::prepare_two_work_buffers(
            rank0_src,
            rank1_src,
            nccl_rank0,
            nccl_rank1,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        run_nccl_iters(
            collective,
            nccl_rank0,
            nccl_rank1,
            numel,
            stream0,
            stream1,
            comms,
            warmup);

        testing::sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync nccl external p2p warmup");

        testing::prepare_two_work_buffers(
            rank0_src,
            rank1_src,
            nccl_rank0,
            nccl_rank1,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        const double nccl_total_ms =
            elapsed_ms_nccl(
                collective,
                nccl_rank0,
                nccl_rank1,
                numel,
                node0_dev,
                node1_dev,
                stream0,
                stream1,
                comms,
                iters);

        testing::sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync nccl external p2p measured");

        verify_collective_result(
            collective,
            "nccl",
            nccl_rank0,
            nccl_rank1,
            numel_arg,
            node0_dev,
            node1_dev);

        results["nccl_ms"] =
            nccl_total_ms / static_cast<double>(iters);

        results["collective"] =
            testing::collective_code(collective);

        results["numel"] =
            static_cast<double>(numel);

        results["bytes"] =
            static_cast<double>(bytes);

        results["iters"] =
            static_cast<double>(iters);

        results["warmup"] =
            static_cast<double>(warmup);

        cleanup(
            node0_dev,
            node1_dev,
            rank0_src,
            rank1_src,
            normal_rank0,
            normal_rank1,
            efficiency_rank0,
            efficiency_rank1,
            nccl_rank0,
            nccl_rank1,
            normal_rank0_buf,
            normal_rank1_buf,
            efficiency_rank0_buf,
            efficiency_rank1_buf,
            node0,
            node1,
            group,
            stream0,
            stream1,
            comms);

        return results;
    } catch (...) {
        const int node0_dev =
            node0 != nullptr ? oo_node_device(node0) : dev0;

        const int node1_dev =
            node1 != nullptr ? oo_node_device(node1) : dev1;

        cleanup(
            node0_dev,
            node1_dev,
            rank0_src,
            rank1_src,
            normal_rank0,
            normal_rank1,
            efficiency_rank0,
            efficiency_rank1,
            nccl_rank0,
            nccl_rank1,
            normal_rank0_buf,
            normal_rank1_buf,
            efficiency_rank0_buf,
            efficiency_rank1_buf,
            node0,
            node1,
            group,
            stream0,
            stream1,
            comms);

        throw;
    }
}

std::map<std::string, double> benchmark_external_p2p_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    return benchmark_external_p2p_two_gpu_collective_sm90(
        "allreduce",
        numel,
        iters,
        warmup,
        dev0,
        dev1);
}

} // namespace ooverlap
