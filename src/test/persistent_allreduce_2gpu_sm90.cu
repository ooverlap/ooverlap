#include "test/persistent_allreduce_2gpu_sm90.h"

#include "comm/launch_config.h"
#include "comm/ooverlap_comm_internal.h"
#include "comm/tma_multi_gpu_all_gather_sm90.h"
#include "comm/tma_multi_gpu_allreduce_sm90.h"
#include "comm/tma_multi_gpu_reduce_scatter_sm90.h"

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
#define OOVERLAP_BENCH_VERIFY_RESULTS 0
#endif

namespace ooverlap {
namespace {

using testing::TestCollective;

enum class BenchPlan {
    Tma = 0,
    SeqFast = 1,
    OverlapFast = 2,
};

bool plan_supported(
    TestCollective collective,
    BenchPlan plan) {
    if (plan != BenchPlan::OverlapFast) {
        return true;
    }

    if (collective != TestCollective::AllReduce) {
        return false;
    }

    return TMA_TWO_GPU_PEER_DEFAULT_MAX_CTAS >= 2;
}

comm::LaunchConfig launch_config_for(
    TestCollective collective,
    BenchPlan plan) {
    if (collective == TestCollective::AllReduce) {
        if (plan == BenchPlan::Tma) {
            return comm::make_allreduce_launch_config(
                comm::AllReducePlanKind::TmaCopy);
        }

        if (plan == BenchPlan::SeqFast) {
            return comm::make_allreduce_launch_config(
                comm::AllReducePlanKind::SeqFastCopyGmem);
        }

        return comm::make_allreduce_launch_config(
            comm::AllReducePlanKind::OverlapFastCopyGmem);
    }

    if (collective == TestCollective::ReduceScatter) {
        if (plan == BenchPlan::Tma) {
            return comm::make_reduce_scatter_launch_config(
                comm::ReduceScatterPlanKind::TmaReduce);
        }

        if (plan == BenchPlan::SeqFast) {
            return comm::make_reduce_scatter_launch_config(
                comm::ReduceScatterPlanKind::SeqFastAddGmem);
        }
    }

    if (collective == TestCollective::AllGather) {
        if (plan == BenchPlan::Tma) {
            return comm::make_all_gather_launch_config(
                comm::AllGatherPlanKind::TmaCopy);
        }

        if (plan == BenchPlan::SeqFast) {
            return comm::make_all_gather_launch_config(
                comm::AllGatherPlanKind::SeqFastCopyGmem);
        }
    }

    throw std::invalid_argument("unsupported collective/plan combination");
}

void launch_ooverlap_once(
    TestCollective collective,
    BenchPlan plan,
    const void* rank0_in,
    const void* rank1_in,
    void* rank0_buf,
    void* rank1_buf,
    void* rank0_peer,
    void* rank1_peer,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int* rank0_ready,
    int* rank1_ready,
    int collective_epoch) {
    const comm::LaunchConfig config =
        launch_config_for(collective, plan);

    void* rank0_peers[] = {rank0_peer};
    void* rank1_peers[] = {rank1_peer};

    const int* rank0_peer_ready[] = {rank1_ready};
    const int* rank1_peer_ready[] = {rank0_ready};

    if (collective == TestCollective::AllReduce) {
        testing::check_cuda(
            enqueue_tma_multi_gpu_allreduce_rank_sm90(
                rank0_in,
                rank0_buf,
                rank0_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                0,
                2,
                dev0,
                stream0,
                rank0_ready,
                rank0_peer_ready,
                collective_epoch,
                config),
            "enqueue allreduce rank0");

        testing::check_cuda(
            enqueue_tma_multi_gpu_allreduce_rank_sm90(
                rank1_in,
                rank1_buf,
                rank1_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                1,
                2,
                dev1,
                stream1,
                rank1_ready,
                rank1_peer_ready,
                collective_epoch,
                config),
            "enqueue allreduce rank1");

        return;
    }

    if (collective == TestCollective::ReduceScatter) {
        testing::check_cuda(
            enqueue_tma_multi_gpu_reduce_scatter_rank_sm90(
                rank0_in,
                rank0_buf,
                rank0_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                0,
                2,
                dev0,
                stream0,
                rank0_ready,
                rank0_peer_ready,
                collective_epoch,
                config),
            "enqueue reduce_scatter rank0");

        testing::check_cuda(
            enqueue_tma_multi_gpu_reduce_scatter_rank_sm90(
                rank1_in,
                rank1_buf,
                rank1_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                1,
                2,
                dev1,
                stream1,
                rank1_ready,
                rank1_peer_ready,
                collective_epoch,
                config),
            "enqueue reduce_scatter rank1");

        return;
    }

    if (collective == TestCollective::AllGather) {
        testing::check_cuda(
            enqueue_tma_multi_gpu_all_gather_rank_sm90(
                rank0_in,
                rank0_buf,
                rank0_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                0,
                2,
                dev0,
                stream0,
                rank0_ready,
                rank0_peer_ready,
                collective_epoch,
                config),
            "enqueue all_gather rank0");

        testing::check_cuda(
            enqueue_tma_multi_gpu_all_gather_rank_sm90(
                rank1_in,
                rank1_buf,
                rank1_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                1,
                2,
                dev1,
                stream1,
                rank1_ready,
                rank1_peer_ready,
                collective_epoch,
                config),
            "enqueue all_gather rank1");

        return;
    }

    throw std::invalid_argument("unsupported ooverlap collective");
}

void run_ooverlap_iters(
    TestCollective collective,
    BenchPlan plan,
    oo_group_t* group,
    half* rank0_work,
    half* rank1_work,
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

    int* rank0_ready =
        testing::ready_signal_ptr(group, 0);
    int* rank1_ready =
        testing::ready_signal_ptr(group, 1);

    for (int i = 0; i < iters; ++i) {
        launch_ooverlap_once(
            collective,
            plan,
            rank0_work,
            rank1_work,
            rank0_work,
            rank1_work,
            rank1_work,
            rank0_work,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            rank0_ready,
            rank1_ready,
            i + 1);
    }

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync ooverlap warmup");
}

double elapsed_ms_ooverlap(
    TestCollective collective,
    BenchPlan plan,
    oo_group_t* group,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    testing::reset_ready_signals(group);

    int epoch = 1;
    int* rank0_ready =
        testing::ready_signal_ptr(group, 0);
    int* rank1_ready =
        testing::ready_signal_ptr(group, 1);

    return testing::elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int) {
            launch_ooverlap_once(
                collective,
                plan,
                rank0_work,
                rank1_work,
                rank0_work,
                rank1_work,
                rank1_work,
                rank0_work,
                numel,
                dev0,
                dev1,
                stream0,
                stream1,
                rank0_ready,
                rank1_ready,
                epoch++);
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

void bench_ooverlap_variant(
    std::map<std::string, double>& results,
    const char* result_key,
    TestCollective collective,
    BenchPlan plan,
    oo_group_t* group,
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
    if (!plan_supported(collective, plan)) {
        return;
    }

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

    run_ooverlap_iters(
        collective,
        plan,
        group,
        rank0_work,
        rank1_work,
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
        elapsed_ms_ooverlap(
            collective,
            plan,
            group,
            rank0_work,
            rank1_work,
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
        "sync ooverlap measured variant");

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


void nccl_mem_alloc_half_on_device(
    int device,
    half** ptr,
    size_t bytes,
    const char* label) {
    if (ptr == nullptr) {
        throw std::invalid_argument(
            "nccl_mem_alloc_half_on_device: ptr must not be null");
    }

    *ptr = nullptr;

    system::runtime::set_device(device);

    void* raw = nullptr;

    OOVERLAP_TEST_NCCL_CHECK(
        ncclMemAlloc(&raw, bytes));

    if (raw == nullptr) {
        throw std::runtime_error(
            std::string(label) + ": ncclMemAlloc returned nullptr");
    }

    *ptr =
        reinterpret_cast<half*>(raw);
}

void nccl_mem_free_on_device(
    int device,
    half*& ptr) {
    if (ptr == nullptr) {
        return;
    }

    // Cleanup path should not throw. Best effort free.
    (void)cudaSetDevice(device);
    (void)ncclMemFree(static_cast<void*>(ptr));

    ptr = nullptr;
}

void register_nccl_symmetric_windows(
    ncclComm_t* comms,
    half* nccl_rank0_buf,
    half* nccl_rank1_buf,
    size_t bytes,
    ncclWindow_t& nccl_rank0_win,
    ncclWindow_t& nccl_rank1_win) {
    nccl_rank0_win = nullptr;
    nccl_rank1_win = nullptr;

#ifndef NCCL_WIN_COLL_SYMMETRIC
    throw std::runtime_error(
        "NCCL_WIN_COLL_SYMMETRIC is not available in this NCCL header. "
        "Make sure the bundled NCCL 2.27+ headers are used at compile time.");
#else
    OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());

    OOVERLAP_TEST_NCCL_CHECK(
        ncclCommWindowRegister(
            comms[0],
            nccl_rank0_buf,
            bytes,
            &nccl_rank0_win,
            NCCL_WIN_COLL_SYMMETRIC));

    OOVERLAP_TEST_NCCL_CHECK(
        ncclCommWindowRegister(
            comms[1],
            nccl_rank1_buf,
            bytes,
            &nccl_rank1_win,
            NCCL_WIN_COLL_SYMMETRIC));

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
#endif
}

void deregister_nccl_window_best_effort(
    ncclComm_t comm,
    ncclWindow_t& win) {
    if (comm == nullptr || win == nullptr) {
        return;
    }

    (void)ncclCommWindowDeregister(comm, win);
    win = nullptr;
}

void cleanup(
    int dev0,
    int dev1,
    half*& rank0_src,
    half*& rank1_src,
    half*& nccl_rank0_buf,
    half*& nccl_rank1_buf,
    ncclWindow_t& nccl_rank0_win,
    ncclWindow_t& nccl_rank1_win,
    oo_buffer_t*& normal_rank0_buf,
    oo_buffer_t*& normal_rank1_buf,
    oo_buffer_t*& seq_rank0_buf,
    oo_buffer_t*& seq_rank1_buf,
    oo_buffer_t*& overlap_rank0_buf,
    oo_buffer_t*& overlap_rank1_buf,
    oo_node_t*& node0,
    oo_node_t*& node1,
    oo_group_t*& group,
    cudaStream_t& stream0,
    cudaStream_t& stream1,
    ncclComm_t* comms) {
    deregister_nccl_window_best_effort(comms[0], nccl_rank0_win);
    deregister_nccl_window_best_effort(comms[1], nccl_rank1_win);

    testing::destroy_nccl_comms(comms, 2);

    testing::cuda_free_on_device(dev0, rank0_src);
    testing::cuda_free_on_device(dev1, rank1_src);
    nccl_mem_free_on_device(dev0, nccl_rank0_buf);
    nccl_mem_free_on_device(dev1, nccl_rank1_buf);

    testing::destroy_oo_buffer(normal_rank0_buf);
    testing::destroy_oo_buffer(normal_rank1_buf);
    testing::destroy_oo_buffer(seq_rank0_buf);
    testing::destroy_oo_buffer(seq_rank1_buf);
    testing::destroy_oo_buffer(overlap_rank0_buf);
    testing::destroy_oo_buffer(overlap_rank1_buf);

    testing::destroy_oo_node(node0);
    testing::destroy_oo_node(node1);
    testing::destroy_oo_group(group);

    testing::destroy_stream_on_device(dev0, stream0);
    testing::destroy_stream_on_device(dev1, stream1);
}

} // namespace

bool tma_persistent_two_gpu_collective_smoke_test(
    const std::string& collective_name_arg,
    int64_t numel,
    int dev0,
    int dev1) {
    std::map<std::string, double> result =
        benchmark_persistent_two_gpu_collective_sm90(
            collective_name_arg,
            numel,
            1,
            0,
            dev0,
            dev1);

    return !result.empty();
}

bool tma_persistent_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    return tma_persistent_two_gpu_collective_smoke_test(
        "allreduce",
        numel,
        dev0,
        dev1);
}

std::map<std::string, double> benchmark_persistent_two_gpu_collective_sm90(
    const std::string& collective_name_arg,
    int64_t numel_arg,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    if (numel_arg <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_collective_sm90: invalid args");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_collective_sm90: dev0 and dev1 must differ");
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
    oo_buffer_t* seq_rank0_buf = nullptr;
    oo_buffer_t* seq_rank1_buf = nullptr;
    oo_buffer_t* overlap_rank0_buf = nullptr;
    oo_buffer_t* overlap_rank1_buf = nullptr;

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;
    half* nccl_rank0_buf = nullptr;
    half* nccl_rank1_buf = nullptr;

    ncclWindow_t nccl_rank0_win = nullptr;
    ncclWindow_t nccl_rank1_win = nullptr;

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    ncclComm_t comms[2] = {
        nullptr,
        nullptr,
    };

    try {
        int devices[2] = {dev0, dev1};

        testing::check_oo(
            oo_group_create(devices, 2, &group),
            "oo_group_create");

        testing::check_oo(
            oo_node_create(group, 0, &node0),
            "oo_node_create(rank0)");

        testing::check_oo(
            oo_node_create(group, 1, &node1),
            "oo_node_create(rank1)");

        const int node0_dev = oo_node_device(node0);
        const int node1_dev = oo_node_device(node1);

        stream0 =
            system::runtime::create_stream_on_device(node0_dev);
        stream1 =
            system::runtime::create_stream_on_device(node1_dev);

        testing::check_oo(
            oo_buffer_alloc(node0, bytes, &normal_rank0_buf),
            "oo_buffer_alloc(normal rank0)");
        testing::check_oo(
            oo_buffer_alloc(node1, bytes, &normal_rank1_buf),
            "oo_buffer_alloc(normal rank1)");

        testing::check_oo(
            oo_buffer_alloc(node0, bytes, &seq_rank0_buf),
            "oo_buffer_alloc(seq rank0)");
        testing::check_oo(
            oo_buffer_alloc(node1, bytes, &seq_rank1_buf),
            "oo_buffer_alloc(seq rank1)");

        testing::check_oo(
            oo_buffer_alloc(node0, bytes, &overlap_rank0_buf),
            "oo_buffer_alloc(overlap rank0)");
        testing::check_oo(
            oo_buffer_alloc(node1, bytes, &overlap_rank1_buf),
            "oo_buffer_alloc(overlap rank1)");

        half* normal_rank0 =
            reinterpret_cast<half*>(oo_buffer_ptr(normal_rank0_buf));
        half* normal_rank1 =
            reinterpret_cast<half*>(oo_buffer_ptr(normal_rank1_buf));

        half* seq_rank0 =
            reinterpret_cast<half*>(oo_buffer_ptr(seq_rank0_buf));
        half* seq_rank1 =
            reinterpret_cast<half*>(oo_buffer_ptr(seq_rank1_buf));

        half* overlap_rank0 =
            reinterpret_cast<half*>(oo_buffer_ptr(overlap_rank0_buf));
        half* overlap_rank1 =
            reinterpret_cast<half*>(oo_buffer_ptr(overlap_rank1_buf));

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

        nccl_mem_alloc_half_on_device(
            node0_dev,
            &nccl_rank0_buf,
            bytes,
            "ncclMemAlloc(nccl_rank0_buf)");

        nccl_mem_alloc_half_on_device(
            node1_dev,
            &nccl_rank1_buf,
            bytes,
            "ncclMemAlloc(nccl_rank1_buf)");

        testing::fill_two_rank_sources_fp16(
            rank0_src,
            rank1_src,
            numel_arg,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommInitAll(comms, 2, devices));

        register_nccl_symmetric_windows(
            comms,
            nccl_rank0_buf,
            nccl_rank1_buf,
            bytes,
            nccl_rank0_win,
            nccl_rank1_win);

        std::map<std::string, double> results;

        bench_ooverlap_variant(
            results,
            "normal_ms",
            collective,
            BenchPlan::Tma,
            group,
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

        bench_ooverlap_variant(
            results,
            "not_fused_ms",
            collective,
            BenchPlan::SeqFast,
            group,
            rank0_src,
            rank1_src,
            seq_rank0,
            seq_rank1,
            numel,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            iters,
            warmup);

        bench_ooverlap_variant(
            results,
            "fused_ms",
            collective,
            BenchPlan::OverlapFast,
            group,
            rank0_src,
            rank1_src,
            overlap_rank0,
            overlap_rank1,
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
            nccl_rank0_buf,
            nccl_rank1_buf,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        run_nccl_iters(
            collective,
            nccl_rank0_buf,
            nccl_rank1_buf,
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
            "sync nccl warmup");

        testing::prepare_two_work_buffers(
            rank0_src,
            rank1_src,
            nccl_rank0_buf,
            nccl_rank1_buf,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        const double nccl_total_ms =
            elapsed_ms_nccl(
                collective,
                nccl_rank0_buf,
                nccl_rank1_buf,
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
            "sync nccl measured");

        verify_collective_result(
            collective,
            "nccl",
            nccl_rank0_buf,
            nccl_rank1_buf,
            numel_arg,
            node0_dev,
            node1_dev);

        results["nccl_ms"] =
            nccl_total_ms / static_cast<double>(iters);

        results["collective"] =
            testing::collective_code(collective);
        results["numel"] =
            static_cast<double>(numel);
        results["iters"] =
            static_cast<double>(iters);
        results["warmup"] =
            static_cast<double>(warmup);

        cleanup(
            node0_dev,
            node1_dev,
            rank0_src,
            rank1_src,
            nccl_rank0_buf,
            nccl_rank1_buf,
            nccl_rank0_win,
            nccl_rank1_win,
            normal_rank0_buf,
            normal_rank1_buf,
            seq_rank0_buf,
            seq_rank1_buf,
            overlap_rank0_buf,
            overlap_rank1_buf,
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
            nccl_rank0_buf,
            nccl_rank1_buf,
            nccl_rank0_win,
            nccl_rank1_win,
            normal_rank0_buf,
            normal_rank1_buf,
            seq_rank0_buf,
            seq_rank1_buf,
            overlap_rank0_buf,
            overlap_rank1_buf,
            node0,
            node1,
            group,
            stream0,
            stream1,
            comms);

        throw;
    }
}

std::map<std::string, double> benchmark_persistent_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    return benchmark_persistent_two_gpu_collective_sm90(
        "allreduce",
        numel,
        iters,
        warmup,
        dev0,
        dev1);
}

} // namespace ooverlap
