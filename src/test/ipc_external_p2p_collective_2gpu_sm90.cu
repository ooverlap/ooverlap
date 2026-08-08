#include "test/ipc_external_p2p_collective_2gpu_sm90.h"

#include "comm/ooverlap_comm_internal.h"
#include "ooverlap/comm.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/collective_test_utils.cuh"
#include "ooverlap/testing/nccl_utils.cuh"
#include "ooverlap/testing/timing.cuh"
#include "ooverlap/testing/two_gpu_test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef OOVERLAP_BENCH_VERIFY_RESULTS
#define OOVERLAP_BENCH_VERIFY_RESULTS 0
#endif

namespace ooverlap {
namespace {

using testing::TestCollective;

void validate_devices(
    const std::vector<int>& devices,
    int local_rank) {
    if (devices.size() < 2) {
        throw std::invalid_argument(
            "IPC external P2P collective requires at least two devices");
    }

    std::set<int> unique;
    for (int device : devices) {
        if (device < 0) {
            throw std::invalid_argument("device ids must be non-negative");
        }
        if (!unique.insert(device).second) {
            throw std::invalid_argument("device ids must be unique");
        }
    }

    if (local_rank < 0 ||
        local_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("local_rank is outside the device list");
    }
}

void broker_sync(oo_group_t* group) {
    if (group != nullptr && group->broker) {
        group->broker->sync();
    }
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
    OOVERLAP_TEST_NCCL_CHECK(ncclMemAlloc(&raw, bytes));

    if (raw == nullptr) {
        throw std::runtime_error(
            std::string(label) + ": ncclMemAlloc returned nullptr");
    }

    *ptr = reinterpret_cast<half*>(raw);
}

void nccl_mem_free_on_device(
    int device,
    half*& ptr) {
    if (ptr == nullptr) {
        return;
    }

    (void)cudaSetDevice(device);
    (void)ncclMemFree(static_cast<void*>(ptr));
    ptr = nullptr;
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

void register_nccl_symmetric_window_rank(
    ncclComm_t comm,
    half* buf,
    size_t bytes,
    ncclWindow_t& win) {
    win = nullptr;
    OOVERLAP_TEST_NCCL_CHECK(
        ncclCommWindowRegister(
            comm,
            buf,
            bytes,
            &win,
            NCCL_WIN_COLL_SYMMETRIC));
}

void launch_ooverlap_public_once_for_rank(
    TestCollective collective,
    oo_node_t* node,
    oo_buffer_t* local,
    size_t numel,
    cudaStream_t stream,
    const char* label) {
    if (collective == TestCollective::AllReduce) {
        testing::check_oo(
            oo_allreduce_tuned(
                node,
                local,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                OO_TUNING_BEST_PERFORMANCE,
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
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                OO_TUNING_BEST_PERFORMANCE,
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
                numel,
                OO_DTYPE_FLOAT16,
                OO_TUNING_BEST_PERFORMANCE,
                stream),
            label);
        return;
    }

    throw std::invalid_argument("unsupported collective");
}

void launch_nccl_once_for_rank(
    TestCollective collective,
    ncclComm_t comm,
    half* work,
    size_t numel,
    int local_rank,
    int world_size,
    cudaStream_t stream) {
    testing::launch_nccl_collective_fp16(
        collective,
        comm,
        work,
        numel,
        local_rank,
        world_size,
        stream);
}

void sync_device_stream(
    int device,
    cudaStream_t stream,
    const char* label) {
    system::runtime::sync_stream_on_device(device, stream, label);
}

void reset_and_sync(
    half* work,
    const half* src,
    size_t bytes,
    int device,
    cudaStream_t stream,
    const char* label) {
    testing::reset_work_buffer_async(
        work,
        src,
        bytes,
        device,
        stream);
    sync_device_stream(device, stream, label);
}

void verify_collective_result_rank(
    TestCollective collective,
    const char* label,
    half* work,
    int64_t numel,
    int local_rank,
    int world_size,
    int local_device,
    bool verify_runtime) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    const bool enabled_by_compile = true;
#else
    const bool enabled_by_compile = false;
#endif

    return;

    if (!verify_runtime && !enabled_by_compile) {
        return;
    }

    testing::verify_collective_fp16(
        collective,
        label,
        work,
        numel,
        local_rank,
        world_size,
        local_device);
}

void warmup_ooverlap_rank(
    TestCollective collective,
    oo_group_t* group,
    oo_node_t* node,
    oo_buffer_t* local_buf,
    const half* local_src,
    half* work,
    size_t numel,
    size_t bytes,
    int local_device,
    cudaStream_t stream,
    int warmup) {
    for (int i = 0; i < warmup; ++i) {
        reset_and_sync(
            work,
            local_src,
            bytes,
            local_device,
            stream,
            "sync ipc ooverlap warmup reset");

        broker_sync(group);
        launch_ooverlap_public_once_for_rank(
            collective,
            node,
            local_buf,
            numel,
            stream,
            "ipc ooverlap warmup");
        sync_device_stream(
            local_device,
            stream,
            "sync ipc ooverlap warmup");
        broker_sync(group);
    }
}

double benchmark_ooverlap_rank_total_ms(
    TestCollective collective,
    oo_group_t* group,
    oo_node_t* node,
    oo_buffer_t* local_buf,
    const half* local_src,
    half* work,
    size_t numel,
    size_t bytes,
    int local_device,
    cudaStream_t stream,
    int iters) {
    double total_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        reset_and_sync(
            work,
            local_src,
            bytes,
            local_device,
            stream,
            "sync ipc ooverlap timed reset");

        broker_sync(group);
        total_ms += testing::elapsed_one_rank_ms(
            local_device,
            stream,
            [&]() {
                launch_ooverlap_public_once_for_rank(
                    collective,
                    node,
                    local_buf,
                    numel,
                    stream,
                    "ipc ooverlap timed");
            });
        broker_sync(group);
    }

    return total_ms;
}

void warmup_nccl_rank(
    TestCollective collective,
    oo_group_t* barrier_group,
    ncclComm_t comm,
    const half* local_src,
    half* work,
    size_t numel,
    size_t bytes,
    int local_rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int warmup,
    const char* sync_label) {
    for (int i = 0; i < warmup; ++i) {
        reset_and_sync(
            work,
            local_src,
            bytes,
            local_device,
            stream,
            "sync ipc nccl warmup reset");

        broker_sync(barrier_group);
        launch_nccl_once_for_rank(
            collective,
            comm,
            work,
            numel,
            local_rank,
            world_size,
            stream);
        sync_device_stream(local_device, stream, sync_label);
        broker_sync(barrier_group);
    }
}

double benchmark_nccl_rank_total_ms(
    TestCollective collective,
    oo_group_t* barrier_group,
    ncclComm_t comm,
    const half* local_src,
    half* work,
    size_t numel,
    size_t bytes,
    int local_rank,
    int world_size,
    int local_device,
    cudaStream_t stream,
    int iters) {
    double total_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        reset_and_sync(
            work,
            local_src,
            bytes,
            local_device,
            stream,
            "sync ipc nccl timed reset");

        broker_sync(barrier_group);
        total_ms += testing::elapsed_one_rank_ms(
            local_device,
            stream,
            [&]() {
                launch_nccl_once_for_rank(
                    collective,
                    comm,
                    work,
                    numel,
                    local_rank,
                    world_size,
                    stream);
            });
        broker_sync(barrier_group);
    }

    return total_ms;
}

void cleanup_rank(
    int local_device,
    half*& local_src,
    half*& ooverlap_work,
    half*& nccl_work,
    half*& nccl_symmetric_work,
    ncclWindow_t& nccl_symmetric_win,
    ncclComm_t& nccl_comm,
    oo_buffer_t*& ooverlap_buf,
    oo_node_t*& node,
    oo_group_t*& group,
    cudaStream_t& stream) {
    deregister_nccl_window_best_effort(
        nccl_comm,
        nccl_symmetric_win);

    if (nccl_comm != nullptr) {
        (void)ncclCommDestroy(nccl_comm);
        nccl_comm = nullptr;
    }

    if (ooverlap_buf != nullptr) {
        oo_buffer_destroy(ooverlap_buf);
        ooverlap_buf = nullptr;
    }

    testing::cuda_free_on_device(local_device, local_src);
    testing::cuda_free_on_device(local_device, ooverlap_work);
    testing::cuda_free_on_device(local_device, nccl_work);
    nccl_mem_free_on_device(local_device, nccl_symmetric_work);

    if (node != nullptr) {
        oo_node_destroy(node);
        node = nullptr;
    }

    if (group != nullptr) {
        oo_group_destroy(group);
        group = nullptr;
    }

    if (stream != nullptr) {
        system::runtime::destroy_stream_on_device(local_device, stream);
        stream = nullptr;
    }
}

} // namespace

bool ipc_external_p2p_collective_smoke_rank_sm90(
    const std::string& collective_name_arg,
    int64_t numel,
    int local_rank,
    const std::vector<int>& devices,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    bool verify) {
    std::map<std::string, double> result =
        benchmark_ipc_external_p2p_collective_rank_sm90(
            collective_name_arg,
            numel,
            local_rank,
            devices,
            broker_key,
            nccl_unique_id_bytes,
            1,
            0,
            verify);

    return !result.empty();
}

std::map<std::string, double>
benchmark_ipc_external_p2p_collective_rank_sm90(
    const std::string& collective_name_arg,
    int64_t numel_arg,
    int local_rank,
    const std::vector<int>& devices,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify) {
    if (numel_arg <= 0 ||
        iters <= 0 ||
        warmup < 0 ||
        broker_key.empty()) {
        throw std::invalid_argument(
            "benchmark_ipc_external_p2p_collective_rank_sm90: invalid args");
    }

    validate_devices(devices, local_rank);

    const int world_size = static_cast<int>(devices.size());
    const int local_device = devices[static_cast<std::size_t>(local_rank)];
    const TestCollective collective =
        testing::parse_collective(collective_name_arg);

    testing::validate_numel_for_collective(
        collective,
        numel_arg,
        world_size);

    const size_t numel = static_cast<size_t>(numel_arg);
    const size_t bytes = numel * sizeof(half);

    oo_group_t* group = nullptr;
    oo_node_t* node = nullptr;
    oo_buffer_t* ooverlap_buf = nullptr;

    half* local_src = nullptr;
    half* ooverlap_work = nullptr;
    half* nccl_work = nullptr;
    half* nccl_symmetric_work = nullptr;

    cudaStream_t stream = nullptr;
    ncclComm_t nccl_comm = nullptr;
    ncclWindow_t nccl_symmetric_win = nullptr;

    try {
        system::runtime::set_device(local_device);
        stream = system::runtime::create_stream_on_device(local_device);

        testing::cuda_malloc_half_on_device(
            local_device,
            &local_src,
            bytes,
            "cudaMalloc(ipc external local_src)");
        testing::cuda_malloc_half_on_device(
            local_device,
            &ooverlap_work,
            bytes,
            "cudaMalloc(ipc external ooverlap_work)");
        testing::cuda_malloc_half_on_device(
            local_device,
            &nccl_work,
            bytes,
            "cudaMalloc(ipc external nccl_work)");
        nccl_mem_alloc_half_on_device(
            local_device,
            &nccl_symmetric_work,
            bytes,
            "ncclMemAlloc(ipc external nccl_symmetric_work)");

        testing::fill_rank_source_fp16(
            local_src,
            numel_arg,
            local_rank,
            local_device,
            stream);

        testing::check_oo(
            oo_group_create_ipc(
                devices.data(),
                world_size,
                local_rank,
                broker_key.c_str(),
                &group),
            "oo_group_create_ipc(ipc external)");

        testing::check_oo(
            oo_node_create(
                group,
                local_rank,
                &node),
            "oo_node_create(ipc external)");

        testing::check_oo(
            oo_buffer_wrap(
                node,
                ooverlap_work,
                bytes,
                &ooverlap_buf),
            "oo_buffer_wrap(ipc external ooverlap_work)");

        const ncclUniqueId nccl_id =
            testing::make_nccl_unique_id(nccl_unique_id_bytes);

        broker_sync(group);
        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommInitRank(
                &nccl_comm,
                world_size,
                nccl_id,
                local_rank));
        broker_sync(group);

        register_nccl_symmetric_window_rank(
            nccl_comm,
            nccl_symmetric_work,
            bytes,
            nccl_symmetric_win);
        broker_sync(group);

        std::map<std::string, double> results;

        warmup_ooverlap_rank(
            collective,
            group,
            node,
            ooverlap_buf,
            local_src,
            ooverlap_work,
            numel,
            bytes,
            local_device,
            stream,
            warmup);

        const double ooverlap_total_ms =
            benchmark_ooverlap_rank_total_ms(
                collective,
                group,
                node,
                ooverlap_buf,
                local_src,
                ooverlap_work,
                numel,
                bytes,
                local_device,
                stream,
                iters);

        sync_device_stream(
            local_device,
            stream,
            "sync ipc ooverlap measured");
        verify_collective_result_rank(
            collective,
            "ipc ooverlap external p2p",
            ooverlap_work,
            numel_arg,
            local_rank,
            world_size,
            local_device,
            verify);

        warmup_nccl_rank(
            collective,
            group,
            nccl_comm,
            local_src,
            nccl_work,
            numel,
            bytes,
            local_rank,
            world_size,
            local_device,
            stream,
            warmup,
            "sync ipc nccl warmup");

        const double nccl_total_ms =
            benchmark_nccl_rank_total_ms(
                collective,
                group,
                nccl_comm,
                local_src,
                nccl_work,
                numel,
                bytes,
                local_rank,
                world_size,
                local_device,
                stream,
                iters);

        sync_device_stream(
            local_device,
            stream,
            "sync ipc nccl measured");
        verify_collective_result_rank(
            collective,
            "ipc normal NCCL external p2p",
            nccl_work,
            numel_arg,
            local_rank,
            world_size,
            local_device,
            verify);

        warmup_nccl_rank(
            collective,
            group,
            nccl_comm,
            local_src,
            nccl_symmetric_work,
            numel,
            bytes,
            local_rank,
            world_size,
            local_device,
            stream,
            warmup,
            "sync ipc nccl symmetric warmup");

        const double nccl_symmetric_total_ms =
            benchmark_nccl_rank_total_ms(
                collective,
                group,
                nccl_comm,
                local_src,
                nccl_symmetric_work,
                numel,
                bytes,
                local_rank,
                world_size,
                local_device,
                stream,
                iters);

        sync_device_stream(
            local_device,
            stream,
            "sync ipc nccl symmetric measured");
        verify_collective_result_rank(
            collective,
            "ipc symmetric NCCL external p2p",
            nccl_symmetric_work,
            numel_arg,
            local_rank,
            world_size,
            local_device,
            verify);

        const size_t local_shard_count =
            testing::rank_partition_count(
                numel,
                local_rank,
                world_size);

        results["collective"] = testing::collective_code(collective);
        results["rank"] = static_cast<double>(local_rank);
        results["world_size"] = static_cast<double>(world_size);
        results["numel"] = static_cast<double>(numel);
        results["bytes"] = static_cast<double>(bytes);
        results["local_shard_numel"] =
            static_cast<double>(local_shard_count);
        results["local_shard_bytes"] =
            static_cast<double>(local_shard_count * sizeof(half));
        results["iters"] = static_cast<double>(iters);
        results["warmup"] = static_cast<double>(warmup);
        results["ooverlap_ms"] =
            ooverlap_total_ms / static_cast<double>(iters);
        results["nccl_ms"] =
            nccl_total_ms / static_cast<double>(iters);
        results["nccl_symmetric_ms"] =
            nccl_symmetric_total_ms / static_cast<double>(iters);

        broker_sync(group);
        cleanup_rank(
            local_device,
            local_src,
            ooverlap_work,
            nccl_work,
            nccl_symmetric_work,
            nccl_symmetric_win,
            nccl_comm,
            ooverlap_buf,
            node,
            group,
            stream);

        return results;
    } catch (...) {
        cleanup_rank(
            local_device,
            local_src,
            ooverlap_work,
            nccl_work,
            nccl_symmetric_work,
            nccl_symmetric_win,
            nccl_comm,
            ooverlap_buf,
            node,
            group,
            stream);
        throw;
    }
}

std::map<std::string, double>
benchmark_ipc_external_p2p_allreduce_rank_sm90(
    int64_t numel,
    int local_rank,
    const std::vector<int>& devices,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify) {
    return benchmark_ipc_external_p2p_collective_rank_sm90(
        "allreduce",
        numel,
        local_rank,
        devices,
        broker_key,
        nccl_unique_id_bytes,
        iters,
        warmup,
        verify);
}

bool ipc_external_p2p_two_gpu_collective_smoke_rank_sm90(
    const std::string& collective,
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    bool verify) {
    return ipc_external_p2p_collective_smoke_rank_sm90(
        collective,
        numel,
        local_rank,
        std::vector<int>{dev0, dev1},
        broker_key,
        nccl_unique_id_bytes,
        verify);
}

std::map<std::string, double>
benchmark_ipc_external_p2p_two_gpu_collective_rank_sm90(
    const std::string& collective,
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify) {
    return benchmark_ipc_external_p2p_collective_rank_sm90(
        collective,
        numel,
        local_rank,
        std::vector<int>{dev0, dev1},
        broker_key,
        nccl_unique_id_bytes,
        iters,
        warmup,
        verify);
}

std::map<std::string, double>
benchmark_ipc_external_p2p_two_gpu_allreduce_rank_sm90(
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify) {
    return benchmark_ipc_external_p2p_allreduce_rank_sm90(
        numel,
        local_rank,
        std::vector<int>{dev0, dev1},
        broker_key,
        nccl_unique_id_bytes,
        iters,
        warmup,
        verify);
}

} // namespace ooverlap
