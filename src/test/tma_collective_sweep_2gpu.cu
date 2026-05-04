#include "test/tma_collective_sweep_2gpu.h"

#include "comm/launch_config.h"
#include "comm/ooverlap_comm_internal.h"
#include "comm/params.h"
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

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ooverlap {
namespace {

using json = nlohmann::json;
using testing::TestCollective;

enum class SweepKernelKind {
    Tma = 0,
    SeqFastGmem = 1,
    OverlapFastGmem = 2,
};

const char* kernel_kind_name(
    SweepKernelKind kind) {
    switch (kind) {
        case SweepKernelKind::Tma:
            return "tma";
        case SweepKernelKind::SeqFastGmem:
            return "seq_fast_gmem";
        case SweepKernelKind::OverlapFastGmem:
            return "overlap_fast_gmem";
        default:
            return "unknown";
    }
}

SweepKernelKind parse_kernel_kind(
    const std::string& name) {
    if (name == "tma" ||
        name == "tma_copy" ||
        name == "normal") {
        return SweepKernelKind::Tma;
    }

    if (name == "seq_fast_gmem" ||
        name == "not_fused" ||
        name == "fast_gmem_seq") {
        return SweepKernelKind::SeqFastGmem;
    }

    if (name == "overlap_fast_gmem" ||
        name == "fused" ||
        name == "fast_gmem_overlap") {
        return SweepKernelKind::OverlapFastGmem;
    }

    throw std::invalid_argument("unknown kernel kind: " + name);
}

bool kernel_supported_for_collective(
    TestCollective collective,
    SweepKernelKind kernel) {
    if (kernel != SweepKernelKind::OverlapFastGmem) {
        return true;
    }

    return collective == TestCollective::AllReduce;
}

comm::LaunchConfig make_launch_config(
    TestCollective collective,
    SweepKernelKind kernel) {
    if (collective == TestCollective::AllReduce) {
        if (kernel == SweepKernelKind::Tma) {
            return comm::make_allreduce_launch_config(
                comm::AllReducePlanKind::TmaCopy);
        }

        if (kernel == SweepKernelKind::SeqFastGmem) {
            return comm::make_allreduce_launch_config(
                comm::AllReducePlanKind::SeqFastCopyGmem);
        }

        return comm::make_allreduce_launch_config(
            comm::AllReducePlanKind::OverlapFastCopyGmem);
    }

    if (collective == TestCollective::ReduceScatter) {
        if (kernel == SweepKernelKind::Tma) {
            return comm::make_reduce_scatter_launch_config(
                comm::ReduceScatterPlanKind::TmaReduce);
        }

        if (kernel == SweepKernelKind::SeqFastGmem) {
            return comm::make_reduce_scatter_launch_config(
                comm::ReduceScatterPlanKind::SeqFastAddGmem);
        }
    }

    if (collective == TestCollective::AllGather) {
        if (kernel == SweepKernelKind::Tma) {
            return comm::make_all_gather_launch_config(
                comm::AllGatherPlanKind::TmaCopy);
        }

        if (kernel == SweepKernelKind::SeqFastGmem) {
            return comm::make_all_gather_launch_config(
                comm::AllGatherPlanKind::SeqFastCopyGmem);
        }
    }

    throw std::invalid_argument("unsupported collective/kernel combination");
}

std::string getenv_string(
    const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

int getenv_int_or(
    const char* name,
    int fallback) {
    const char* value = std::getenv(name);

    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    return std::atoi(value);
}

template <typename T>
T get_with_fallback(
    const json& scenario,
    const json& root,
    const char* key,
    T fallback) {
    if (scenario.contains(key) && !scenario.at(key).is_null()) {
        return scenario.at(key).get<T>();
    }

    if (root.contains(key) && !root.at(key).is_null()) {
        return root.at(key).get<T>();
    }

    return fallback;
}

json scenario_id(
    const json& scenario,
    int index) {
    if (scenario.contains("id")) {
        return scenario.at("id");
    }

    return index;
}

size_t scenario_numel(
    const json& scenario) {
    if (scenario.contains("numel")) {
        const int64_t numel =
            scenario.at("numel").get<int64_t>();

        if (numel <= 0) {
            throw std::invalid_argument("numel must be > 0");
        }

        return static_cast<size_t>(numel);
    }

    if (scenario.contains("bytes_per_rank")) {
        const int64_t bytes =
            scenario.at("bytes_per_rank").get<int64_t>();

        if (bytes <= 0 ||
            (bytes % static_cast<int64_t>(sizeof(half))) != 0) {
            throw std::invalid_argument(
                "bytes_per_rank must be positive and divisible by sizeof(half)");
        }

        return static_cast<size_t>(
            bytes / static_cast<int64_t>(sizeof(half)));
    }

    throw std::invalid_argument("scenario must contain numel or bytes_per_rank");
}

void add_env_metadata(json& row) {
    const std::string ooverlap_max_ctas =
        getenv_string("OOVERLAP_MAX_CTAS");

    const std::string nccl_max_ctas =
        getenv_string("NCCL_MAX_CTAS");

    const std::string nccl_min_ctas =
        getenv_string("NCCL_MIN_CTAS");

    const std::string nccl_algo =
        getenv_string("NCCL_ALGO");

    const std::string nccl_proto =
        getenv_string("NCCL_PROTO");

    row["ooverlap_max_ctas_env"] =
        ooverlap_max_ctas.empty() ? json(nullptr) : json(ooverlap_max_ctas);

    row["nccl_max_ctas_env"] =
        nccl_max_ctas.empty() ? json(nullptr) : json(nccl_max_ctas);

    row["nccl_min_ctas_env"] =
        nccl_min_ctas.empty() ? json(nullptr) : json(nccl_min_ctas);

    row["nccl_algo_env"] =
        nccl_algo.empty() ? json(nullptr) : json(nccl_algo);

    row["nccl_proto_env"] =
        nccl_proto.empty() ? json(nullptr) : json(nccl_proto);
}

void add_common_metrics(
    json& row,
    size_t numel,
    size_t bytes,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    double total_ms) {
    const double avg_ms =
        total_ms / static_cast<double>(iters);

    const double gbps_per_rank =
        avg_ms > 0.0
            ? static_cast<double>(bytes) / (avg_ms * 1.0e-3) / 1.0e9
            : 0.0;

    const double gbps_aggregate =
        avg_ms > 0.0
            ? 2.0 * static_cast<double>(bytes) / (avg_ms * 1.0e-3) / 1.0e9
            : 0.0;

    row["numel"] = numel;
    row["bytes_per_rank"] = bytes;
    row["iters"] = iters;
    row["warmup"] = warmup;
    row["dev0"] = dev0;
    row["dev1"] = dev1;
    row["total_ms"] = total_ms;
    row["avg_ms"] = avg_ms;
    row["latency_us"] = avg_ms * 1000.0;
    row["effective_gbps_per_rank"] = gbps_per_rank;
    row["effective_gbps_aggregate_2gpu"] = gbps_aggregate;
}

void launch_ooverlap_rank_once(
    TestCollective collective,
    half* local_work,
    half* peer_work,
    size_t numel,
    int rank,
    int local_device,
    cudaStream_t stream,
    int* local_ready,
    int* peer_ready,
    int collective_epoch,
    comm::LaunchConfig config) {
    void* peer_bufs[] = {
        peer_work,
    };

    const int* peer_ready_signals[] = {
        peer_ready,
    };

    cudaError_t err = cudaSuccess;

    if (collective == TestCollective::AllReduce) {
        err =
            enqueue_tma_multi_gpu_allreduce_rank_sm90(
                local_work,
                local_work,
                peer_bufs,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                rank,
                2,
                local_device,
                stream,
                local_ready,
                peer_ready_signals,
                collective_epoch,
                config);
    } else if (collective == TestCollective::ReduceScatter) {
        err =
            enqueue_tma_multi_gpu_reduce_scatter_rank_sm90(
                local_work,
                local_work,
                peer_bufs,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                rank,
                2,
                local_device,
                stream,
                local_ready,
                peer_ready_signals,
                collective_epoch,
                config);
    } else if (collective == TestCollective::AllGather) {
        err =
            enqueue_tma_multi_gpu_all_gather_rank_sm90(
                local_work,
                local_work,
                peer_bufs,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                rank,
                2,
                local_device,
                stream,
                local_ready,
                peer_ready_signals,
                collective_epoch,
                config);
    } else {
        throw std::invalid_argument("unknown ooverlap collective");
    }

    testing::check_cuda(err, "enqueue ooverlap sweep rank");
}

void launch_ooverlap_candidate_once(
    TestCollective collective,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int* rank0_ready,
    int* rank1_ready,
    int collective_epoch,
    comm::LaunchConfig config) {
    launch_ooverlap_rank_once(
        collective,
        rank0_work,
        rank1_work,
        numel,
        0,
        dev0,
        stream0,
        rank0_ready,
        rank1_ready,
        collective_epoch,
        config);

    launch_ooverlap_rank_once(
        collective,
        rank1_work,
        rank0_work,
        numel,
        1,
        dev1,
        stream1,
        rank1_ready,
        rank0_ready,
        collective_epoch,
        config);
}

void run_ooverlap_candidate_iters(
    oo_group_t* group,
    TestCollective collective,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters,
    comm::LaunchConfig config) {
    if (iters <= 0) {
        return;
    }

    testing::reset_ready_signals(group);

    int* rank0_ready =
        testing::ready_signal_ptr(group, 0);

    int* rank1_ready =
        testing::ready_signal_ptr(group, 1);

    for (int i = 0; i < iters; ++i) {
        launch_ooverlap_candidate_once(
            collective,
            rank0_work,
            rank1_work,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            rank0_ready,
            rank1_ready,
            i + 1,
            config);
    }
}

double elapsed_ms_ooverlap_candidate(
    oo_group_t* group,
    TestCollective collective,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters,
    comm::LaunchConfig config) {
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
            launch_ooverlap_candidate_once(
                collective,
                rank0_work,
                rank1_work,
                numel,
                dev0,
                dev1,
                stream0,
                stream1,
                rank0_ready,
                rank1_ready,
                epoch++,
                config);
        });
}

void launch_nccl_collective_once(
    TestCollective collective,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    ncclComm_t* comms,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());

    testing::launch_nccl_collective_fp16(
        collective,
        comms[0],
        rank0_work,
        numel,
        0,
        2,
        stream0);

    testing::launch_nccl_collective_fp16(
        collective,
        comms[1],
        rank1_work,
        numel,
        1,
        2,
        stream1);

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
}

void run_nccl_iters(
    TestCollective collective,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    ncclComm_t* comms,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    if (iters <= 0) {
        return;
    }

    for (int i = 0; i < iters; ++i) {
        launch_nccl_collective_once(
            collective,
            rank0_work,
            rank1_work,
            numel,
            comms,
            stream0,
            stream1);
    }
}

double elapsed_ms_nccl(
    TestCollective collective,
    half* rank0_work,
    half* rank1_work,
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
            launch_nccl_collective_once(
                collective,
                rank0_work,
                rank1_work,
                numel,
                comms,
                stream0,
                stream1);
        });
}

comm::LaunchConfig scenario_launch_config(
    const json& scenario,
    TestCollective collective,
    SweepKernelKind kernel) {
    comm::LaunchConfig config =
        make_launch_config(collective, kernel);

    config.threads =
        scenario.value("threads", config.threads);

    config.window_chunks =
        scenario.value("window_chunks", config.window_chunks);

    config.chunk_bytes =
        scenario.value("chunk_bytes", config.chunk_bytes);

    config.stage_depth =
        scenario.value("stage_depth", config.stage_depth);

    config.max_ctas =
        scenario.value(
            "max_ctas",
            getenv_int_or("OOVERLAP_MAX_CTAS", config.max_ctas));

    return config;
}

json skipped_row(
    const json& scenario,
    int scenario_index,
    const char* backend,
    const char* collective,
    const char* kernel,
    const char* reason) {
    json row;
    row["id"] = scenario_id(scenario, scenario_index);
    row["status"] = "skipped";
    row["backend"] = backend;
    row["collective"] = collective;
    row["kernel"] = kernel;
    row["reason"] = reason;
    add_env_metadata(row);
    return row;
}

json run_ooverlap_scenario(
    const json& root,
    const json& scenario,
    int scenario_index) {
    const int iters =
        get_with_fallback<int>(scenario, root, "iters", 100);

    const int warmup =
        get_with_fallback<int>(scenario, root, "warmup", 20);

    const int dev0 =
        get_with_fallback<int>(scenario, root, "dev0", 0);

    const int dev1 =
        get_with_fallback<int>(scenario, root, "dev1", 1);

    if (iters <= 0 || warmup < 0) {
        throw std::invalid_argument("iters must be > 0 and warmup must be >= 0");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument("dev0 and dev1 must differ");
    }

    const TestCollective collective =
        testing::parse_collective(
            scenario.value("collective", std::string("allreduce")));

    const SweepKernelKind kernel =
        parse_kernel_kind(
            scenario.value("kernel", std::string("seq_fast_gmem")));

    const size_t numel =
        scenario_numel(scenario);

    testing::validate_numel_for_collective(
        collective,
        static_cast<int64_t>(numel),
        2);

    const size_t bytes =
        numel * sizeof(half);

    const char* collective_name =
        testing::collective_name(collective);

    const char* kernel_name =
        kernel_kind_name(kernel);

    if (!kernel_supported_for_collective(collective, kernel)) {
        return skipped_row(
            scenario,
            scenario_index,
            "ooverlap",
            collective_name,
            kernel_name,
            "kernel not supported for collective");
    }

    comm::LaunchConfig config =
        scenario_launch_config(
            scenario,
            collective,
            kernel);

    json row;
    row["id"] = scenario_id(scenario, scenario_index);
    row["status"] = "ok";
    row["backend"] = "ooverlap";
    row["collective"] = collective_name;
    row["kernel"] = kernel_name;
    row["threads"] = config.threads;
    row["max_ctas"] = config.max_ctas;
    row["window_chunks"] = config.window_chunks;
    row["chunk_bytes"] = config.chunk_bytes;
    row["stage_depth"] = config.stage_depth;

    if (!comm::launch_config_valid(config)) {
        row["status"] = "skipped";
        row["reason"] = "invalid launch config";
        add_env_metadata(row);
        return row;
    }

    oo_group_t* group = nullptr;
    oo_node_t* node0 = nullptr;
    oo_node_t* node1 = nullptr;
    oo_buffer_t* rank0_buf = nullptr;
    oo_buffer_t* rank1_buf = nullptr;

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    int cleanup_dev0 = dev0;
    int cleanup_dev1 = dev1;

    try {
        int devices[2] = {
            dev0,
            dev1,
        };

        testing::check_oo(
            oo_group_create(devices, 2, &group),
            "oo_group_create");

        testing::check_oo(
            oo_node_create(group, 0, &node0),
            "oo_node_create(rank0)");

        testing::check_oo(
            oo_node_create(group, 1, &node1),
            "oo_node_create(rank1)");

        cleanup_dev0 = oo_node_device(node0);
        cleanup_dev1 = oo_node_device(node1);

        stream0 =
            system::runtime::create_stream_on_device(cleanup_dev0);

        stream1 =
            system::runtime::create_stream_on_device(cleanup_dev1);

        testing::cuda_malloc_half_on_device(
            cleanup_dev0,
            &rank0_src,
            bytes,
            "cudaMalloc(rank0_src)");

        testing::cuda_malloc_half_on_device(
            cleanup_dev1,
            &rank1_src,
            bytes,
            "cudaMalloc(rank1_src)");

        testing::check_oo(
            oo_buffer_alloc(node0, bytes, &rank0_buf),
            "oo_buffer_alloc(rank0)");

        testing::check_oo(
            oo_buffer_alloc(node1, bytes, &rank1_buf),
            "oo_buffer_alloc(rank1)");

        half* rank0_work =
            reinterpret_cast<half*>(oo_buffer_ptr(rank0_buf));

        half* rank1_work =
            reinterpret_cast<half*>(oo_buffer_ptr(rank1_buf));

        testing::fill_two_rank_sources_fp16(
            rank0_src,
            rank1_src,
            static_cast<int64_t>(numel),
            cleanup_dev0,
            cleanup_dev1,
            stream0,
            stream1);

        testing::prepare_two_work_buffers(
            rank0_src,
            rank1_src,
            rank0_work,
            rank1_work,
            bytes,
            cleanup_dev0,
            cleanup_dev1,
            stream0,
            stream1);

        run_ooverlap_candidate_iters(
            group,
            collective,
            rank0_work,
            rank1_work,
            numel,
            cleanup_dev0,
            cleanup_dev1,
            stream0,
            stream1,
            warmup,
            config);

        testing::sync_two_streams(
            cleanup_dev0,
            stream0,
            cleanup_dev1,
            stream1,
            "sync ooverlap warmup");

        testing::prepare_two_work_buffers(
            rank0_src,
            rank1_src,
            rank0_work,
            rank1_work,
            bytes,
            cleanup_dev0,
            cleanup_dev1,
            stream0,
            stream1);

        const double total_ms =
            elapsed_ms_ooverlap_candidate(
                group,
                collective,
                rank0_work,
                rank1_work,
                numel,
                cleanup_dev0,
                cleanup_dev1,
                stream0,
                stream1,
                iters,
                config);

        testing::sync_two_streams(
            cleanup_dev0,
            stream0,
            cleanup_dev1,
            stream1,
            "sync ooverlap timed");

        add_common_metrics(
            row,
            numel,
            bytes,
            iters,
            warmup,
            cleanup_dev0,
            cleanup_dev1,
            total_ms);

        add_env_metadata(row);

        testing::cuda_free_on_device(cleanup_dev0, rank0_src);
        testing::cuda_free_on_device(cleanup_dev1, rank1_src);

        testing::destroy_oo_buffer(rank0_buf);
        testing::destroy_oo_buffer(rank1_buf);
        testing::destroy_oo_node(node0);
        testing::destroy_oo_node(node1);
        testing::destroy_oo_group(group);

        testing::destroy_stream_on_device(cleanup_dev0, stream0);
        testing::destroy_stream_on_device(cleanup_dev1, stream1);

        return row;
    } catch (...) {
        testing::cuda_free_on_device(cleanup_dev0, rank0_src);
        testing::cuda_free_on_device(cleanup_dev1, rank1_src);

        testing::destroy_oo_buffer(rank0_buf);
        testing::destroy_oo_buffer(rank1_buf);
        testing::destroy_oo_node(node0);
        testing::destroy_oo_node(node1);
        testing::destroy_oo_group(group);

        testing::destroy_stream_on_device(cleanup_dev0, stream0);
        testing::destroy_stream_on_device(cleanup_dev1, stream1);

        throw;
    }
}

json run_nccl_scenario(
    const json& root,
    const json& scenario,
    int scenario_index) {
    const int iters =
        get_with_fallback<int>(scenario, root, "iters", 100);

    const int warmup =
        get_with_fallback<int>(scenario, root, "warmup", 20);

    const int dev0 =
        get_with_fallback<int>(scenario, root, "dev0", 0);

    const int dev1 =
        get_with_fallback<int>(scenario, root, "dev1", 1);

    if (iters <= 0 || warmup < 0) {
        throw std::invalid_argument("iters must be > 0 and warmup must be >= 0");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument("dev0 and dev1 must differ");
    }

    const TestCollective collective =
        testing::parse_collective(
            scenario.value("collective", std::string("allreduce")));

    const size_t numel =
        scenario_numel(scenario);

    testing::validate_numel_for_collective(
        collective,
        static_cast<int64_t>(numel),
        2);

    const size_t bytes =
        numel * sizeof(half);

    half* rank0_work = nullptr;
    half* rank1_work = nullptr;

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    ncclComm_t comms[2] = {
        nullptr,
        nullptr,
    };

    json row;
    row["id"] = scenario_id(scenario, scenario_index);
    row["status"] = "ok";
    row["backend"] = "nccl";
    row["collective"] = testing::collective_name(collective);
    row["kernel"] = "nccl";

    try {
        system::runtime::ensure_context_on_device(dev0);
        system::runtime::ensure_context_on_device(dev1);

        stream0 =
            system::runtime::create_stream_on_device(dev0);

        stream1 =
            system::runtime::create_stream_on_device(dev1);

        testing::cuda_malloc_half_on_device(
            dev0,
            &rank0_work,
            bytes,
            "cudaMalloc(rank0_work)");

        testing::cuda_malloc_half_on_device(
            dev1,
            &rank1_work,
            bytes,
            "cudaMalloc(rank1_work)");

        testing::fill_two_rank_sources_fp16(
            rank0_work,
            rank1_work,
            static_cast<int64_t>(numel),
            dev0,
            dev1,
            stream0,
            stream1);

        int devices[2] = {
            dev0,
            dev1,
        };

        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommInitAll(comms, 2, devices));

        run_nccl_iters(
            collective,
            rank0_work,
            rank1_work,
            numel,
            comms,
            stream0,
            stream1,
            warmup);

        testing::sync_two_streams(
            dev0,
            stream0,
            dev1,
            stream1,
            "sync nccl warmup");

        testing::fill_two_rank_sources_fp16(
            rank0_work,
            rank1_work,
            static_cast<int64_t>(numel),
            dev0,
            dev1,
            stream0,
            stream1);

        const double total_ms =
            elapsed_ms_nccl(
                collective,
                rank0_work,
                rank1_work,
                numel,
                dev0,
                dev1,
                stream0,
                stream1,
                comms,
                iters);

        testing::sync_two_streams(
            dev0,
            stream0,
            dev1,
            stream1,
            "sync nccl timed");

        add_common_metrics(
            row,
            numel,
            bytes,
            iters,
            warmup,
            dev0,
            dev1,
            total_ms);

        add_env_metadata(row);

        testing::destroy_nccl_comms(comms, 2);
        testing::cuda_free_on_device(dev0, rank0_work);
        testing::cuda_free_on_device(dev1, rank1_work);
        testing::destroy_stream_on_device(dev0, stream0);
        testing::destroy_stream_on_device(dev1, stream1);

        return row;
    } catch (...) {
        testing::destroy_nccl_comms(comms, 2);
        testing::cuda_free_on_device(dev0, rank0_work);
        testing::cuda_free_on_device(dev1, rank1_work);
        testing::destroy_stream_on_device(dev0, stream0);
        testing::destroy_stream_on_device(dev1, stream1);

        throw;
    }
}

json run_one_scenario(
    const json& root,
    const json& scenario,
    int scenario_index) {
    const std::string backend =
        scenario.value("backend", std::string("ooverlap"));

    if (backend == "ooverlap" || backend == "oo") {
        return run_ooverlap_scenario(root, scenario, scenario_index);
    }

    if (backend == "nccl") {
        return run_nccl_scenario(root, scenario, scenario_index);
    }

    throw std::invalid_argument("unknown backend: " + backend);
}

} // namespace

std::string benchmark_tma_two_gpu_collective_sweep_json(
    const std::string& request_json) {
    json response;
    response["ok"] = true;
    response["results"] = json::array();
    response["errors"] = json::array();

    try {
        const json root =
            json::parse(request_json);

        if (!root.contains("scenarios") ||
            !root.at("scenarios").is_array()) {
            throw std::invalid_argument(
                "request must contain scenarios array");
        }

        const json& scenarios =
            root.at("scenarios");

        for (size_t i = 0; i < scenarios.size(); ++i) {
            const json& scenario =
                scenarios.at(i);

            try {
                response["results"].push_back(
                    run_one_scenario(
                        root,
                        scenario,
                        static_cast<int>(i)));
            } catch (const std::exception& exc) {
                response["ok"] = false;

                json err;
                err["id"] =
                    scenario_id(scenario, static_cast<int>(i));
                err["index"] =
                    i;
                err["error"] =
                    exc.what();

                response["errors"].push_back(err);
            }
        }
    } catch (const std::exception& exc) {
        response["ok"] = false;

        json err;
        err["id"] = nullptr;
        err["index"] = nullptr;
        err["error"] = exc.what();

        response["errors"].push_back(err);
    }

    return response.dump();
}

} // namespace ooverlap
