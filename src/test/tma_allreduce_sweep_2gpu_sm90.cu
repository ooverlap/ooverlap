#include "test/tma_allreduce_sweep_2gpu_sm90.h"

#include "comm/launch_config.h"
#include "comm/ooverlap_comm.h"
#include "comm/ooverlap_comm_internal.h"
#include "comm/params.h"
#include "comm/tma_multi_gpu_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#define OOVERLAP_SWEEP_NCCL_CHECK(cmd)                                        \
    do {                                                                      \
        ncclResult_t result__ = (cmd);                                        \
        if (result__ != ncclSuccess) {                                        \
            throw std::runtime_error(                                         \
                std::string("NCCL error: ") + ncclGetErrorString(result__));  \
        }                                                                     \
    } while (0)

namespace ooverlap {
namespace {

using json = nlohmann::json;

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

void check_oo(
    oo_status_t status,
    const char* what) {
    if (status != OO_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + " failed: " + oo_status_string(status));
    }
}

const char* kernel_kind_name(
    SweepKernelKind kind) {
    switch (kind) {
        case SweepKernelKind::kTmaCopy:
            return "tma_copy";
        case SweepKernelKind::kSeqFastGmem:
            return "seq_fast_gmem";
        case SweepKernelKind::kOverlapFastGmem:
            return "overlap_fast_gmem";
        default:
            return "unknown";
    }
}

SweepKernelKind parse_kernel_kind(
    const std::string& name) {
    if (name == "tma_copy" || name == "normal" || name == "tma") {
        return SweepKernelKind::kTmaCopy;
    }

    if (name == "seq_fast_gmem" ||
        name == "not_fused" ||
        name == "fast_gmem_seq") {
        return SweepKernelKind::kSeqFastGmem;
    }

    if (name == "overlap_fast_gmem" ||
        name == "fused" ||
        name == "fast_gmem_overlap") {
        return SweepKernelKind::kOverlapFastGmem;
    }

    throw std::invalid_argument("unknown kernel kind: " + name);
}

comm::AllreducePlanKind launch_kernel_kind(
    SweepKernelKind kind) {
    switch (kind) {
        case SweepKernelKind::kTmaCopy:
            return comm::AllreducePlanKind::TmaCopy;
        case SweepKernelKind::kSeqFastGmem:
            return comm::AllreducePlanKind::SeqFastGmem;
        case SweepKernelKind::kOverlapFastGmem:
            return comm::AllreducePlanKind::OverlapFastGmem;
        default:
            return comm::AllreducePlanKind::TmaCopy;
    }
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
        const int64_t numel = scenario.at("numel").get<int64_t>();

        if (numel <= 0) {
            throw std::invalid_argument("numel must be > 0");
        }

        return static_cast<size_t>(numel);
    }

    if (scenario.contains("bytes_per_rank")) {
        const int64_t bytes =
            scenario.at("bytes_per_rank").get<int64_t>();

        if (bytes <= 0 || (bytes % static_cast<int64_t>(sizeof(half))) != 0) {
            throw std::invalid_argument(
                "bytes_per_rank must be positive and divisible by sizeof(half)");
        }

        return static_cast<size_t>(bytes / static_cast<int64_t>(sizeof(half)));
    }

    throw std::invalid_argument("scenario must contain numel or bytes_per_rank");
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

    sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync fill inputs");
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
        cudaMemcpyAsync(
            rank0_work,
            rank0_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream0),
        "cudaMemcpyAsync(rank0_src -> rank0_work)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            rank1_work,
            rank1_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream1),
        "cudaMemcpyAsync(rank1_src -> rank1_work)");
}

void prepare_work_buffers(
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_work,
    half* rank1_work,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
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

    sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync reset working inputs");
}

int* ready_signal_ptr(
    oo_group_t* group,
    int rank) {
    if (group == nullptr ||
        rank < 0 ||
        rank >= group->num_devices ||
        group->ready_signal_slots[rank].ptr == nullptr) {
        throw std::runtime_error("ready_signal_ptr: invalid ready signal");
    }

    return reinterpret_cast<int*>(group->ready_signal_slots[rank].ptr);
}

void reset_ready_signals(
    oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    for (int r = 0; r < group->num_devices; ++r) {
        oo_ready_signal& slot = group->ready_signal_slots[r];

        if (slot.ptr == nullptr || slot.owner_device < 0) {
            continue;
        }

        system::runtime::set_device(slot.owner_device);
        system::runtime::check_cuda(
            cudaMemset(slot.ptr, 0, sizeof(int)),
            "cudaMemset(ready signal)");
    }
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
    system::runtime::check_cuda(
        cudaEventCreate(&start0),
        "cudaEventCreate(start0)");
    system::runtime::check_cuda(
        cudaEventCreate(&stop0),
        "cudaEventCreate(stop0)");
    system::runtime::check_cuda(
        cudaEventRecord(start0, stream0),
        "cudaEventRecord(start0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaEventCreate(&start1),
        "cudaEventCreate(start1)");
    system::runtime::check_cuda(
        cudaEventCreate(&stop1),
        "cudaEventCreate(stop1)");
    system::runtime::check_cuda(
        cudaEventRecord(start1, stream1),
        "cudaEventRecord(start1)");

    for (int i = 0; i < iters; ++i) {
        launch_once(i);
    }

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

    system::runtime::set_device(dev0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);

    system::runtime::set_device(dev1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    return static_cast<double>(std::max(ms0, ms1));
}

void launch_candidate_once(
    SweepKernelKind kernel,
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
    comm::LaunchConfig launch_config) {
    launch_config.plan_kind = launch_kernel_kind(kernel);

    void* rank0_peers[] = {
        rank1_work,
    };

    const int* rank0_peer_ready[] = {
        rank1_ready,
    };

    cudaError_t err0 =
        enqueue_tma_multi_gpu_allreduce_rank_sm90(
            rank0_work,
            rank0_work,
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
            launch_config);

    void* rank1_peers[] = {
        rank0_work,
    };

    const int* rank1_peer_ready[] = {
        rank0_ready,
    };

    cudaError_t err1 =
        enqueue_tma_multi_gpu_allreduce_rank_sm90(
            rank1_work,
            rank1_work,
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
            launch_config);

    system::runtime::check_cuda(err0, "enqueue sweep candidate rank0");
    system::runtime::check_cuda(err1, "enqueue sweep candidate rank1");
}

void run_candidate_iters(
    oo_group_t* group,
    SweepKernelKind kernel,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters,
    comm::LaunchConfig launch_config) {
    if (iters <= 0) {
        return;
    }

    reset_ready_signals(group);

    int* rank0_ready = ready_signal_ptr(group, 0);
    int* rank1_ready = ready_signal_ptr(group, 1);

    for (int i = 0; i < iters; ++i) {
        launch_candidate_once(
            kernel,
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
            launch_config);
    }
}

double elapsed_ms_candidate(
    oo_group_t* group,
    SweepKernelKind kernel,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters,
    comm::LaunchConfig launch_config) {
    reset_ready_signals(group);

    int epoch = 1;
    int* rank0_ready = ready_signal_ptr(group, 0);
    int* rank1_ready = ready_signal_ptr(group, 1);

    return elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int) {
            launch_candidate_once(
                kernel,
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
                launch_config);
        });
}

void run_nccl_iters(
    const half* rank0_src,
    const half* rank1_src,
    half* nccl_rank0_out,
    half* nccl_rank1_out,
    size_t numel,
    ncclComm_t* comms,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    if (iters <= 0) {
        return;
    }

    for (int i = 0; i < iters; ++i) {
        OOVERLAP_SWEEP_NCCL_CHECK(ncclGroupStart());

        OOVERLAP_SWEEP_NCCL_CHECK(
            ncclAllReduce(
                rank0_src,
                nccl_rank0_out,
                numel,
                ncclFloat16,
                ncclSum,
                comms[0],
                stream0));

        OOVERLAP_SWEEP_NCCL_CHECK(
            ncclAllReduce(
                rank1_src,
                nccl_rank1_out,
                numel,
                ncclFloat16,
                ncclSum,
                comms[1],
                stream1));

        OOVERLAP_SWEEP_NCCL_CHECK(ncclGroupEnd());
    }
}

double elapsed_ms_nccl(
    const half* rank0_src,
    const half* rank1_src,
    half* nccl_rank0_out,
    half* nccl_rank1_out,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters) {
    return elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int) {
            OOVERLAP_SWEEP_NCCL_CHECK(ncclGroupStart());

            OOVERLAP_SWEEP_NCCL_CHECK(
                ncclAllReduce(
                    rank0_src,
                    nccl_rank0_out,
                    numel,
                    ncclFloat16,
                    ncclSum,
                    comms[0],
                    stream0));

            OOVERLAP_SWEEP_NCCL_CHECK(
                ncclAllReduce(
                    rank1_src,
                    nccl_rank1_out,
                    numel,
                    ncclFloat16,
                    ncclSum,
                    comms[1],
                    stream1));

            OOVERLAP_SWEEP_NCCL_CHECK(ncclGroupEnd());
        });
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
    const double avg_ms = total_ms / static_cast<double>(iters);

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
    row["effective_gbps_per_rank"] = gbps_per_rank;
    row["effective_gbps_aggregate_2gpu"] = gbps_aggregate;
}

void add_env_metadata(
    json& row) {
    const std::string ooverlap_max_ctas = getenv_string("OOVERLAP_MAX_CTAS");
    const std::string nccl_max_ctas = getenv_string("NCCL_MAX_CTAS");
    const std::string nccl_min_ctas = getenv_string("NCCL_MIN_CTAS");
    const std::string nccl_algo = getenv_string("NCCL_ALGO");
    const std::string nccl_proto = getenv_string("NCCL_PROTO");

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

json run_ooverlap_scenario(
    const json& root,
    const json& scenario,
    int scenario_index) {
    const json id = scenario_id(scenario, scenario_index);

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

    const size_t numel = scenario_numel(scenario);
    const size_t bytes = numel * sizeof(half);

    const std::string kernel_name =
        scenario.value("kernel", std::string("seq_fast_gmem"));

    const SweepKernelKind kernel =
        parse_kernel_kind(kernel_name);

    comm::LaunchConfig config{};
    config.threads = scenario.value("threads", config.threads);
    config.window_chunks = scenario.value("window_chunks", config.window_chunks);
    config.chunk_bytes = scenario.value("chunk_bytes", config.chunk_bytes);
    config.stage_depth = scenario.value("stage_depth", config.stage_depth);
    config.max_ctas =
        scenario.value(
            "max_ctas",
            getenv_int_or("OOVERLAP_MAX_CTAS", config.max_ctas));
    config.plan_kind = launch_kernel_kind(kernel);

    json row;
    row["id"] = id;
    row["status"] = "ok";
    row["backend"] = "ooverlap";
    row["kernel"] = kernel_kind_name(kernel);
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

    if (kernel == SweepKernelKind::kOverlapFastGmem &&
        !comm::launch_config_valid_for_overlap(config)) {
        row["status"] = "skipped";
        row["reason"] = "invalid overlap launch config";
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

    try {
        int devices[2] = {
            dev0,
            dev1,
        };

        check_oo(oo_group_create(devices, 2, &group), "oo_group_create");
        check_oo(oo_node_create(group, 0, &node0), "oo_node_create(rank0)");
        check_oo(oo_node_create(group, 1, &node1), "oo_node_create(rank1)");

        const int node0_dev = oo_node_device(node0);
        const int node1_dev = oo_node_device(node1);

        stream0 =
            system::runtime::create_stream_on_device(node0_dev);
        stream1 =
            system::runtime::create_stream_on_device(node1_dev);

        system::runtime::set_device(node0_dev);
        system::runtime::check_cuda(
            cudaMalloc(&rank0_src, bytes),
            "cudaMalloc(rank0_src)");

        system::runtime::set_device(node1_dev);
        system::runtime::check_cuda(
            cudaMalloc(&rank1_src, bytes),
            "cudaMalloc(rank1_src)");

        check_oo(
            oo_buffer_alloc(node0, bytes, &rank0_buf),
            "oo_buffer_alloc(rank0)");
        check_oo(
            oo_buffer_alloc(node1, bytes, &rank1_buf),
            "oo_buffer_alloc(rank1)");

        half* rank0_work =
            reinterpret_cast<half*>(oo_buffer_ptr(rank0_buf));
        half* rank1_work =
            reinterpret_cast<half*>(oo_buffer_ptr(rank1_buf));

        fill_inputs(
            rank0_src,
            rank1_src,
            static_cast<int64_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            rank0_work,
            rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        run_candidate_iters(
            group,
            kernel,
            rank0_work,
            rank1_work,
            numel,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            warmup,
            config);

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync ooverlap warmup");

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            rank0_work,
            rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        const double total_ms =
            elapsed_ms_candidate(
                group,
                kernel,
                rank0_work,
                rank1_work,
                numel,
                node0_dev,
                node1_dev,
                stream0,
                stream1,
                iters,
                config);

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync ooverlap timed");

        add_common_metrics(
            row,
            numel,
            bytes,
            iters,
            warmup,
            node0_dev,
            node1_dev,
            total_ms);

        add_env_metadata(row);

        system::runtime::set_device(node0_dev);
        cudaFree(rank0_src);
        rank0_src = nullptr;

        system::runtime::set_device(node1_dev);
        cudaFree(rank1_src);
        rank1_src = nullptr;

        oo_buffer_destroy(rank0_buf);
        rank0_buf = nullptr;

        oo_buffer_destroy(rank1_buf);
        rank1_buf = nullptr;

        oo_node_destroy(node0);
        node0 = nullptr;

        oo_node_destroy(node1);
        node1 = nullptr;

        oo_group_destroy(group);
        group = nullptr;

        system::runtime::destroy_stream_on_device(node0_dev, stream0);
        stream0 = nullptr;

        system::runtime::destroy_stream_on_device(node1_dev, stream1);
        stream1 = nullptr;

        return row;
    } catch (...) {
        const int cleanup_dev0 =
            node0 != nullptr ? oo_node_device(node0) : dev0;
        const int cleanup_dev1 =
            node1 != nullptr ? oo_node_device(node1) : dev1;

        if (rank0_src != nullptr) {
            system::runtime::set_device(cleanup_dev0);
            cudaFree(rank0_src);
        }

        if (rank1_src != nullptr) {
            system::runtime::set_device(cleanup_dev1);
            cudaFree(rank1_src);
        }

        if (rank0_buf != nullptr) {
            oo_buffer_destroy(rank0_buf);
        }

        if (rank1_buf != nullptr) {
            oo_buffer_destroy(rank1_buf);
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
            system::runtime::destroy_stream_on_device(cleanup_dev0, stream0);
        }

        if (stream1 != nullptr) {
            system::runtime::destroy_stream_on_device(cleanup_dev1, stream1);
        }

        throw;
    }
}

json run_nccl_scenario(
    const json& root,
    const json& scenario,
    int scenario_index) {
    const json id = scenario_id(scenario, scenario_index);

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

    const size_t numel = scenario_numel(scenario);
    const size_t bytes = numel * sizeof(half);

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;
    half* rank0_out = nullptr;
    half* rank1_out = nullptr;

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    ncclComm_t comms[2] = {
        nullptr,
        nullptr,
    };

    bool nccl_initialized = false;

    json row;
    row["id"] = id;
    row["status"] = "ok";
    row["backend"] = "nccl";
    row["kernel"] = "nccl";

    try {
        system::runtime::ensure_context_on_device(dev0);
        system::runtime::ensure_context_on_device(dev1);

        stream0 =
            system::runtime::create_stream_on_device(dev0);
        stream1 =
            system::runtime::create_stream_on_device(dev1);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaMalloc(&rank0_src, bytes),
            "cudaMalloc(rank0_src)");
        system::runtime::check_cuda(
            cudaMalloc(&rank0_out, bytes),
            "cudaMalloc(rank0_out)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaMalloc(&rank1_src, bytes),
            "cudaMalloc(rank1_src)");
        system::runtime::check_cuda(
            cudaMalloc(&rank1_out, bytes),
            "cudaMalloc(rank1_out)");

        fill_inputs(
            rank0_src,
            rank1_src,
            static_cast<int64_t>(numel),
            dev0,
            dev1,
            stream0,
            stream1);

        int devices[2] = {
            dev0,
            dev1,
        };

        OOVERLAP_SWEEP_NCCL_CHECK(
            ncclCommInitAll(comms, 2, devices));

        nccl_initialized = true;

        run_nccl_iters(
            rank0_src,
            rank1_src,
            rank0_out,
            rank1_out,
            numel,
            comms,
            stream0,
            stream1,
            warmup);

        sync_two_streams(
            dev0,
            stream0,
            dev1,
            stream1,
            "sync nccl warmup");

        const double total_ms =
            elapsed_ms_nccl(
                rank0_src,
                rank1_src,
                rank0_out,
                rank1_out,
                numel,
                dev0,
                dev1,
                stream0,
                stream1,
                comms,
                iters);

        sync_two_streams(
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

        ncclCommDestroy(comms[0]);
        comms[0] = nullptr;

        ncclCommDestroy(comms[1]);
        comms[1] = nullptr;

        nccl_initialized = false;

        system::runtime::set_device(dev0);
        cudaFree(rank0_src);
        rank0_src = nullptr;
        cudaFree(rank0_out);
        rank0_out = nullptr;

        system::runtime::set_device(dev1);
        cudaFree(rank1_src);
        rank1_src = nullptr;
        cudaFree(rank1_out);
        rank1_out = nullptr;

        system::runtime::destroy_stream_on_device(dev0, stream0);
        stream0 = nullptr;

        system::runtime::destroy_stream_on_device(dev1, stream1);
        stream1 = nullptr;

        return row;
    } catch (...) {
        if (nccl_initialized) {
            if (comms[0] != nullptr) {
                ncclCommDestroy(comms[0]);
            }

            if (comms[1] != nullptr) {
                ncclCommDestroy(comms[1]);
            }
        }

        if (rank0_src != nullptr) {
            system::runtime::set_device(dev0);
            cudaFree(rank0_src);
        }

        if (rank0_out != nullptr) {
            system::runtime::set_device(dev0);
            cudaFree(rank0_out);
        }

        if (rank1_src != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(rank1_src);
        }

        if (rank1_out != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(rank1_out);
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

std::string benchmark_tma_two_gpu_allreduce_sweep_json_sm90(
    const std::string& request_json) {
    json response;
    response["ok"] = true;
    response["results"] = json::array();
    response["errors"] = json::array();

    try {
        const json root = json::parse(request_json);

        if (!root.contains("scenarios") || !root.at("scenarios").is_array()) {
            throw std::invalid_argument("request must contain scenarios array");
        }

        const json& scenarios = root.at("scenarios");

        for (size_t i = 0; i < scenarios.size(); ++i) {
            const json& scenario = scenarios.at(i);

            try {
                response["results"].push_back(
                    run_one_scenario(
                        root,
                        scenario,
                        static_cast<int>(i)));
            } catch (const std::exception& exc) {
                response["ok"] = false;

                json err;
                err["id"] = scenario_id(scenario, static_cast<int>(i));
                err["index"] = i;
                err["error"] = exc.what();

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
