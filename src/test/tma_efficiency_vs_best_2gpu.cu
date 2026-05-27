#include "test/tma_efficiency_vs_best_2gpu.h"

#include "ooverlap/comm.h"
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
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OOVERLAP_BENCH_VERIFY_RESULTS
#define OOVERLAP_BENCH_VERIFY_RESULTS 0
#endif

namespace ooverlap {
namespace {

using json = nlohmann::json;
using testing::TestCollective;

struct ModeSpec {
    const char* name = nullptr;
    oo_tuning_mode_t mode = OO_TUNING_BEST_PERFORMANCE;
};

class ScopedEnvOverride {
public:
    ScopedEnvOverride(
        const char* name,
        const std::string& value,
        bool enabled)
        : name_(name),
          enabled_(enabled) {
        const char* old = std::getenv(name_);

        if (old != nullptr) {
            had_old_ = true;
            old_value_ = old;
        }

        if (enabled_) {
            if (::setenv(name_, value.c_str(), 1) != 0) {
                throw std::runtime_error(
                    std::string("setenv failed for ") + name_);
            }
        }
    }

    ~ScopedEnvOverride() {
        if (!enabled_) {
            return;
        }

        if (had_old_) {
            (void)::setenv(name_, old_value_.c_str(), 1);
        } else {
            (void)::unsetenv(name_);
        }
    }

    ScopedEnvOverride(const ScopedEnvOverride&) = delete;
    ScopedEnvOverride& operator=(const ScopedEnvOverride&) = delete;

private:
    const char* name_ = nullptr;
    bool enabled_ = false;
    bool had_old_ = false;
    std::string old_value_;
};

std::vector<int64_t> read_i64_list_required(
    const json& root,
    const char* key) {
    if (!root.contains(key) || root.at(key).is_null()) {
        throw std::invalid_argument(std::string("request must contain ") + key);
    }

    const json& value = root.at(key);
    std::vector<int64_t> out;

    if (value.is_number_integer() || value.is_number_unsigned()) {
        out.push_back(value.get<int64_t>());
    } else if (value.is_array()) {
        out.reserve(value.size());

        for (const json& item : value) {
            if (!item.is_number_integer() && !item.is_number_unsigned()) {
                throw std::invalid_argument(
                    std::string(key) + " entries must be integers");
            }

            out.push_back(item.get<int64_t>());
        }
    } else {
        throw std::invalid_argument(
            std::string(key) + " must be an integer or array");
    }

    if (out.empty()) {
        throw std::invalid_argument(std::string(key) + " must not be empty");
    }

    for (int64_t x : out) {
        if (x <= 0) {
            throw std::invalid_argument(
                std::string(key) + " entries must be > 0");
        }
    }

    return out;
}

std::vector<std::string> read_string_list(
    const json& root,
    const char* key,
    std::vector<std::string> fallback) {
    if (!root.contains(key) || root.at(key).is_null()) {
        return fallback;
    }

    const json& value = root.at(key);

    if (value.is_string()) {
        return {value.get<std::string>()};
    }

    if (!value.is_array()) {
        throw std::invalid_argument(
            std::string(key) + " must be a string or array");
    }

    std::vector<std::string> out;
    out.reserve(value.size());

    for (const json& item : value) {
        if (!item.is_string()) {
            throw std::invalid_argument(
                std::string(key) + " entries must be strings");
        }

        out.push_back(item.get<std::string>());
    }

    if (out.empty()) {
        throw std::invalid_argument(std::string(key) + " must not be empty");
    }

    return out;
}

int read_int_or(
    const json& root,
    const char* key,
    int fallback) {
    if (!root.contains(key) || root.at(key).is_null()) {
        return fallback;
    }

    return root.at(key).get<int>();
}

bool read_bool_or(
    const json& root,
    const char* key,
    bool fallback) {
    if (!root.contains(key) || root.at(key).is_null()) {
        return fallback;
    }

    return root.at(key).get<bool>();
}

std::vector<int> read_max_cta_constraints(
    const json& root) {
    if (root.contains("ctas") && !root.at("ctas").is_null()) {
        const json& value = root.at("ctas");

        if (!value.is_array()) {
            throw std::invalid_argument("ctas must be an array");
        }

        std::vector<int> out;
        out.reserve(value.size());

        for (const json& item : value) {
            if (!item.is_number_integer() && !item.is_number_unsigned()) {
                throw std::invalid_argument("ctas entries must be integers");
            }

            const int max_ctas = item.get<int>();

            if (max_ctas <= 0) {
                throw std::invalid_argument("ctas entries must be > 0");
            }

            out.push_back(max_ctas);
        }

        if (out.empty()) {
            throw std::invalid_argument("ctas must not be empty");
        }

        return out;
    }

    if (root.contains("max_ctas") && !root.at("max_ctas").is_null()) {
        const int max_ctas = root.at("max_ctas").get<int>();

        if (max_ctas <= 0) {
            throw std::invalid_argument("max_ctas must be > 0");
        }

        return {max_ctas};
    }

    // -1 means: do not override OOVERLAP_MAX_CTAS. This lets the benchmark
    // evaluate the policy under the caller's ambient environment.
    return {-1};
}

std::vector<ModeSpec> read_modes(
    const json& root) {
    const std::vector<std::string> names =
        read_string_list(
            root,
            "modes",
            {"best_performance", "best_efficiency"});

    std::vector<ModeSpec> out;
    out.reserve(names.size());

    for (const std::string& name : names) {
        if (name == "best_performance" ||
            name == "performance" ||
            name == "best") {
            out.push_back(
                ModeSpec{
                    "best_performance",
                    OO_TUNING_BEST_PERFORMANCE,
                });
            continue;
        }

        if (name == "best_efficiency" ||
            name == "efficiency" ||
            name == "efficient") {
            out.push_back(
                ModeSpec{
                    "best_efficiency",
                    OO_TUNING_BEST_EFFICIENCY,
                });
            continue;
        }

        throw std::invalid_argument("unknown tuning mode: " + name);
    }

    return out;
}

void add_env_metadata(json& row) {
    const char* max_ctas = std::getenv("OOVERLAP_MAX_CTAS");
    const char* policy = std::getenv("OOVERLAP_TUNING_POLICY");
    const char* tolerance = std::getenv("OOVERLAP_TUNING_TOLERANCE");

    row["ooverlap_max_ctas_env"] =
        max_ctas != nullptr ? json(max_ctas) : json(nullptr);

    row["ooverlap_tuning_policy_env"] =
        policy != nullptr ? json(policy) : json(nullptr);

    row["ooverlap_tuning_tolerance_env"] =
        tolerance != nullptr ? json(tolerance) : json(nullptr);
}

void add_common_metrics(
    json& row,
    TestCollective collective,
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

    row["collective"] = testing::collective_name(collective);
    row["collective_code"] = testing::collective_code(collective);
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

void launch_public_tuned_once(
    TestCollective collective,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    oo_buffer_t* rank0_peers[] = {rank1_buf};
    oo_buffer_t* rank1_peers[] = {rank0_buf};

    if (collective == TestCollective::AllReduce) {
        testing::check_oo(
            oo_allreduce_tuned(
                node0,
                rank0_buf,
                rank0_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                tuning_mode,
                stream0),
            "oo_allreduce_tuned(rank0)");

        testing::check_oo(
            oo_allreduce_tuned(
                node1,
                rank1_buf,
                rank1_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                tuning_mode,
                stream1),
            "oo_allreduce_tuned(rank1)");

        return;
    }

    if (collective == TestCollective::ReduceScatter) {
        oo_tensor_slice_t slice0{};
        oo_tensor_slice_t slice1{};

        testing::check_oo(
            oo_reduce_scatter_tuned(
                node0,
                rank0_buf,
                rank0_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                tuning_mode,
                &slice0,
                stream0),
            "oo_reduce_scatter_tuned(rank0)");

        testing::check_oo(
            oo_reduce_scatter_tuned(
                node1,
                rank1_buf,
                rank1_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                tuning_mode,
                &slice1,
                stream1),
            "oo_reduce_scatter_tuned(rank1)");

        return;
    }

    if (collective == TestCollective::AllGather) {
        testing::check_oo(
            oo_all_gather_tuned(
                node0,
                rank0_buf,
                rank0_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                tuning_mode,
                stream0),
            "oo_all_gather_tuned(rank0)");

        testing::check_oo(
            oo_all_gather_tuned(
                node1,
                rank1_buf,
                rank1_peers,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                tuning_mode,
                stream1),
            "oo_all_gather_tuned(rank1)");

        return;
    }

    throw std::invalid_argument("unsupported public tuned collective");
}

void run_public_tuned_iters(
    oo_group_t* group,
    TestCollective collective,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    oo_tuning_mode_t tuning_mode,
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
        launch_public_tuned_once(
            collective,
            node0,
            node1,
            rank0_buf,
            rank1_buf,
            numel,
            tuning_mode,
            stream0,
            stream1);
    }

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync public tuned warmup");
}

double elapsed_ms_public_tuned(
    oo_group_t* group,
    TestCollective collective,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    oo_tuning_mode_t tuning_mode,
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
            launch_public_tuned_once(
                collective,
                node0,
                node1,
                rank0_buf,
                rank1_buf,
                numel,
                tuning_mode,
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

void cleanup(
    int dev0,
    int dev1,
    half*& rank0_src,
    half*& rank1_src,
    half*& nccl_rank0_buf,
    half*& nccl_rank1_buf,
    oo_buffer_t*& rank0_work_buf,
    oo_buffer_t*& rank1_work_buf,
    oo_node_t*& node0,
    oo_node_t*& node1,
    oo_group_t*& group,
    cudaStream_t& stream0,
    cudaStream_t& stream1,
    ncclComm_t* comms) {
    testing::destroy_nccl_comms(comms, 2);

    testing::cuda_free_on_device(dev0, rank0_src);
    testing::cuda_free_on_device(dev1, rank1_src);
    testing::cuda_free_on_device(dev0, nccl_rank0_buf);
    testing::cuda_free_on_device(dev1, nccl_rank1_buf);

    testing::destroy_oo_buffer(rank0_work_buf);
    testing::destroy_oo_buffer(rank1_work_buf);

    testing::destroy_oo_node(node0);
    testing::destroy_oo_node(node1);
    testing::destroy_oo_group(group);

    testing::destroy_stream_on_device(dev0, stream0);
    testing::destroy_stream_on_device(dev1, stream1);
}

json benchmark_one_collective_size(
    TestCollective collective,
    int64_t numel_arg,
    const std::vector<ModeSpec>& modes,
    const std::vector<int>& max_cta_constraints,
    bool include_nccl,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
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

    oo_buffer_t* rank0_work_buf = nullptr;
    oo_buffer_t* rank1_work_buf = nullptr;

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;
    half* nccl_rank0_buf = nullptr;
    half* nccl_rank1_buf = nullptr;

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    ncclComm_t comms[2] = {
        nullptr,
        nullptr,
    };

    json rows = json::array();

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
            oo_buffer_alloc(node0, bytes, &rank0_work_buf),
            "oo_buffer_alloc(rank0_work)");

        testing::check_oo(
            oo_buffer_alloc(node1, bytes, &rank1_work_buf),
            "oo_buffer_alloc(rank1_work)");

        half* rank0_work =
            reinterpret_cast<half*>(oo_buffer_ptr(rank0_work_buf));

        half* rank1_work =
            reinterpret_cast<half*>(oo_buffer_ptr(rank1_work_buf));

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

        testing::fill_two_rank_sources_fp16(
            rank0_src,
            rank1_src,
            numel_arg,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        if (include_nccl) {
            testing::cuda_malloc_half_on_device(
                node0_dev,
                &nccl_rank0_buf,
                bytes,
                "cudaMalloc(nccl_rank0_buf)");

            testing::cuda_malloc_half_on_device(
                node1_dev,
                &nccl_rank1_buf,
                bytes,
                "cudaMalloc(nccl_rank1_buf)");

            OOVERLAP_TEST_NCCL_CHECK(
                ncclCommInitAll(comms, 2, devices));
        }

        for (int max_ctas : max_cta_constraints) {
            const bool override_max_ctas =
                max_ctas > 0;

            const std::string max_ctas_value =
                override_max_ctas ? std::to_string(max_ctas) : std::string();

            ScopedEnvOverride cta_env(
                "OOVERLAP_MAX_CTAS",
                max_ctas_value,
                override_max_ctas);

            for (const ModeSpec& mode : modes) {
                testing::prepare_two_work_buffers(
                    rank0_src,
                    rank1_src,
                    rank0_work,
                    rank1_work,
                    bytes,
                    node0_dev,
                    node1_dev,
                    stream0,
                    stream1);

                run_public_tuned_iters(
                    group,
                    collective,
                    node0,
                    node1,
                    rank0_work_buf,
                    rank1_work_buf,
                    numel,
                    mode.mode,
                    node0_dev,
                    node1_dev,
                    stream0,
                    stream1,
                    warmup);

                testing::prepare_two_work_buffers(
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
                    elapsed_ms_public_tuned(
                        group,
                        collective,
                        node0,
                        node1,
                        rank0_work_buf,
                        rank1_work_buf,
                        numel,
                        mode.mode,
                        node0_dev,
                        node1_dev,
                        stream0,
                        stream1,
                        iters);

                testing::sync_two_streams(
                    node0_dev,
                    stream0,
                    node1_dev,
                    stream1,
                    "sync public tuned measured");

                verify_collective_result(
                    collective,
                    mode.name,
                    rank0_work,
                    rank1_work,
                    numel_arg,
                    node0_dev,
                    node1_dev);

                json row;
                row["id"] = {
                    {"kind", "ooverlap_public_tuned"},
                    {"collective", testing::collective_name(collective)},
                    {"numel", numel},
                    {"tuning_mode", mode.name},
                };

                if (override_max_ctas) {
                    row["id"]["max_ctas"] = max_ctas;
                    row["max_ctas"] = max_ctas;
                    row["max_ctas_source"] = "request";
                } else {
                    row["id"]["max_ctas"] = nullptr;
                    row["max_ctas"] = nullptr;
                    row["max_ctas_source"] = "ambient_env";
                }

                row["status"] = "ok";
                row["backend"] = "ooverlap_public";
                row["kernel"] = "public_tuned";
                row["tuning_mode"] = mode.name;

                add_common_metrics(
                    row,
                    collective,
                    numel,
                    bytes,
                    iters,
                    warmup,
                    node0_dev,
                    node1_dev,
                    total_ms);

                add_env_metadata(row);
                rows.push_back(std::move(row));
            }
        }

        if (include_nccl) {
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

            json row;
            row["id"] = {
                {"kind", "nccl_baseline"},
                {"collective", testing::collective_name(collective)},
                {"numel", numel},
            };
            row["status"] = "ok";
            row["backend"] = "nccl";
            row["kernel"] = "nccl";
            row["tuning_mode"] = nullptr;
            row["max_ctas"] = nullptr;
            row["max_ctas_source"] = nullptr;

            add_common_metrics(
                row,
                collective,
                numel,
                bytes,
                iters,
                warmup,
                node0_dev,
                node1_dev,
                nccl_total_ms);

            add_env_metadata(row);
            rows.push_back(std::move(row));
        }

        cleanup(
            node0_dev,
            node1_dev,
            rank0_src,
            rank1_src,
            nccl_rank0_buf,
            nccl_rank1_buf,
            rank0_work_buf,
            rank1_work_buf,
            node0,
            node1,
            group,
            stream0,
            stream1,
            comms);

        return rows;
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
            rank0_work_buf,
            rank1_work_buf,
            node0,
            node1,
            group,
            stream0,
            stream1,
            comms);

        throw;
    }
}

} // namespace

std::string benchmark_tma_efficiency_vs_best_2gpu_json(
    const std::string& request_json) {
    json out;

    try {
        const json request =
            json::parse(request_json);

        const std::vector<int64_t> numels =
            read_i64_list_required(request, "numels");

        const std::vector<std::string> collective_names =
            read_string_list(
                request,
                "collectives",
                {"allreduce", "reduce_scatter", "all_gather"});

        const std::vector<ModeSpec> modes =
            read_modes(request);

        const std::vector<int> max_cta_constraints =
            read_max_cta_constraints(request);

        const int iters =
            read_int_or(request, "iters", 100);

        const int warmup =
            read_int_or(request, "warmup", 20);

        const int dev0 =
            read_int_or(request, "dev0", 0);

        const int dev1 =
            read_int_or(request, "dev1", 1);

        const bool include_nccl =
            read_bool_or(request, "include_nccl", true);

        if (iters <= 0) {
            throw std::invalid_argument("iters must be > 0");
        }

        if (warmup < 0) {
            throw std::invalid_argument("warmup must be >= 0");
        }

        if (dev0 == dev1) {
            throw std::invalid_argument("dev0 and dev1 must differ");
        }

        json rows = json::array();

        for (const std::string& collective_name : collective_names) {
            const TestCollective collective =
                testing::parse_collective(collective_name);

            for (int64_t numel : numels) {
                json case_rows =
                    benchmark_one_collective_size(
                        collective,
                        numel,
                        modes,
                        max_cta_constraints,
                        include_nccl,
                        iters,
                        warmup,
                        dev0,
                        dev1);

                for (json& row : case_rows) {
                    rows.push_back(std::move(row));
                }
            }
        }

        out["ok"] = true;
        out["schema"] = "tma_efficiency_vs_best_2gpu_public_tuned_v1";
        out["request"] = request;
        out["raw_results"] = rows;
        out["results"] = rows;
        out["notes"] = {
            "This benchmark calls the public oo_*_tuned APIs.",
            "It does not manually sweep chunk_bytes, stage_depth, threads, or window_chunks.",
            "When ctas/max_ctas is provided, OOVERLAP_MAX_CTAS is temporarily set before each public tuned run.",
        };

        return out.dump(2);
    } catch (const std::exception& e) {
        out["ok"] = false;
        out["schema"] = "tma_efficiency_vs_best_2gpu_public_tuned_v1";
        out["error"] = e.what();
        return out.dump(2);
    } catch (...) {
        out["ok"] = false;
        out["schema"] = "tma_efficiency_vs_best_2gpu_public_tuned_v1";
        out["error"] = "unknown exception";
        return out.dump(2);
    }
}

} // namespace ooverlap
