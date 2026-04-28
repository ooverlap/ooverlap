#include "test/public_allreduce_benchmark_2gpu_sm90.h"

#include "comm/ooverlap_comm.h"
#include "comm/ooverlap_comm_internal.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define OOVERLAP_PUBLIC_BENCH_NCCL_CHECK(cmd)                                \
    do {                                                                      \
        ncclResult_t result__ = (cmd);                                        \
        if (result__ != ncclSuccess) {                                        \
            throw std::runtime_error(                                         \
                std::string("NCCL error: ") + ncclGetErrorString(result__));  \
        }                                                                     \
    } while (0)

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

oo_tuning_mode_t parse_tuning_mode(int mode) {
    if (mode == static_cast<int>(OO_TUNING_BEST_EFFICIENCY)) {
        return OO_TUNING_BEST_EFFICIENCY;
    }

    return OO_TUNING_BEST_PERFORMANCE;
}

const char* tuning_mode_name(oo_tuning_mode_t mode) {
    switch (mode) {
        case OO_TUNING_BEST_EFFICIENCY:
            return "best_efficiency";
        case OO_TUNING_BEST_PERFORMANCE:
        default:
            return "best_performance";
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

void launch_public_once(
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    check_oo(
        oo_allreduce_tuned(
            node0,
            rank0_buf,
            rank1_buf,
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            tuning_mode,
            stream0),
        "oo_allreduce_tuned(rank0)");

    check_oo(
        oo_allreduce_tuned(
            node1,
            rank1_buf,
            rank0_buf,
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            tuning_mode,
            stream1),
        "oo_allreduce_tuned(rank1)");
}

void run_public_iters(
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    for (int i = 0; i < iters; ++i) {
        launch_public_once(
            node0,
            node1,
            rank0_buf,
            rank1_buf,
            numel,
            tuning_mode,
            stream0,
            stream1);
    }
}

double elapsed_ms_public_allreduce(
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
    return elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int) {
            launch_public_once(
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
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    ncclComm_t* comms,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    OOVERLAP_PUBLIC_BENCH_NCCL_CHECK(ncclGroupStart());

    OOVERLAP_PUBLIC_BENCH_NCCL_CHECK(
        ncclAllReduce(
            rank0_buf,
            rank0_buf,
            numel,
            ncclFloat16,
            ncclSum,
            comms[0],
            stream0));

    OOVERLAP_PUBLIC_BENCH_NCCL_CHECK(
        ncclAllReduce(
            rank1_buf,
            rank1_buf,
            numel,
            ncclFloat16,
            ncclSum,
            comms[1],
            stream1));

    OOVERLAP_PUBLIC_BENCH_NCCL_CHECK(ncclGroupEnd());
}

void run_nccl_iters(
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    ncclComm_t* comms,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    for (int i = 0; i < iters; ++i) {
        launch_nccl_once(rank0_buf, rank1_buf, numel, comms, stream0, stream1);
    }
}

double elapsed_ms_nccl_allreduce(
    half* rank0_buf,
    half* rank1_buf,
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
            launch_nccl_once(rank0_buf, rank1_buf, numel, comms, stream0, stream1);
        });
}

std::vector<int64_t> make_byte_points(
    int64_t min_bytes,
    int64_t max_bytes,
    int points) {
    if (min_bytes <= 0 || max_bytes <= 0 || max_bytes < min_bytes || points <= 0) {
        throw std::invalid_argument("invalid byte sweep range");
    }

    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(points));

    if (points == 1) {
        out.push_back(min_bytes);
        return out;
    }

    const double log_min = std::log(static_cast<double>(min_bytes));
    const double log_max = std::log(static_cast<double>(max_bytes));

    int64_t last = -1;

    for (int i = 0; i < points; ++i) {
        const double t =
            static_cast<double>(i) / static_cast<double>(points - 1);

        int64_t bytes =
            static_cast<int64_t>(
                std::llround(std::exp(log_min + t * (log_max - log_min))));

        /*
         * We benchmark float16, so force bytes to a positive even count.
         */
        bytes = std::max<int64_t>(2, bytes);
        bytes = (bytes / 2) * 2;

        /*
         * Avoid duplicate adjacent points after rounding.
         */
        if (bytes <= last) {
            bytes = last + 2;
        }

        if (bytes > max_bytes) {
            bytes = (max_bytes / 2) * 2;
        }

        if (bytes <= 0) {
            bytes = 2;
        }

        if (!out.empty() && out.back() == bytes) {
            continue;
        }

        out.push_back(bytes);
        last = bytes;
    }

    if (out.empty() || out.front() != ((min_bytes / 2) * 2)) {
        int64_t first = (min_bytes / 2) * 2;
        if (first <= 0) {
            first = 2;
        }
        out.insert(out.begin(), first);
    }

    const int64_t last_target = (max_bytes / 2) * 2;
    if (out.back() != last_target) {
        out.push_back(last_target);
    }

    return out;
}

void append_row(
    std::ostringstream& rows,
    const char* backend,
    int64_t bytes_per_rank,
    int64_t numel,
    int points,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    oo_tuning_mode_t tuning_mode,
    double total_ms,
    double nccl_avg_ms_for_speedup) {
    const double avg_ms = total_ms / static_cast<double>(iters);

    /*
     * This is an allreduce benchmark. Report both:
     *   per-rank payload bandwidth: bytes_per_rank / time
     *   aggregate 2-GPU payload bandwidth: 2 * bytes_per_rank / time
     *
     * Units: GB/s using decimal 1e9.
     */
    const double gbps_per_rank =
        (avg_ms > 0.0)
            ? (static_cast<double>(bytes_per_rank) / (avg_ms * 1.0e-3) / 1.0e9)
            : 0.0;

    const double gbps_aggregate =
        (avg_ms > 0.0)
            ? (2.0 * static_cast<double>(bytes_per_rank) / (avg_ms * 1.0e-3) / 1.0e9)
            : 0.0;

    const double speedup_vs_nccl =
        (avg_ms > 0.0 && nccl_avg_ms_for_speedup > 0.0)
            ? (nccl_avg_ms_for_speedup / avg_ms)
            : ((std::string(backend) == "nccl") ? 1.0 : 0.0);

    rows << std::setprecision(12);
    rows << "{";
    rows << "\"backend\":\"" << backend << "\"";
    rows << ",\"bytes_per_rank\":" << bytes_per_rank;
    rows << ",\"numel\":" << numel;
    rows << ",\"points\":" << points;
    rows << ",\"iters\":" << iters;
    rows << ",\"warmup\":" << warmup;
    rows << ",\"dev0\":" << dev0;
    rows << ",\"dev1\":" << dev1;
    rows << ",\"tuning_mode\":\"" << tuning_mode_name(tuning_mode) << "\"";
    rows << ",\"total_ms\":" << total_ms;
    rows << ",\"avg_ms\":" << avg_ms;
    rows << ",\"gbps_per_rank\":" << gbps_per_rank;
    rows << ",\"gbps_aggregate_2gpu\":" << gbps_aggregate;
    rows << ",\"speedup_vs_nccl\":" << speedup_vs_nccl;
    rows << "}\n";
}

} // namespace

std::string benchmark_public_allreduce_2gpu_sm90(
    int64_t min_bytes,
    int64_t max_bytes,
    int points,
    int iters,
    int warmup,
    int tuning_mode,
    int dev0,
    int dev1) {
    if (iters <= 0 || warmup < 0) {
        throw std::invalid_argument("iters must be > 0 and warmup must be >= 0");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument("dev0 and dev1 must differ");
    }

    const oo_tuning_mode_t mode = parse_tuning_mode(tuning_mode);
    const std::vector<int64_t> byte_points =
        make_byte_points(min_bytes, max_bytes, points);

    std::ostringstream rows;

    for (int64_t bytes_per_rank : byte_points) {
        const int64_t numel = bytes_per_rank / static_cast<int64_t>(sizeof(half));
        const size_t bytes = static_cast<size_t>(bytes_per_rank);

        oo_group_t* group = nullptr;
        oo_node_t* node0 = nullptr;
        oo_node_t* node1 = nullptr;
        oo_buffer_t* oo_rank0_buf = nullptr;
        oo_buffer_t* oo_rank1_buf = nullptr;

        half* nccl_rank0_buf = nullptr;
        half* nccl_rank1_buf = nullptr;

        cudaStream_t stream0 = nullptr;
        cudaStream_t stream1 = nullptr;

        ncclComm_t comms[2] = {nullptr, nullptr};

        try {
            int devices[2] = {dev0, dev1};

            check_oo(oo_group_create(devices, 2, &group), "oo_group_create");
            check_oo(oo_node_create(group, 0, &node0), "oo_node_create(rank0)");
            check_oo(oo_node_create(group, 1, &node1), "oo_node_create(rank1)");

            const int node0_dev = oo_node_device(node0);
            const int node1_dev = oo_node_device(node1);

            stream0 = system::runtime::create_stream_on_device(node0_dev);
            stream1 = system::runtime::create_stream_on_device(node1_dev);

            check_oo(
                oo_buffer_alloc(node0, bytes, &oo_rank0_buf),
                "oo_buffer_alloc(rank0)");
            check_oo(
                oo_buffer_alloc(node1, bytes, &oo_rank1_buf),
                "oo_buffer_alloc(rank1)");

            half* oo_rank0_ptr =
                reinterpret_cast<half*>(oo_buffer_ptr(oo_rank0_buf));
            half* oo_rank1_ptr =
                reinterpret_cast<half*>(oo_buffer_ptr(oo_rank1_buf));

            system::runtime::set_device(node0_dev);
            system::runtime::check_cuda(
                cudaMalloc(&nccl_rank0_buf, bytes),
                "cudaMalloc(nccl_rank0_buf)");

            system::runtime::set_device(node1_dev);
            system::runtime::check_cuda(
                cudaMalloc(&nccl_rank1_buf, bytes),
                "cudaMalloc(nccl_rank1_buf)");

            int nccl_devices[2] = {node0_dev, node1_dev};
            OOVERLAP_PUBLIC_BENCH_NCCL_CHECK(ncclCommInitAll(comms, 2, nccl_devices));

            fill_inputs(
                oo_rank0_ptr,
                oo_rank1_ptr,
                numel,
                node0_dev,
                node1_dev,
                stream0,
                stream1);

            fill_inputs(
                nccl_rank0_buf,
                nccl_rank1_buf,
                numel,
                node0_dev,
                node1_dev,
                stream0,
                stream1);

            run_public_iters(
                node0,
                node1,
                oo_rank0_buf,
                oo_rank1_buf,
                static_cast<size_t>(numel),
                mode,
                stream0,
                stream1,
                warmup);

            sync_two_streams(
                node0_dev,
                stream0,
                node1_dev,
                stream1,
                "sync ooverlap warmup");

            const double ooverlap_total_ms =
                elapsed_ms_public_allreduce(
                    node0,
                    node1,
                    oo_rank0_buf,
                    oo_rank1_buf,
                    static_cast<size_t>(numel),
                    mode,
                    node0_dev,
                    node1_dev,
                    stream0,
                    stream1,
                    iters);

            run_nccl_iters(
                nccl_rank0_buf,
                nccl_rank1_buf,
                static_cast<size_t>(numel),
                comms,
                stream0,
                stream1,
                warmup);

            sync_two_streams(
                node0_dev,
                stream0,
                node1_dev,
                stream1,
                "sync nccl warmup");

            const double nccl_total_ms =
                elapsed_ms_nccl_allreduce(
                    nccl_rank0_buf,
                    nccl_rank1_buf,
                    static_cast<size_t>(numel),
                    node0_dev,
                    node1_dev,
                    stream0,
                    stream1,
                    comms,
                    iters);

            const double nccl_avg_ms =
                nccl_total_ms / static_cast<double>(iters);

            append_row(
                rows,
                "ooverlap",
                bytes_per_rank,
                numel,
                points,
                iters,
                warmup,
                node0_dev,
                node1_dev,
                mode,
                ooverlap_total_ms,
                nccl_avg_ms);

            append_row(
                rows,
                "nccl",
                bytes_per_rank,
                numel,
                points,
                iters,
                warmup,
                node0_dev,
                node1_dev,
                mode,
                nccl_total_ms,
                nccl_avg_ms);

            sync_two_streams(
                node0_dev,
                stream0,
                node1_dev,
                stream1,
                "sync benchmark cleanup");

            if (comms[0] != nullptr) {
                ncclCommDestroy(comms[0]);
                comms[0] = nullptr;
            }

            if (comms[1] != nullptr) {
                ncclCommDestroy(comms[1]);
                comms[1] = nullptr;
            }

            system::runtime::set_device(node0_dev);
            cudaFree(nccl_rank0_buf);
            nccl_rank0_buf = nullptr;

            system::runtime::set_device(node1_dev);
            cudaFree(nccl_rank1_buf);
            nccl_rank1_buf = nullptr;

            oo_buffer_destroy(oo_rank0_buf);
            oo_rank0_buf = nullptr;
            oo_buffer_destroy(oo_rank1_buf);
            oo_rank1_buf = nullptr;

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
        } catch (...) {
            const int node0_dev =
                (node0 != nullptr) ? oo_node_device(node0) : dev0;
            const int node1_dev =
                (node1 != nullptr) ? oo_node_device(node1) : dev1;

            if (comms[0] != nullptr) {
                ncclCommDestroy(comms[0]);
            }
            if (comms[1] != nullptr) {
                ncclCommDestroy(comms[1]);
            }

            if (nccl_rank0_buf != nullptr) {
                system::runtime::set_device(node0_dev);
                cudaFree(nccl_rank0_buf);
            }
            if (nccl_rank1_buf != nullptr) {
                system::runtime::set_device(node1_dev);
                cudaFree(nccl_rank1_buf);
            }

            if (oo_rank0_buf != nullptr) {
                oo_buffer_destroy(oo_rank0_buf);
            }
            if (oo_rank1_buf != nullptr) {
                oo_buffer_destroy(oo_rank1_buf);
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

    return rows.str();
}

} // namespace ooverlap
