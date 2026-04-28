#include "test/tma_allreduce_sweep_2gpu_sm90.h"

#include "comm/launch_config.h"
#include "comm/ooverlap_comm.h"
#include "comm/ooverlap_comm_internal.h"
#include "comm/params.h"
#include "comm/tma_two_gpu_peer_allreduce_fast_gmem_sm90.h"
#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define OOVERLAP_SWEEP_NCCL_CHECK(cmd)                                      \
    do {                                                                    \
        ncclResult_t result__ = (cmd);                                      \
        if (result__ != ncclSuccess) {                                      \
            throw std::runtime_error(                                       \
                std::string("NCCL error: ") + ncclGetErrorString(result__));\
        }                                                                   \
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

const char* kernel_kind_name(SweepKernelKind kind) {
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

SweepKernelKind parse_kernel_kind(const std::string& name) {
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

    throw std::invalid_argument(
        "benchmark_tma_two_gpu_allreduce_sweep_sm90: unknown kernel kind: " +
        name);
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;

    for (char c : s) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << c;
                break;
        }
    }

    return out.str();
}

std::string getenv_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

void append_json_string_or_null(
    std::ostringstream& out,
    const char* key,
    const std::string& value) {
    out << ",\"" << key << "\":";
    if (value.empty()) {
        out << "null";
    } else {
        out << "\"" << json_escape(value) << "\"";
    }
}

void append_nccl_env_metadata(std::ostringstream& out) {
    append_json_string_or_null(
        out,
        "nccl_max_ctas_env",
        getenv_or_empty("NCCL_MAX_CTAS"));
    append_json_string_or_null(
        out,
        "nccl_min_ctas_env",
        getenv_or_empty("NCCL_MIN_CTAS"));
    append_json_string_or_null(
        out,
        "nccl_algo_env",
        getenv_or_empty("NCCL_ALGO"));
    append_json_string_or_null(
        out,
        "nccl_proto_env",
        getenv_or_empty("NCCL_PROTO"));
}

void append_variant_null_metadata(std::ostringstream& out) {
    out << ",\"chunk_bytes\":null";
    out << ",\"stage_depth\":null";
    out << ",\"stage_gap\":null";

    out << ",\"compile_chunk_bytes\":null";
    out << ",\"compile_reduce_stage_depth\":null";
    out << ",\"compile_reduce_stage_gap\":null";
    out << ",\"compile_copy_stage_depth\":null";
    out << ",\"compile_copy_stage_gap\":null";
    out << ",\"compile_fast_copy_unroll\":null";
}

void append_variant_metadata(
    std::ostringstream& out,
    comm::LaunchConfig config) {
    const int stage_gap = config.stage_depth / 2;

    out << ",\"chunk_bytes\":" << config.chunk_bytes;
    out << ",\"stage_depth\":" << config.stage_depth;
    out << ",\"stage_gap\":" << stage_gap;

    /*
     * Keep the old compile_* names too because the policy builder already
     * consumes them. They now describe the selected precompiled variant.
     */
    out << ",\"compile_chunk_bytes\":" << config.chunk_bytes;
    out << ",\"compile_reduce_stage_depth\":" << config.stage_depth;
    out << ",\"compile_reduce_stage_gap\":" << stage_gap;
    out << ",\"compile_copy_stage_depth\":" << config.stage_depth;
    out << ",\"compile_copy_stage_gap\":" << stage_gap;
    out << ",\"compile_fast_copy_unroll\":"
        << TMA_TWO_GPU_PEER_FAST_COPY_UNROLL;
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

void reset_ready_signals(oo_group_t* group) {
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
    cudaError_t err0 = cudaSuccess;
    cudaError_t err1 = cudaSuccess;

    if (kernel == SweepKernelKind::kTmaCopy) {
        err0 = enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
            rank0_work,
            rank0_work,
            rank1_work,
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            0,
            dev0,
            dev1,
            stream0,
            rank0_ready,
            rank1_ready,
            collective_epoch,
            launch_config);

        err1 = enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
            rank1_work,
            rank1_work,
            rank0_work,
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            1,
            dev0,
            dev1,
            stream1,
            rank1_ready,
            rank0_ready,
            collective_epoch,
            launch_config);
    } else if (kernel == SweepKernelKind::kSeqFastGmem) {
        err0 = enqueue_tma_two_gpu_peer_allreduce_rank_seq_fastcopy_sm90(
            rank0_work,
            rank0_work,
            rank1_work,
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            0,
            dev0,
            dev1,
            stream0,
            rank0_ready,
            rank1_ready,
            collective_epoch,
            launch_config);

        err1 = enqueue_tma_two_gpu_peer_allreduce_rank_seq_fastcopy_sm90(
            rank1_work,
            rank1_work,
            rank0_work,
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            1,
            dev0,
            dev1,
            stream1,
            rank1_ready,
            rank0_ready,
            collective_epoch,
            launch_config);
    } else if (kernel == SweepKernelKind::kOverlapFastGmem) {
        err0 = enqueue_tma_two_gpu_peer_allreduce_rank_overlap_fastcopy_sm90(
            rank0_work,
            rank0_work,
            rank1_work,
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            0,
            dev0,
            dev1,
            stream0,
            rank0_ready,
            rank1_ready,
            collective_epoch,
            launch_config);

        err1 = enqueue_tma_two_gpu_peer_allreduce_rank_overlap_fastcopy_sm90(
            rank1_work,
            rank1_work,
            rank0_work,
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            1,
            dev0,
            dev1,
            stream1,
            rank1_ready,
            rank0_ready,
            collective_epoch,
            launch_config);
    } else {
        throw std::invalid_argument("launch_candidate_once: unknown kernel");
    }

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
            const int collective_epoch = epoch++;

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
                collective_epoch,
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

void append_nccl_row(
    std::ostringstream& rows,
    int64_t numel,
    size_t bytes,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    double nccl_total_ms) {
    const double avg_ms = nccl_total_ms / static_cast<double>(iters);
    const double gbps_per_rank =
        (avg_ms > 0.0)
            ? (static_cast<double>(bytes) / avg_ms / 1.0e6)
            : 0.0;
    const double gbps_aggregate =
        (avg_ms > 0.0)
            ? (2.0 * static_cast<double>(bytes) / avg_ms / 1.0e6)
            : 0.0;

    rows << std::setprecision(12);
    rows << "{";
    rows << "\"kernel\":\"nccl\"";
    rows << ",\"numel\":" << numel;
    rows << ",\"bytes_per_rank\":" << bytes;
    rows << ",\"iters\":" << iters;
    rows << ",\"warmup\":" << warmup;
    rows << ",\"dev0\":" << dev0;
    rows << ",\"dev1\":" << dev1;
    rows << ",\"threads\":null";
    rows << ",\"max_ctas\":null";
    rows << ",\"window_chunks\":null";
    append_variant_null_metadata(rows);
    rows << ",\"total_ms\":" << nccl_total_ms;
    rows << ",\"avg_ms\":" << avg_ms;
    rows << ",\"nccl_avg_ms\":" << avg_ms;
    rows << ",\"speedup_vs_nccl\":1.0";
    rows << ",\"effective_gbps_per_rank\":" << gbps_per_rank;
    rows << ",\"effective_gbps_aggregate_2gpu\":" << gbps_aggregate;
    append_nccl_env_metadata(rows);
    rows << "}\n";
}

void append_candidate_row(
    std::ostringstream& rows,
    int64_t numel,
    size_t bytes,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    SweepKernelKind kernel,
    comm::LaunchConfig config,
    double total_ms,
    double nccl_total_ms) {
    const double avg_ms = total_ms / static_cast<double>(iters);
    const double nccl_avg_ms = nccl_total_ms / static_cast<double>(iters);
    const double speedup =
        (avg_ms > 0.0) ? (nccl_avg_ms / avg_ms) : 0.0;
    const double gbps_per_rank =
        (avg_ms > 0.0)
            ? (static_cast<double>(bytes) / avg_ms / 1.0e6)
            : 0.0;
    const double gbps_aggregate =
        (avg_ms > 0.0)
            ? (2.0 * static_cast<double>(bytes) / avg_ms / 1.0e6)
            : 0.0;

    rows << std::setprecision(12);
    rows << "{";
    rows << "\"kernel\":\"" << kernel_kind_name(kernel) << "\"";
    rows << ",\"numel\":" << numel;
    rows << ",\"bytes_per_rank\":" << bytes;
    rows << ",\"iters\":" << iters;
    rows << ",\"warmup\":" << warmup;
    rows << ",\"dev0\":" << dev0;
    rows << ",\"dev1\":" << dev1;
    rows << ",\"threads\":" << config.threads;
    rows << ",\"max_ctas\":" << config.max_ctas;
    rows << ",\"window_chunks\":" << config.window_chunks;
    append_variant_metadata(rows, config);
    rows << ",\"total_ms\":" << total_ms;
    rows << ",\"avg_ms\":" << avg_ms;
    rows << ",\"nccl_avg_ms\":" << nccl_avg_ms;
    rows << ",\"speedup_vs_nccl\":" << speedup;
    rows << ",\"effective_gbps_per_rank\":" << gbps_per_rank;
    rows << ",\"effective_gbps_aggregate_2gpu\":" << gbps_aggregate;
    append_nccl_env_metadata(rows);
    rows << "}\n";
}

} // namespace

std::string benchmark_tma_two_gpu_allreduce_sweep_sm90(
    const std::vector<int64_t>& numels,
    const std::vector<std::string>& kernels,
    const std::vector<int>& threads,
    const std::vector<int>& max_ctas,
    const std::vector<int>& window_chunks,
    const std::vector<int>& chunk_bytes,
    const std::vector<int>& stage_depths,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    if (numels.empty() ||
        kernels.empty() ||
        threads.empty() ||
        max_ctas.empty() ||
        window_chunks.empty() ||
        chunk_bytes.empty() ||
        stage_depths.empty()) {
        throw std::invalid_argument(
            "benchmark_tma_two_gpu_allreduce_sweep_sm90: sweep lists must be non-empty");
    }

    if (chunk_bytes.size() != stage_depths.size()) {
        throw std::invalid_argument(
            "benchmark_tma_two_gpu_allreduce_sweep_sm90: chunk_bytes and stage_depths must have the same length");
    }

    if (iters <= 0 || warmup < 0) {
        throw std::invalid_argument(
            "benchmark_tma_two_gpu_allreduce_sweep_sm90: invalid iters/warmup");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument(
            "benchmark_tma_two_gpu_allreduce_sweep_sm90: dev0 and dev1 must differ");
    }

    std::vector<SweepKernelKind> parsed_kernels;
    parsed_kernels.reserve(kernels.size());

    for (const std::string& name : kernels) {
        parsed_kernels.push_back(parse_kernel_kind(name));
    }

    std::ostringstream rows;

    for (int64_t numel : numels) {
        if (numel <= 0) {
            throw std::invalid_argument(
                "benchmark_tma_two_gpu_allreduce_sweep_sm90: numel must be > 0");
        }

        oo_group_t* group = nullptr;
        oo_node_t* node0 = nullptr;
        oo_node_t* node1 = nullptr;
        oo_buffer_t* rank0_buf = nullptr;
        oo_buffer_t* rank1_buf = nullptr;

        half* rank0_src = nullptr;
        half* rank1_src = nullptr;
        half* nccl_rank0_out = nullptr;
        half* nccl_rank1_out = nullptr;

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
            const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

            stream0 = system::runtime::create_stream_on_device(node0_dev);
            stream1 = system::runtime::create_stream_on_device(node1_dev);

            system::runtime::set_device(node0_dev);
            system::runtime::check_cuda(
                cudaMalloc(&rank0_src, bytes),
                "cudaMalloc(rank0_src)");
            system::runtime::check_cuda(
                cudaMalloc(&nccl_rank0_out, bytes),
                "cudaMalloc(nccl_rank0_out)");

            system::runtime::set_device(node1_dev);
            system::runtime::check_cuda(
                cudaMalloc(&rank1_src, bytes),
                "cudaMalloc(rank1_src)");
            system::runtime::check_cuda(
                cudaMalloc(&nccl_rank1_out, bytes),
                "cudaMalloc(nccl_rank1_out)");

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
                numel,
                node0_dev,
                node1_dev,
                stream0,
                stream1);

            int nccl_devices[2] = {node0_dev, node1_dev};
            OOVERLAP_SWEEP_NCCL_CHECK(ncclCommInitAll(comms, 2, nccl_devices));

            run_nccl_iters(
                rank0_src,
                rank1_src,
                nccl_rank0_out,
                nccl_rank1_out,
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
                elapsed_ms_nccl(
                    rank0_src,
                    rank1_src,
                    nccl_rank0_out,
                    nccl_rank1_out,
                    static_cast<size_t>(numel),
                    node0_dev,
                    node1_dev,
                    stream0,
                    stream1,
                    comms,
                    iters);

            append_nccl_row(
                rows,
                numel,
                bytes,
                iters,
                warmup,
                node0_dev,
                node1_dev,
                nccl_total_ms);

            for (SweepKernelKind kernel : parsed_kernels) {
                for (int thread_count : threads) {
                    for (int max_cta_count : max_ctas) {
                        for (int window_chunk_count : window_chunks) {
                            for (size_t variant_idx = 0;
                                 variant_idx < chunk_bytes.size();
                                 ++variant_idx) {
                                comm::LaunchConfig config{};
                                config.threads = thread_count;
                                config.max_ctas = max_cta_count;
                                config.window_chunks = window_chunk_count;
                                config.chunk_bytes = chunk_bytes[variant_idx];
                                config.stage_depth = stage_depths[variant_idx];

                                const bool config_ok =
                                    (kernel == SweepKernelKind::kOverlapFastGmem)
                                        ? comm::launch_config_valid_for_overlap(config)
                                        : comm::launch_config_valid(config);

                                if (!config_ok) {
                                    continue;
                                }

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
                                    static_cast<size_t>(numel),
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
                                    "sync candidate warmup");

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
                                        static_cast<size_t>(numel),
                                        node0_dev,
                                        node1_dev,
                                        stream0,
                                        stream1,
                                        iters,
                                        config);

                                append_candidate_row(
                                    rows,
                                    numel,
                                    bytes,
                                    iters,
                                    warmup,
                                    node0_dev,
                                    node1_dev,
                                    kernel,
                                    config,
                                    total_ms,
                                    nccl_total_ms);
                            }
                        }
                    }
                }
            }

            sync_two_streams(
                node0_dev,
                stream0,
                node1_dev,
                stream1,
                "sync sweep cleanup");

            if (comms[0] != nullptr) {
                ncclCommDestroy(comms[0]);
                comms[0] = nullptr;
            }

            if (comms[1] != nullptr) {
                ncclCommDestroy(comms[1]);
                comms[1] = nullptr;
            }

            oo_buffer_destroy(rank0_buf);
            rank0_buf = nullptr;
            oo_buffer_destroy(rank1_buf);
            rank1_buf = nullptr;

            system::runtime::set_device(node0_dev);
            cudaFree(rank0_src);
            rank0_src = nullptr;
            cudaFree(nccl_rank0_out);
            nccl_rank0_out = nullptr;

            system::runtime::set_device(node1_dev);
            cudaFree(rank1_src);
            rank1_src = nullptr;
            cudaFree(nccl_rank1_out);
            nccl_rank1_out = nullptr;

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

            if (rank0_buf != nullptr) {
                oo_buffer_destroy(rank0_buf);
            }
            if (rank1_buf != nullptr) {
                oo_buffer_destroy(rank1_buf);
            }

            if (rank0_src != nullptr) {
                system::runtime::set_device(node0_dev);
                cudaFree(rank0_src);
            }
            if (nccl_rank0_out != nullptr) {
                system::runtime::set_device(node0_dev);
                cudaFree(nccl_rank0_out);
            }
            if (rank1_src != nullptr) {
                system::runtime::set_device(node1_dev);
                cudaFree(rank1_src);
            }
            if (nccl_rank1_out != nullptr) {
                system::runtime::set_device(node1_dev);
                cudaFree(nccl_rank1_out);
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
