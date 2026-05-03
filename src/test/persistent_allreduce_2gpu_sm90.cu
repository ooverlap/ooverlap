#include "test/persistent_allreduce_2gpu_sm90.h"

#include "comm/launch_config.h"
#include "comm/ooverlap_comm.h"
#include "comm/ooverlap_comm_internal.h"
#include "comm/tma_multi_gpu_all_gather_sm90.h"
#include "comm/tma_multi_gpu_allreduce_sm90.h"
#include "comm/tma_multi_gpu_reduce_scatter_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define OOVERLAP_PERSIST_NCCL_CHECK(cmd)                                      \
    do {                                                                      \
        ncclResult_t result__ = (cmd);                                        \
        if (result__ != ncclSuccess) {                                        \
            throw std::runtime_error(                                         \
                std::string("NCCL error: ") + ncclGetErrorString(result__));  \
        }                                                                     \
    } while (0)

#ifndef OOVERLAP_BENCH_VERIFY_RESULTS
#define OOVERLAP_BENCH_VERIFY_RESULTS 0
#endif

namespace ooverlap {
namespace {

enum class BenchCollective {
    AllReduce,
    ReduceScatter,
    AllGather,
};

BenchCollective parse_collective(const std::string& value) {
    if (value == "allreduce" ||
        value == "all_reduce" ||
        value == "all-reduce" ||
        value == "ar") {
        return BenchCollective::AllReduce;
    }

    if (value == "reduce_scatter" ||
        value == "reducescatter" ||
        value == "reduce-scatter" ||
        value == "rs") {
        return BenchCollective::ReduceScatter;
    }

    if (value == "all_gather" ||
        value == "allgather" ||
        value == "all-gather" ||
        value == "ag") {
        return BenchCollective::AllGather;
    }

    throw std::invalid_argument(
        "unknown collective '" + value +
        "'; expected allreduce, reduce_scatter, or all_gather");
}

const char* collective_name(BenchCollective collective) {
    switch (collective) {
        case BenchCollective::AllReduce:
            return "allreduce";
        case BenchCollective::ReduceScatter:
            return "reduce_scatter";
        case BenchCollective::AllGather:
            return "all_gather";
        default:
            return "unknown";
    }
}

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

void copy_two_buffers_async(
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_dst,
    half* rank1_dst,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            rank0_dst,
            rank0_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream0),
        "cudaMemcpyAsync(rank0_src -> rank0_dst)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            rank1_dst,
            rank1_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream1),
        "cudaMemcpyAsync(rank1_src -> rank1_dst)");
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
    copy_two_buffers_async(
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

comm::LaunchConfig config_for_kind(
    comm::AllreducePlanKind kind) {
    comm::LaunchConfig config{};
    config.plan_kind = kind;
    return config;
}

size_t rank_partition_begin(
    size_t count,
    int rank,
    int world_size) {
    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = count / world;
    const size_t rem = count % world;
    return r * base + ((r < rem) ? r : rem);
}

size_t rank_partition_count(
    size_t count,
    int rank,
    int world_size) {
    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = count / world;
    const size_t rem = count % world;
    return base + ((r < rem) ? 1 : 0);
}

void validate_collective_size(
    BenchCollective collective,
    size_t numel) {
    if (collective == BenchCollective::AllReduce) {
        return;
    }

    /*
     * The ooverlap primitive can partition uneven counts, but NCCL all-gather
     * and reduce-scatter require equal per-rank counts. This benchmark compares
     * against NCCL, so keep the 2-GPU shard size equal.
     */
    if ((numel % 2) != 0) {
        throw std::invalid_argument(
            std::string(collective_name(collective)) +
            " benchmark requires numel divisible by 2 for NCCL comparison");
    }
}

void launch_ooverlap_once(
    BenchCollective collective,
    comm::AllreducePlanKind kind,
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
    comm::LaunchConfig config = config_for_kind(kind);

    void* rank0_peers[] = {rank0_peer};
    void* rank1_peers[] = {rank1_peer};

    const int* rank0_peer_ready[] = {rank1_ready};
    const int* rank1_peer_ready[] = {rank0_ready};

    if (collective == BenchCollective::AllReduce) {
        system::runtime::check_cuda(
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
            "enqueue ooverlap allreduce rank0");

        system::runtime::check_cuda(
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
            "enqueue ooverlap allreduce rank1");

        return;
    }

    if (collective == BenchCollective::ReduceScatter) {
        system::runtime::check_cuda(
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
            "enqueue ooverlap reduce_scatter rank0");

        system::runtime::check_cuda(
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
            "enqueue ooverlap reduce_scatter rank1");

        return;
    }

    if (collective == BenchCollective::AllGather) {
        system::runtime::check_cuda(
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
            "enqueue ooverlap all_gather rank0");

        system::runtime::check_cuda(
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
            "enqueue ooverlap all_gather rank1");

        return;
    }

    throw std::invalid_argument("unsupported ooverlap collective");
}

void run_ooverlap_iters(
    BenchCollective collective,
    comm::AllreducePlanKind kind,
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

    reset_ready_signals(group);

    int* rank0_ready = ready_signal_ptr(group, 0);
    int* rank1_ready = ready_signal_ptr(group, 1);

    for (int i = 0; i < iters; ++i) {
        launch_ooverlap_once(
            collective,
            kind,
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

    sync_two_streams(dev0, stream0, dev1, stream1, "sync ooverlap warmup");
}

double elapsed_ms_ooverlap(
    BenchCollective collective,
    comm::AllreducePlanKind kind,
    oo_group_t* group,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
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
            launch_ooverlap_once(
                collective,
                kind,
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
    BenchCollective collective,
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms) {
    const size_t shard0_begin = rank_partition_begin(numel, 0, 2);
    const size_t shard1_begin = rank_partition_begin(numel, 1, 2);
    const size_t shard_count = rank_partition_count(numel, 0, 2);

    OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());

    if (collective == BenchCollective::AllReduce) {
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank0_buf,
                rank0_buf,
                numel,
                ncclFloat16,
                ncclSum,
                comms[0],
                stream0));

        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank1_buf,
                rank1_buf,
                numel,
                ncclFloat16,
                ncclSum,
                comms[1],
                stream1));
    } else if (collective == BenchCollective::ReduceScatter) {
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclReduceScatter(
                rank0_buf,
                rank0_buf + shard0_begin,
                shard_count,
                ncclFloat16,
                ncclSum,
                comms[0],
                stream0));

        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclReduceScatter(
                rank1_buf,
                rank1_buf + shard1_begin,
                shard_count,
                ncclFloat16,
                ncclSum,
                comms[1],
                stream1));
    } else if (collective == BenchCollective::AllGather) {
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllGather(
                rank0_buf + shard0_begin,
                rank0_buf,
                shard_count,
                ncclFloat16,
                comms[0],
                stream0));

        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllGather(
                rank1_buf + shard1_begin,
                rank1_buf,
                shard_count,
                ncclFloat16,
                comms[1],
                stream1));
    } else {
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
        throw std::invalid_argument("unsupported NCCL collective");
    }

    OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
}

void run_nccl_iters(
    BenchCollective collective,
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
    BenchCollective collective,
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

std::vector<float> reference_rank0_fp16(int64_t numel) {
    return testing::host_reference_pattern_fp16(numel, 0.25f, 1.0f);
}

std::vector<float> reference_rank1_fp16(int64_t numel) {
    return testing::host_reference_pattern_fp16(numel, 0.50f, 2.0f);
}

std::vector<float> reference_sum_fp16(int64_t numel) {
    auto ref0 = reference_rank0_fp16(numel);
    auto ref1 = reference_rank1_fp16(numel);

    std::vector<float> ref(static_cast<size_t>(numel));

    for (int64_t i = 0; i < numel; ++i) {
        float acc = ref0[static_cast<size_t>(i)];
        acc = testing::round_to_half(acc + ref1[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = acc;
    }

    return ref;
}

void verify_collective_result(
    BenchCollective collective,
    const char* label,
    half* rank0,
    half* rank1,
    int64_t numel,
    int dev0,
    int dev1) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    auto got0 = testing::copy_half_device_to_host_float(rank0, numel, dev0);
    auto got1 = testing::copy_half_device_to_host_float(rank1, numel, dev1);

    if (collective == BenchCollective::AllReduce) {
        auto ref = reference_sum_fp16(numel);

        testing::expect_allclose(
            got0,
            ref,
            (std::string(label) + " rank0 allreduce").c_str());

        testing::expect_allclose(
            got1,
            ref,
            (std::string(label) + " rank1 allreduce").c_str());

        return;
    }

    if (collective == BenchCollective::ReduceScatter) {
        auto ref = reference_sum_fp16(numel);

        const size_t begin0 = rank_partition_begin(
            static_cast<size_t>(numel),
            0,
            2);
        const size_t count0 = rank_partition_count(
            static_cast<size_t>(numel),
            0,
            2);

        const size_t begin1 = rank_partition_begin(
            static_cast<size_t>(numel),
            1,
            2);
        const size_t count1 = rank_partition_count(
            static_cast<size_t>(numel),
            1,
            2);

        for (size_t i = 0; i < count0; ++i) {
            const size_t idx = begin0 + i;
            testing::expect_allclose(
                std::vector<float>{got0[idx]},
                std::vector<float>{ref[idx]},
                (std::string(label) + " rank0 reduce_scatter").c_str());
        }

        for (size_t i = 0; i < count1; ++i) {
            const size_t idx = begin1 + i;
            testing::expect_allclose(
                std::vector<float>{got1[idx]},
                std::vector<float>{ref[idx]},
                (std::string(label) + " rank1 reduce_scatter").c_str());
        }

        return;
    }

    if (collective == BenchCollective::AllGather) {
        auto ref0 = reference_rank0_fp16(numel);
        auto ref1 = reference_rank1_fp16(numel);

        std::vector<float> ref(static_cast<size_t>(numel));

        const size_t begin0 = rank_partition_begin(
            static_cast<size_t>(numel),
            0,
            2);
        const size_t count0 = rank_partition_count(
            static_cast<size_t>(numel),
            0,
            2);

        const size_t begin1 = rank_partition_begin(
            static_cast<size_t>(numel),
            1,
            2);
        const size_t count1 = rank_partition_count(
            static_cast<size_t>(numel),
            1,
            2);

        for (size_t i = 0; i < count0; ++i) {
            const size_t idx = begin0 + i;
            ref[idx] = ref0[idx];
        }

        for (size_t i = 0; i < count1; ++i) {
            const size_t idx = begin1 + i;
            ref[idx] = ref1[idx];
        }

        testing::expect_allclose(
            got0,
            ref,
            (std::string(label) + " rank0 all_gather").c_str());

        testing::expect_allclose(
            got1,
            ref,
            (std::string(label) + " rank1 all_gather").c_str());

        return;
    }
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

void cuda_malloc_on_device(
    int device,
    half** out,
    size_t bytes,
    const char* what) {
    if (out == nullptr) {
        throw std::invalid_argument("cuda_malloc_on_device: out is null");
    }

    *out = nullptr;

    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMalloc(reinterpret_cast<void**>(out), bytes),
        what);
}

void cuda_free_on_device(
    int device,
    half*& ptr) {
    if (ptr == nullptr) {
        return;
    }

    system::runtime::set_device(device);
    cudaFree(ptr);
    ptr = nullptr;
}

void destroy_oo_buffer(oo_buffer_t*& buffer) {
    if (buffer != nullptr) {
        oo_buffer_destroy(buffer);
        buffer = nullptr;
    }
}

void destroy_oo_node(oo_node_t*& node) {
    if (node != nullptr) {
        oo_node_destroy(node);
        node = nullptr;
    }
}

void destroy_oo_group(oo_group_t*& group) {
    if (group != nullptr) {
        oo_group_destroy(group);
        group = nullptr;
    }
}

void destroy_stream(
    int device,
    cudaStream_t& stream) {
    if (stream != nullptr) {
        system::runtime::destroy_stream_on_device(device, stream);
        stream = nullptr;
    }
}

void destroy_nccl_comms(ncclComm_t* comms) {
    if (comms == nullptr) {
        return;
    }

    for (int i = 0; i < 2; ++i) {
        if (comms[i] != nullptr) {
            ncclCommDestroy(comms[i]);
            comms[i] = nullptr;
        }
    }
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
    BenchCollective collective = parse_collective(collective_name_arg);

    if (numel_arg <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_collective_sm90: invalid args");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_collective_sm90: dev0 and dev1 must differ");
    }

    const size_t numel = static_cast<size_t>(numel_arg);
    validate_collective_size(collective, numel);

    oo_group_t* group = nullptr;
    oo_node_t* node0 = nullptr;
    oo_node_t* node1 = nullptr;

    oo_buffer_t* normal_rank0_buf = nullptr;
    oo_buffer_t* normal_rank1_buf = nullptr;
    oo_buffer_t* not_fused_rank0_buf = nullptr;
    oo_buffer_t* not_fused_rank1_buf = nullptr;
    oo_buffer_t* fused_rank0_buf = nullptr;
    oo_buffer_t* fused_rank1_buf = nullptr;

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;
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
        const size_t bytes = numel * sizeof(half);

        stream0 = system::runtime::create_stream_on_device(node0_dev);
        stream1 = system::runtime::create_stream_on_device(node1_dev);

        check_oo(
            oo_buffer_alloc(node0, bytes, &normal_rank0_buf),
            "oo_buffer_alloc(normal rank0)");
        check_oo(
            oo_buffer_alloc(node1, bytes, &normal_rank1_buf),
            "oo_buffer_alloc(normal rank1)");

        check_oo(
            oo_buffer_alloc(node0, bytes, &not_fused_rank0_buf),
            "oo_buffer_alloc(not_fused rank0)");
        check_oo(
            oo_buffer_alloc(node1, bytes, &not_fused_rank1_buf),
            "oo_buffer_alloc(not_fused rank1)");

        check_oo(
            oo_buffer_alloc(node0, bytes, &fused_rank0_buf),
            "oo_buffer_alloc(fused rank0)");
        check_oo(
            oo_buffer_alloc(node1, bytes, &fused_rank1_buf),
            "oo_buffer_alloc(fused rank1)");

        half* normal_rank0 =
            reinterpret_cast<half*>(oo_buffer_ptr(normal_rank0_buf));
        half* normal_rank1 =
            reinterpret_cast<half*>(oo_buffer_ptr(normal_rank1_buf));

        half* not_fused_rank0 =
            reinterpret_cast<half*>(oo_buffer_ptr(not_fused_rank0_buf));
        half* not_fused_rank1 =
            reinterpret_cast<half*>(oo_buffer_ptr(not_fused_rank1_buf));

        half* fused_rank0 =
            reinterpret_cast<half*>(oo_buffer_ptr(fused_rank0_buf));
        half* fused_rank1 =
            reinterpret_cast<half*>(oo_buffer_ptr(fused_rank1_buf));

        cuda_malloc_on_device(
            node0_dev,
            &rank0_src,
            bytes,
            "cudaMalloc(rank0_src)");
        cuda_malloc_on_device(
            node1_dev,
            &rank1_src,
            bytes,
            "cudaMalloc(rank1_src)");
        cuda_malloc_on_device(
            node0_dev,
            &nccl_rank0_buf,
            bytes,
            "cudaMalloc(nccl_rank0_buf)");
        cuda_malloc_on_device(
            node1_dev,
            &nccl_rank1_buf,
            bytes,
            "cudaMalloc(nccl_rank1_buf)");

        fill_inputs(
            rank0_src,
            rank1_src,
            numel_arg,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclCommInitAll(comms, 2, devices));

        std::map<std::string, double> results;

        auto bench_ooverlap_variant =
            [&](const char* result_key,
                comm::AllreducePlanKind kind,
                half* rank0_work,
                half* rank1_work) {
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

                run_ooverlap_iters(
                    collective,
                    kind,
                    group,
                    rank0_work,
                    rank1_work,
                    numel,
                    node0_dev,
                    node1_dev,
                    stream0,
                    stream1,
                    warmup);

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
                    elapsed_ms_ooverlap(
                        collective,
                        kind,
                        group,
                        rank0_work,
                        rank1_work,
                        numel,
                        node0_dev,
                        node1_dev,
                        stream0,
                        stream1,
                        iters);

                sync_two_streams(
                    node0_dev,
                    stream0,
                    node1_dev,
                    stream1,
                    "sync ooverlap measured variant");

                verify_collective_result(
                    collective,
                    result_key,
                    rank0_work,
                    rank1_work,
                    numel_arg,
                    node0_dev,
                    node1_dev);

                results[result_key] = total_ms / static_cast<double>(iters);
            };

        bench_ooverlap_variant(
            "normal_ms",
            comm::AllreducePlanKind::TmaCopy,
            normal_rank0,
            normal_rank1);

        bench_ooverlap_variant(
            "not_fused_ms",
            comm::AllreducePlanKind::SeqFastGmem,
            not_fused_rank0,
            not_fused_rank1);

        bench_ooverlap_variant(
            "fused_ms",
            comm::AllreducePlanKind::OverlapFastGmem,
            fused_rank0,
            fused_rank1);

        prepare_work_buffers(
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

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync nccl warmup");

        prepare_work_buffers(
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

        sync_two_streams(
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

        results["nccl_ms"] = nccl_total_ms / static_cast<double>(iters);

        results["numel"] = static_cast<double>(numel);
        results["iters"] = static_cast<double>(iters);
        results["warmup"] = static_cast<double>(warmup);

        destroy_nccl_comms(comms);

        cuda_free_on_device(node0_dev, rank0_src);
        cuda_free_on_device(node1_dev, rank1_src);
        cuda_free_on_device(node0_dev, nccl_rank0_buf);
        cuda_free_on_device(node1_dev, nccl_rank1_buf);

        destroy_oo_buffer(normal_rank0_buf);
        destroy_oo_buffer(normal_rank1_buf);
        destroy_oo_buffer(not_fused_rank0_buf);
        destroy_oo_buffer(not_fused_rank1_buf);
        destroy_oo_buffer(fused_rank0_buf);
        destroy_oo_buffer(fused_rank1_buf);

        destroy_oo_node(node0);
        destroy_oo_node(node1);
        destroy_oo_group(group);

        destroy_stream(node0_dev, stream0);
        destroy_stream(node1_dev, stream1);

        return results;
    } catch (...) {
        const int node0_dev = (node0 != nullptr) ? oo_node_device(node0) : dev0;
        const int node1_dev = (node1 != nullptr) ? oo_node_device(node1) : dev1;

        destroy_nccl_comms(comms);

        cuda_free_on_device(node0_dev, rank0_src);
        cuda_free_on_device(node1_dev, rank1_src);
        cuda_free_on_device(node0_dev, nccl_rank0_buf);
        cuda_free_on_device(node1_dev, nccl_rank1_buf);

        destroy_oo_buffer(normal_rank0_buf);
        destroy_oo_buffer(normal_rank1_buf);
        destroy_oo_buffer(not_fused_rank0_buf);
        destroy_oo_buffer(not_fused_rank1_buf);
        destroy_oo_buffer(fused_rank0_buf);
        destroy_oo_buffer(fused_rank1_buf);

        destroy_oo_node(node0);
        destroy_oo_node(node1);
        destroy_oo_group(group);

        destroy_stream(node0_dev, stream0);
        destroy_stream(node1_dev, stream1);

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
