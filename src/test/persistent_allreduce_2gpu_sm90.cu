#include "test/persistent_allreduce_2gpu_sm90.h"

#include "comm/ooverlap_comm.h"
#include "comm/ooverlap_comm_internal.h"
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

void launch_normal_once(
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int* rank0_ready,
    int* rank1_ready,
    int collective_epoch) {
    system::runtime::check_cuda(
        enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
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
            collective_epoch),
        "enqueue normal allreduce rank0");

    system::runtime::check_cuda(
        enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
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
            collective_epoch),
        "enqueue normal allreduce rank1");
}

void launch_normal_diff_buffer_once(
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_out,
    half* rank1_out,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int* rank0_ready,
    int* rank1_ready,
    int collective_epoch) {
    system::runtime::check_cuda(
        enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
            rank0_src,
            rank0_out,
            const_cast<half*>(rank1_src),
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            0,
            dev0,
            dev1,
            stream0,
            rank0_ready,
            rank1_ready,
            collective_epoch),
        "enqueue normal diff-buffer allreduce rank0");

    system::runtime::check_cuda(
        enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
            rank1_src,
            rank1_out,
            const_cast<half*>(rank0_src),
            numel,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            1,
            dev0,
            dev1,
            stream1,
            rank1_ready,
            rank0_ready,
            collective_epoch),
        "enqueue normal diff-buffer allreduce rank1");
}

void launch_not_fused_once(
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int* rank0_ready,
    int* rank1_ready,
    int collective_epoch) {
    system::runtime::check_cuda(
        enqueue_tma_two_gpu_peer_allreduce_rank_seq_fastcopy_sm90(
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
            collective_epoch),
        "enqueue not_fused allreduce rank0");

    system::runtime::check_cuda(
        enqueue_tma_two_gpu_peer_allreduce_rank_seq_fastcopy_sm90(
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
            collective_epoch),
        "enqueue not_fused allreduce rank1");
}

void launch_fused_once(
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int* rank0_ready,
    int* rank1_ready,
    int collective_epoch) {
    system::runtime::check_cuda(
        enqueue_tma_two_gpu_peer_allreduce_rank_overlap_fastcopy_sm90(
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
            collective_epoch),
        "enqueue fused allreduce rank0");

    system::runtime::check_cuda(
        enqueue_tma_two_gpu_peer_allreduce_rank_overlap_fastcopy_sm90(
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
            collective_epoch),
        "enqueue fused allreduce rank1");
}

void run_normal_iters(
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
        launch_normal_once(
            rank0_work,
            rank1_work,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            rank0_ready,
            rank1_ready,
            i + 1);
    }

    sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync normal warmup");
}

void run_normal_diff_buffer_iters(
    oo_group_t* group,
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_out,
    half* rank1_out,
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
        launch_normal_diff_buffer_once(
            rank0_src,
            rank1_src,
            rank0_out,
            rank1_out,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            rank0_ready,
            rank1_ready,
            i + 1);
    }

    sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync normal diff-buffer warmup");
}

void run_not_fused_iters(
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
        launch_not_fused_once(
            rank0_work,
            rank1_work,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            rank0_ready,
            rank1_ready,
            i + 1);
    }

    sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync not_fused warmup");
}

void run_fused_iters(
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
        launch_fused_once(
            rank0_work,
            rank1_work,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            rank0_ready,
            rank1_ready,
            i + 1);
    }

    sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync fused warmup");
}

double elapsed_ms_normal_allreduce(
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
            const int collective_epoch = epoch++;

            launch_normal_once(
                rank0_work,
                rank1_work,
                numel,
                dev0,
                dev1,
                stream0,
                stream1,
                rank0_ready,
                rank1_ready,
                collective_epoch);
        });
}

double elapsed_ms_normal_diff_buffer_allreduce(
    oo_group_t* group,
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_out,
    half* rank1_out,
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
            const int collective_epoch = epoch++;

            launch_normal_diff_buffer_once(
                rank0_src,
                rank1_src,
                rank0_out,
                rank1_out,
                numel,
                dev0,
                dev1,
                stream0,
                stream1,
                rank0_ready,
                rank1_ready,
                collective_epoch);
        });
}

double elapsed_ms_not_fused_allreduce(
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
            const int collective_epoch = epoch++;

            launch_not_fused_once(
                rank0_work,
                rank1_work,
                numel,
                dev0,
                dev1,
                stream0,
                stream1,
                rank0_ready,
                rank1_ready,
                collective_epoch);
        });
}

double elapsed_ms_fused_allreduce(
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
            const int collective_epoch = epoch++;

            launch_fused_once(
                rank0_work,
                rank1_work,
                numel,
                dev0,
                dev1,
                stream0,
                stream1,
                rank0_ready,
                rank1_ready,
                collective_epoch);
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
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());

        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank0_src,
                nccl_rank0_out,
                numel,
                ncclFloat16,
                ncclSum,
                comms[0],
                stream0));

        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank1_src,
                nccl_rank1_out,
                numel,
                ncclFloat16,
                ncclSum,
                comms[1],
                stream1));

        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
    }
}

double elapsed_ms_nccl_allreduce(
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
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());

            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank0_src,
                    nccl_rank0_out,
                    numel,
                    ncclFloat16,
                    ncclSum,
                    comms[0],
                    stream0));

            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank1_src,
                    nccl_rank1_out,
                    numel,
                    ncclFloat16,
                    ncclSum,
                    comms[1],
                    stream1));

            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());
        });
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

    testing::expect_allclose(
        got0,
        ref,
        (std::string(label) + " rank0").c_str());

    testing::expect_allclose(
        got1,
        ref,
        (std::string(label) + " rank1").c_str());
}

void verify_nccl_result(
    const char* label,
    half* rank0,
    half* rank1,
    int64_t numel,
    int dev0,
    int dev1) {
    verify_two_gpu_allreduce_result(label, rank0, rank1, numel, dev0, dev1);
}

} // namespace

bool tma_persistent_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    if (numel <= 0) {
        throw std::invalid_argument(
            "tma_persistent_two_gpu_allreduce_smoke_test: numel must be > 0");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument(
            "tma_persistent_two_gpu_allreduce_smoke_test: dev0 and dev1 must differ");
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

        check_oo(oo_group_create(devices, 2, &group), "oo_group_create");
        check_oo(oo_node_create(group, 0, &node0), "oo_node_create(rank0)");
        check_oo(oo_node_create(group, 1, &node1), "oo_node_create(rank1)");

        const int node0_dev = oo_node_device(node0);
        const int node1_dev = oo_node_device(node1);
        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

        stream0 = system::runtime::create_stream_on_device(node0_dev);
        stream1 = system::runtime::create_stream_on_device(node1_dev);

        check_oo(oo_buffer_alloc(node0, bytes, &buf0), "oo_buffer_alloc(rank0)");
        check_oo(oo_buffer_alloc(node1, bytes, &buf1), "oo_buffer_alloc(rank1)");

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

        verify_two_gpu_allreduce_result(
            "oo allreduce smoke",
            rank0_buf,
            rank1_buf,
            numel,
            node0_dev,
            node1_dev);

        oo_buffer_destroy(buf0);
        oo_buffer_destroy(buf1);
        oo_node_destroy(node0);
        oo_node_destroy(node1);
        oo_group_destroy(group);

        system::runtime::destroy_stream_on_device(node0_dev, stream0);
        system::runtime::destroy_stream_on_device(node1_dev, stream1);

        return true;
    } catch (...) {
        const int node0_dev = (node0 != nullptr) ? oo_node_device(node0) : dev0;
        const int node1_dev = (node1 != nullptr) ? oo_node_device(node1) : dev1;

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
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_allreduce_sm90: invalid args");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_allreduce_sm90: dev0 and dev1 must differ");
    }

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

        half* normal_rank0_work =
            reinterpret_cast<half*>(oo_buffer_ptr(normal_rank0_buf));
        half* normal_rank1_work =
            reinterpret_cast<half*>(oo_buffer_ptr(normal_rank1_buf));

        half* not_fused_rank0_work =
            reinterpret_cast<half*>(oo_buffer_ptr(not_fused_rank0_buf));
        half* not_fused_rank1_work =
            reinterpret_cast<half*>(oo_buffer_ptr(not_fused_rank1_buf));

        half* fused_rank0_work =
            reinterpret_cast<half*>(oo_buffer_ptr(fused_rank0_buf));
        half* fused_rank1_work =
            reinterpret_cast<half*>(oo_buffer_ptr(fused_rank1_buf));

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

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            normal_rank0_work,
            normal_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        run_normal_iters(
            group,
            normal_rank0_work,
            normal_rank1_work,
            static_cast<size_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            warmup);

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            normal_rank0_work,
            normal_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        const double normal_total_ms =
            elapsed_ms_normal_allreduce(
                group,
                normal_rank0_work,
                normal_rank1_work,
                static_cast<size_t>(numel),
                node0_dev,
                node1_dev,
                stream0,
                stream1,
                iters);

        run_normal_diff_buffer_iters(
            group,
            rank0_src,
            rank1_src,
            normal_rank0_work,
            normal_rank1_work,
            static_cast<size_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            warmup);
        
        const double normal_diff_buffer_total_ms =
            elapsed_ms_normal_diff_buffer_allreduce(
                group,
                rank0_src,
                rank1_src,
                normal_rank0_work,
                normal_rank1_work,
                static_cast<size_t>(numel),
                node0_dev,
                node1_dev,
                stream0,
                stream1,
                iters);
        
        prepare_work_buffers(
            rank0_src,
            rank1_src,
            not_fused_rank0_work,
            not_fused_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        run_not_fused_iters(
            group,
            not_fused_rank0_work,
            not_fused_rank1_work,
            static_cast<size_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            warmup);

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            not_fused_rank0_work,
            not_fused_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        const double not_fused_total_ms =
            elapsed_ms_not_fused_allreduce(
                group,
                not_fused_rank0_work,
                not_fused_rank1_work,
                static_cast<size_t>(numel),
                node0_dev,
                node1_dev,
                stream0,
                stream1,
                iters);

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            fused_rank0_work,
            fused_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        run_fused_iters(
            group,
            fused_rank0_work,
            fused_rank1_work,
            static_cast<size_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            warmup);

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            fused_rank0_work,
            fused_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        const double fused_total_ms =
            elapsed_ms_fused_allreduce(
                group,
                fused_rank0_work,
                fused_rank1_work,
                static_cast<size_t>(numel),
                node0_dev,
                node1_dev,
                stream0,
                stream1,
                iters);

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
            "sync NCCL warmup");

        const double nccl_total_ms =
            elapsed_ms_nccl_allreduce(
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

#if OOVERLAP_BENCH_VERIFY_RESULTS
        prepare_work_buffers(
            rank0_src,
            rank1_src,
            normal_rank0_work,
            normal_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        reset_ready_signals(group);
        launch_normal_once(
            normal_rank0_work,
            normal_rank1_work,
            static_cast<size_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            ready_signal_ptr(group, 0),
            ready_signal_ptr(group, 1),
            1);

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync normal verify");

        verify_two_gpu_allreduce_result(
            "normal benchmark verify",
            normal_rank0_work,
            normal_rank1_work,
            numel,
            node0_dev,
            node1_dev);

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            not_fused_rank0_work,
            not_fused_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        reset_ready_signals(group);
        launch_not_fused_once(
            not_fused_rank0_work,
            not_fused_rank1_work,
            static_cast<size_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            ready_signal_ptr(group, 0),
            ready_signal_ptr(group, 1),
            1);

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync not_fused verify");

        verify_two_gpu_allreduce_result(
            "not_fused benchmark verify",
            not_fused_rank0_work,
            not_fused_rank1_work,
            numel,
            node0_dev,
            node1_dev);

        prepare_work_buffers(
            rank0_src,
            rank1_src,
            fused_rank0_work,
            fused_rank1_work,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        reset_ready_signals(group);
        launch_fused_once(
            fused_rank0_work,
            fused_rank1_work,
            static_cast<size_t>(numel),
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            ready_signal_ptr(group, 0),
            ready_signal_ptr(group, 1),
            1);

        sync_two_streams(
            node0_dev,
            stream0,
            node1_dev,
            stream1,
            "sync fused verify");

        verify_two_gpu_allreduce_result(
            "fused benchmark verify",
            fused_rank0_work,
            fused_rank1_work,
            numel,
            node0_dev,
            node1_dev);

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

        verify_nccl_result(
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
        system::runtime::check_cuda(
            cudaFree(rank0_src),
            "cudaFree(rank0_src)");
        system::runtime::check_cuda(
            cudaFree(nccl_rank0_out),
            "cudaFree(nccl_rank0_out)");
        rank0_src = nullptr;
        nccl_rank0_out = nullptr;

        system::runtime::set_device(node1_dev);
        system::runtime::check_cuda(
            cudaFree(rank1_src),
            "cudaFree(rank1_src)");
        system::runtime::check_cuda(
            cudaFree(nccl_rank1_out),
            "cudaFree(nccl_rank1_out)");
        rank1_src = nullptr;
        nccl_rank1_out = nullptr;

        oo_buffer_destroy(normal_rank0_buf);
        oo_buffer_destroy(normal_rank1_buf);
        oo_buffer_destroy(not_fused_rank0_buf);
        oo_buffer_destroy(not_fused_rank1_buf);
        oo_buffer_destroy(fused_rank0_buf);
        oo_buffer_destroy(fused_rank1_buf);
        normal_rank0_buf = nullptr;
        normal_rank1_buf = nullptr;
        not_fused_rank0_buf = nullptr;
        not_fused_rank1_buf = nullptr;
        fused_rank0_buf = nullptr;
        fused_rank1_buf = nullptr;

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

        const double avg_normal_ms =
            normal_total_ms / static_cast<double>(iters);
        const double avg_not_fused_ms =
            not_fused_total_ms / static_cast<double>(iters);
        const double avg_fused_ms =
            fused_total_ms / static_cast<double>(iters);
        const double avg_nccl_ms =
            nccl_total_ms / static_cast<double>(iters);
        const double avg_ms_normal_diff_buffer =
            normal_diff_buffer_total_ms / static_cast<double>(iters);

        return {
            {"numel", static_cast<double>(numel)},
            {"iters", static_cast<double>(iters)},
            {"warmup", static_cast<double>(warmup)},

            {"avg_ms_normal", avg_normal_ms},
            {"avg_ms_not_fused", avg_not_fused_ms},
            {"avg_ms_fused", avg_fused_ms},
            {"avg_ms_nccl", avg_nccl_ms},
            {"avg_ms_normal_diff_buffer", avg_ms_normal_diff_buffer},

            {"speedup_normal_over_nccl", avg_nccl_ms / avg_normal_ms},
            {"speedup_fused_over_nccl", avg_nccl_ms / avg_fused_ms},
            {"speedup_not_fused_over_nccl", avg_nccl_ms / avg_not_fused_ms},
            {"speedup_normal_diff_buffer_over_nccl", avg_nccl_ms / avg_ms_normal_diff_buffer},
            {"speedup_fused_over_normal", avg_normal_ms / avg_fused_ms},
            {"speedup_not_fused_over_normal", avg_normal_ms / avg_not_fused_ms},
            {"speedup_normal_diff_buffer_over_normal", avg_normal_ms / avg_ms_normal_diff_buffer},

            {"verify_results", static_cast<double>(OOVERLAP_BENCH_VERIFY_RESULTS)}
        };
    } catch (...) {
        if (comms[0] != nullptr) {
            ncclCommDestroy(comms[0]);
        }
        if (comms[1] != nullptr) {
            ncclCommDestroy(comms[1]);
        }

        if (rank0_src != nullptr) {
            system::runtime::set_device(dev0);
            cudaFree(rank0_src);
        }
        if (nccl_rank0_out != nullptr) {
            system::runtime::set_device(dev0);
            cudaFree(nccl_rank0_out);
        }

        if (rank1_src != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(rank1_src);
        }
        if (nccl_rank1_out != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(nccl_rank1_out);
        }

        if (normal_rank0_buf != nullptr) {
            oo_buffer_destroy(normal_rank0_buf);
        }
        if (normal_rank1_buf != nullptr) {
            oo_buffer_destroy(normal_rank1_buf);
        }

        if (not_fused_rank0_buf != nullptr) {
            oo_buffer_destroy(not_fused_rank0_buf);
        }
        if (not_fused_rank1_buf != nullptr) {
            oo_buffer_destroy(not_fused_rank1_buf);
        }

        if (fused_rank0_buf != nullptr) {
            oo_buffer_destroy(fused_rank0_buf);
        }
        if (fused_rank1_buf != nullptr) {
            oo_buffer_destroy(fused_rank1_buf);
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
