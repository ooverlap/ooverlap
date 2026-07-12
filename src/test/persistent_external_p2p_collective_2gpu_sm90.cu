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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OOVERLAP_BENCH_VERIFY_RESULTS
#define OOVERLAP_BENCH_VERIFY_RESULTS 0
#endif

namespace ooverlap {
namespace {

using testing::TestCollective;

/*
 * OOVERLAP_PERSISTENT_EXTERNAL_P2P_RING_MODE_PATCH
 *
 * This benchmark intentionally uses a ring of independent work buffers for all
 * three implementations:
 *
 *   - ooverlap external cudaMalloc + oo_buffer_wrap
 *   - normal NCCL cudaMalloc
 *   - NCCL symmetric ncclMemAlloc + ncclCommWindowRegister
 *
 * Setup remains outside the measured CUDA-event region.  The measured loop
 * rotates through ring[i % ring_size] so each implementation sees many distinct
 * addresses instead of one hot address.
 *
 * Tune with:
 *
 *   OOVERLAP_BENCH_RING_SIZE=16
 */

int ring_size_from_env() {
    constexpr int kDefaultRingSize = 16;
    constexpr int kMaxReasonableRingSize = 4096;

    const char* env = std::getenv("OOVERLAP_BENCH_RING_SIZE");
    if (env == nullptr || env[0] == '\0') {
        return kDefaultRingSize;
    }

    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed <= 0) {
        return kDefaultRingSize;
    }

    return static_cast<int>(
        std::min<long>(parsed, kMaxReasonableRingSize));
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

    /*
     * Cleanup path should not throw.
     */
    (void)cudaSetDevice(device);
    (void)ncclMemFree(static_cast<void*>(ptr));

    ptr = nullptr;
}

void register_nccl_symmetric_windows(
    ncclComm_t* comms,
    half* rank0_buf,
    half* rank1_buf,
    size_t bytes,
    ncclWindow_t& rank0_win,
    ncclWindow_t& rank1_win) {
    rank0_win = nullptr;
    rank1_win = nullptr;

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());

    OOVERLAP_TEST_NCCL_CHECK(
        ncclCommWindowRegister(
            comms[0],
            rank0_buf,
            bytes,
            &rank0_win,
            NCCL_WIN_COLL_SYMMETRIC));

    OOVERLAP_TEST_NCCL_CHECK(
        ncclCommWindowRegister(
            comms[1],
            rank1_buf,
            bytes,
            &rank1_win,
            NCCL_WIN_COLL_SYMMETRIC));

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
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

struct OoverlapRingSlot {
    half* rank0_work = nullptr;
    half* rank1_work = nullptr;
    oo_buffer_t* rank0_buf = nullptr;
    oo_buffer_t* rank1_buf = nullptr;
};

struct NcclRingSlot {
    half* rank0_work = nullptr;
    half* rank1_work = nullptr;
    ncclWindow_t rank0_win = nullptr;
    ncclWindow_t rank1_win = nullptr;
};

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

void launch_ooverlap_public_once(
    TestCollective collective,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    launch_ooverlap_public_once_for_rank(
        collective,
        node0,
        rank0_buf,
        numel,
        stream0,
        "ooverlap rank0");

    launch_ooverlap_public_once_for_rank(
        collective,
        node1,
        rank1_buf,
        numel,
        stream1,
        "ooverlap rank1");
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

void prepare_ooverlap_ring(
    const std::vector<OoverlapRingSlot>& ring,
    const half* rank0_src,
    const half* rank1_src,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    for (const OoverlapRingSlot& slot : ring) {
        testing::prepare_two_work_buffers(
            rank0_src,
            rank1_src,
            slot.rank0_work,
            slot.rank1_work,
            bytes,
            dev0,
            dev1,
            stream0,
            stream1);
    }
}

void prepare_nccl_ring(
    const std::vector<NcclRingSlot>& ring,
    const half* rank0_src,
    const half* rank1_src,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    for (const NcclRingSlot& slot : ring) {
        testing::prepare_two_work_buffers(
            rank0_src,
            rank1_src,
            slot.rank0_work,
            slot.rank1_work,
            bytes,
            dev0,
            dev1,
            stream0,
            stream1);
    }
}

void run_ooverlap_ring_iters(
    TestCollective collective,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    const std::vector<OoverlapRingSlot>& ring,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters,
    const char* sync_label) {
    if (iters <= 0) {
        return;
    }

    if (ring.empty()) {
        throw std::invalid_argument("run_ooverlap_ring_iters: ring is empty");
    }

    testing::reset_ready_signals(group);

    for (int i = 0; i < iters; ++i) {
        const OoverlapRingSlot& slot =
            ring[static_cast<std::size_t>(i) % ring.size()];

        launch_ooverlap_public_once(
            collective,
            node0,
            node1,
            slot.rank0_buf,
            slot.rank1_buf,
            numel,
            stream0,
            stream1);
    }

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        sync_label);
}

double elapsed_ms_ooverlap_ring(
    TestCollective collective,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    const std::vector<OoverlapRingSlot>& ring,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    if (ring.empty()) {
        throw std::invalid_argument("elapsed_ms_ooverlap_ring: ring is empty");
    }

    testing::reset_ready_signals(group);

    return testing::elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int i) {
            const OoverlapRingSlot& slot =
                ring[static_cast<std::size_t>(i) % ring.size()];

            launch_ooverlap_public_once(
                collective,
                node0,
                node1,
                slot.rank0_buf,
                slot.rank1_buf,
                numel,
                stream0,
                stream1);
        });
}

void run_nccl_ring_iters(
    TestCollective collective,
    const std::vector<NcclRingSlot>& ring,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters,
    const char* sync_label) {
    if (iters <= 0) {
        return;
    }

    if (ring.empty()) {
        throw std::invalid_argument("run_nccl_ring_iters: ring is empty");
    }

    for (int i = 0; i < iters; ++i) {
        const NcclRingSlot& slot =
            ring[static_cast<std::size_t>(i) % ring.size()];

        launch_nccl_once(
            collective,
            slot.rank0_work,
            slot.rank1_work,
            numel,
            stream0,
            stream1,
            comms);
    }

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        sync_label);
}

double elapsed_ms_nccl_ring(
    TestCollective collective,
    const std::vector<NcclRingSlot>& ring,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters) {
    if (ring.empty()) {
        throw std::invalid_argument("elapsed_ms_nccl_ring: ring is empty");
    }

    return testing::elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int i) {
            const NcclRingSlot& slot =
                ring[static_cast<std::size_t>(i) % ring.size()];

            launch_nccl_once(
                collective,
                slot.rank0_work,
                slot.rank1_work,
                numel,
                stream0,
                stream1,
                comms);
        });
}

void allocate_ooverlap_ring(
    std::vector<OoverlapRingSlot>& ring,
    int ring_size,
    oo_node_t* node0,
    oo_node_t* node1,
    size_t bytes,
    int dev0,
    int dev1) {
    ring.clear();
    ring.resize(static_cast<std::size_t>(ring_size));

    for (int i = 0; i < ring_size; ++i) {
        OoverlapRingSlot& slot =
            ring[static_cast<std::size_t>(i)];

        testing::cuda_malloc_half_on_device(
            dev0,
            &slot.rank0_work,
            bytes,
            "cudaMalloc(ooverlap ring rank0)");

        testing::cuda_malloc_half_on_device(
            dev1,
            &slot.rank1_work,
            bytes,
            "cudaMalloc(ooverlap ring rank1)");

        testing::check_oo(
            oo_buffer_wrap(
                node0,
                slot.rank0_work,
                bytes,
                &slot.rank0_buf),
            "oo_buffer_wrap(ooverlap ring rank0)");

        testing::check_oo(
            oo_buffer_wrap(
                node1,
                slot.rank1_work,
                bytes,
                &slot.rank1_buf),
            "oo_buffer_wrap(ooverlap ring rank1)");
    }
}

void allocate_nccl_ring(
    std::vector<NcclRingSlot>& ring,
    int ring_size,
    size_t bytes,
    int dev0,
    int dev1,
    bool symmetric,
    ncclComm_t* comms) {
    ring.clear();
    ring.resize(static_cast<std::size_t>(ring_size));

    for (int i = 0; i < ring_size; ++i) {
        NcclRingSlot& slot =
            ring[static_cast<std::size_t>(i)];

        if (symmetric) {
            nccl_mem_alloc_half_on_device(
                dev0,
                &slot.rank0_work,
                bytes,
                "ncclMemAlloc(nccl symmetric ring rank0)");

            nccl_mem_alloc_half_on_device(
                dev1,
                &slot.rank1_work,
                bytes,
                "ncclMemAlloc(nccl symmetric ring rank1)");

            register_nccl_symmetric_windows(
                comms,
                slot.rank0_work,
                slot.rank1_work,
                bytes,
                slot.rank0_win,
                slot.rank1_win);
        } else {
            testing::cuda_malloc_half_on_device(
                dev0,
                &slot.rank0_work,
                bytes,
                "cudaMalloc(nccl ring rank0)");

            testing::cuda_malloc_half_on_device(
                dev1,
                &slot.rank1_work,
                bytes,
                "cudaMalloc(nccl ring rank1)");
        }
    }
}

void destroy_ooverlap_ring(
    std::vector<OoverlapRingSlot>& ring,
    int dev0,
    int dev1) {
    for (OoverlapRingSlot& slot : ring) {
        if (slot.rank0_buf != nullptr) {
            oo_buffer_destroy(slot.rank0_buf);
            slot.rank0_buf = nullptr;
        }

        if (slot.rank1_buf != nullptr) {
            oo_buffer_destroy(slot.rank1_buf);
            slot.rank1_buf = nullptr;
        }

        testing::cuda_free_on_device(dev0, slot.rank0_work);
        testing::cuda_free_on_device(dev1, slot.rank1_work);
    }

    ring.clear();
}

void destroy_nccl_ring(
    std::vector<NcclRingSlot>& ring,
    int dev0,
    int dev1,
    bool symmetric,
    ncclComm_t* comms) {
    for (NcclRingSlot& slot : ring) {
        if (symmetric) {
            deregister_nccl_window_best_effort(
                comms != nullptr ? comms[0] : nullptr,
                slot.rank0_win);

            deregister_nccl_window_best_effort(
                comms != nullptr ? comms[1] : nullptr,
                slot.rank1_win);

            nccl_mem_free_on_device(dev0, slot.rank0_work);
            nccl_mem_free_on_device(dev1, slot.rank1_work);
        } else {
            testing::cuda_free_on_device(dev0, slot.rank0_work);
            testing::cuda_free_on_device(dev1, slot.rank1_work);
        }
    }

    ring.clear();
}

void verify_ooverlap_once(
    TestCollective collective,
    oo_node_t* node0,
    oo_node_t* node1,
    const std::vector<OoverlapRingSlot>& ring,
    const half* rank0_src,
    const half* rank1_src,
    size_t numel,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    if (ring.empty()) {
        return;
    }

    const OoverlapRingSlot& slot = ring[0];

    testing::prepare_two_work_buffers(
        rank0_src,
        rank1_src,
        slot.rank0_work,
        slot.rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    testing::reset_ready_signals(oo_node_group(node0));

    launch_ooverlap_public_once(
        collective,
        node0,
        node1,
        slot.rank0_buf,
        slot.rank1_buf,
        numel,
        stream0,
        stream1);

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync ooverlap ring verification");

    verify_collective_result(
        collective,
        "ooverlap",
        slot.rank0_work,
        slot.rank1_work,
        static_cast<int64_t>(numel),
        dev0,
        dev1);
#else
    (void)collective;
    (void)node0;
    (void)node1;
    (void)ring;
    (void)rank0_src;
    (void)rank1_src;
    (void)numel;
    (void)bytes;
    (void)dev0;
    (void)dev1;
    (void)stream0;
    (void)stream1;
#endif
}

void verify_nccl_once(
    TestCollective collective,
    const char* label,
    const std::vector<NcclRingSlot>& ring,
    const half* rank0_src,
    const half* rank1_src,
    size_t numel,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    if (ring.empty()) {
        return;
    }

    const NcclRingSlot& slot = ring[0];

    testing::prepare_two_work_buffers(
        rank0_src,
        rank1_src,
        slot.rank0_work,
        slot.rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    launch_nccl_once(
        collective,
        slot.rank0_work,
        slot.rank1_work,
        numel,
        stream0,
        stream1,
        comms);

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync nccl ring verification");

    verify_collective_result(
        collective,
        label,
        slot.rank0_work,
        slot.rank1_work,
        static_cast<int64_t>(numel),
        dev0,
        dev1);
#else
    (void)collective;
    (void)label;
    (void)ring;
    (void)rank0_src;
    (void)rank1_src;
    (void)numel;
    (void)bytes;
    (void)dev0;
    (void)dev1;
    (void)stream0;
    (void)stream1;
    (void)comms;
#endif
}

void bench_ooverlap_external_ring(
    std::map<std::string, double>& results,
    TestCollective collective,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    const std::vector<OoverlapRingSlot>& ring,
    const half* rank0_src,
    const half* rank1_src,
    size_t numel,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters,
    int warmup) {
    prepare_ooverlap_ring(
        ring,
        rank0_src,
        rank1_src,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    run_ooverlap_ring_iters(
        collective,
        group,
        node0,
        node1,
        ring,
        numel,
        dev0,
        dev1,
        stream0,
        stream1,
        warmup,
        "sync ooverlap ring warmup");

    prepare_ooverlap_ring(
        ring,
        rank0_src,
        rank1_src,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    const double total_ms =
        elapsed_ms_ooverlap_ring(
            collective,
            group,
            node0,
            node1,
            ring,
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
        "sync ooverlap ring measured");

    verify_ooverlap_once(
        collective,
        node0,
        node1,
        ring,
        rank0_src,
        rank1_src,
        numel,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    results["ooverlap_ms"] =
        total_ms / static_cast<double>(iters);
}

void bench_nccl_external_ring(
    std::map<std::string, double>& results,
    const char* result_key,
    TestCollective collective,
    const std::vector<NcclRingSlot>& ring,
    const half* rank0_src,
    const half* rank1_src,
    size_t numel,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters,
    int warmup) {
    prepare_nccl_ring(
        ring,
        rank0_src,
        rank1_src,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    run_nccl_ring_iters(
        collective,
        ring,
        numel,
        dev0,
        dev1,
        stream0,
        stream1,
        comms,
        warmup,
        "sync nccl ring warmup");

    prepare_nccl_ring(
        ring,
        rank0_src,
        rank1_src,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    const double total_ms =
        elapsed_ms_nccl_ring(
            collective,
            ring,
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
        "sync nccl ring measured");

    verify_nccl_once(
        collective,
        result_key,
        ring,
        rank0_src,
        rank1_src,
        numel,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1,
        comms);

    results[result_key] =
        total_ms / static_cast<double>(iters);
}

void cleanup(
    int dev0,
    int dev1,
    half*& rank0_src,
    half*& rank1_src,
    std::vector<OoverlapRingSlot>& ooverlap_ring,
    std::vector<NcclRingSlot>& nccl_ring,
    std::vector<NcclRingSlot>& nccl_symmetric_ring,
    oo_node_t*& node0,
    oo_node_t*& node1,
    oo_group_t*& group,
    cudaStream_t& stream0,
    cudaStream_t& stream1,
    ncclComm_t* comms) {
    destroy_ooverlap_ring(
        ooverlap_ring,
        dev0,
        dev1);

    destroy_nccl_ring(
        nccl_ring,
        dev0,
        dev1,
        false,
        comms);

    destroy_nccl_ring(
        nccl_symmetric_ring,
        dev0,
        dev1,
        true,
        comms);

    testing::destroy_nccl_comms(comms, 2);

    testing::cuda_free_on_device(dev0, rank0_src);
    testing::cuda_free_on_device(dev1, rank1_src);

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

    const int ring_size =
        ring_size_from_env();

    oo_group_t* group = nullptr;
    oo_node_t* node0 = nullptr;
    oo_node_t* node1 = nullptr;

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;

    std::vector<OoverlapRingSlot> ooverlap_ring;
    std::vector<NcclRingSlot> nccl_ring;
    std::vector<NcclRingSlot> nccl_symmetric_ring;

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

        allocate_ooverlap_ring(
            ooverlap_ring,
            ring_size,
            node0,
            node1,
            bytes,
            node0_dev,
            node1_dev);

        allocate_nccl_ring(
            nccl_ring,
            ring_size,
            bytes,
            node0_dev,
            node1_dev,
            false,
            comms);

        allocate_nccl_ring(
            nccl_symmetric_ring,
            ring_size,
            bytes,
            node0_dev,
            node1_dev,
            true,
            comms);

        std::map<std::string, double> results;

        bench_ooverlap_external_ring(
            results,
            collective,
            group,
            node0,
            node1,
            ooverlap_ring,
            rank0_src,
            rank1_src,
            numel,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            iters,
            warmup);

        bench_nccl_external_ring(
            results,
            "nccl_ms",
            collective,
            nccl_ring,
            rank0_src,
            rank1_src,
            numel,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            comms,
            iters,
            warmup);

        bench_nccl_external_ring(
            results,
            "nccl_symmetric_ms",
            collective,
            nccl_symmetric_ring,
            rank0_src,
            rank1_src,
            numel,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            comms,
            iters,
            warmup);

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

        results["ring_size"] =
            static_cast<double>(ring_size);

        cleanup(
            node0_dev,
            node1_dev,
            rank0_src,
            rank1_src,
            ooverlap_ring,
            nccl_ring,
            nccl_symmetric_ring,
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
            ooverlap_ring,
            nccl_ring,
            nccl_symmetric_ring,
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
