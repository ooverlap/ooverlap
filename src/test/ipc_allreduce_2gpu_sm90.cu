#include "test/ipc_allreduce_2gpu_sm90.h"

#include "comm/ooverlap_comm_internal.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

float rank_scale(int rank) {
    return rank == 0 ? 0.25f : 0.50f;
}

float rank_offset(int rank) {
    return rank == 0 ? 1.0f : 2.0f;
}

std::vector<float> reference_two_gpu_sum_fp16(int64_t numel) {
    auto ref0 = testing::host_reference_pattern_fp16(
        numel,
        rank_scale(0),
        rank_offset(0));

    auto ref1 = testing::host_reference_pattern_fp16(
        numel,
        rank_scale(1),
        rank_offset(1));

    std::vector<float> ref(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        float acc = ref0[static_cast<size_t>(i)];
        acc = testing::round_to_half(acc + ref1[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = acc;
    }

    return ref;
}

void fill_local_source(
    half* local_src,
    int64_t numel,
    int local_rank,
    int device,
    cudaStream_t stream) {
    system::runtime::set_device(device);

    testing::fill_pattern(
        local_src,
        numel,
        rank_scale(local_rank),
        rank_offset(local_rank),
        stream);

    system::runtime::sync_stream_on_device(
        device,
        stream,
        "sync fill_local_source");
}

void reset_work_buffer(
    half* local_work,
    const half* local_src,
    size_t bytes,
    int device,
    cudaStream_t stream) {
    system::runtime::set_device(device);

    system::runtime::check_cuda(
        cudaMemcpyAsync(
            local_work,
            local_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream),
        "cudaMemcpyAsync(local_src -> local_work)");

    system::runtime::sync_stream_on_device(
        device,
        stream,
        "sync reset_work_buffer");
}

void verify_local_result(
    const char* label,
    half* local_work,
    int64_t numel,
    int local_rank,
    int device) {
    auto got = testing::copy_half_device_to_host_float(
        local_work,
        numel,
        device);

    auto ref = reference_two_gpu_sum_fp16(numel);

    testing::expect_allclose(
        got,
        ref,
        (std::string(label) + " rank" + std::to_string(local_rank)).c_str());
}

} // namespace

bool tma_ipc_two_gpu_allreduce_rank_smoke_test(
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    int iters) {
    if (numel <= 0) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: numel must be > 0");
    }
    if (iters <= 0) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: iters must be > 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: dev0 and dev1 must differ");
    }
    if (local_rank != 0 && local_rank != 1) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: local_rank must be 0 or 1");
    }
    if (broker_key.empty()) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: broker_key must be non-empty");
    }

    int devices[2] = {dev0, dev1};
    const int peer_rank = local_rank ^ 1;

    oo_group_t* group = nullptr;
    oo_node_t* node = nullptr;
    oo_buffer_t* local_buf = nullptr;
    oo_buffer_t* peer_buf = nullptr;

    half* local_src = nullptr;
    half* local_work = nullptr;

    cudaStream_t stream = nullptr;

    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    /*
     * Important:
     *   This function is intended to be run from exactly two OS processes:
     *     process 0 calls local_rank=0
     *     process 1 calls local_rank=1
     *
     *   Do not call this twice sequentially in one process; the broker expects
     *   both local ranks to be alive at the same time.
     */

    check_oo(
        oo_group_create_ipc(
            devices,
            2,
            local_rank,
            broker_key.c_str(),
            &group),
        "oo_group_create_ipc");

    check_oo(
        oo_node_create(
            group,
            local_rank,
            &node),
        "oo_node_create(local)");

    const int local_device = oo_node_device(node);

    stream = system::runtime::create_stream_on_device(local_device);

    system::runtime::set_device(local_device);
    system::runtime::check_cuda(
        cudaMalloc(&local_src, bytes),
        "cudaMalloc(local_src)");
    system::runtime::check_cuda(
        cudaMalloc(&local_work, bytes),
        "cudaMalloc(local_work)");

    fill_local_source(
        local_src,
        numel,
        local_rank,
        local_device,
        stream);

    /*
     * Use cudaMalloc + oo_buffer_wrap for the local data buffer.
     *
     * Do not use oo_buffer_alloc here: that allocates VMM memory and the current
     * oo_buffer_export_legacy_descriptor helper intentionally rejects VMM for
     * legacy CUDA IPC export.
     */
    check_oo(
        oo_buffer_wrap(
            node,
            local_work,
            bytes,
            &local_buf),
        "oo_buffer_wrap(local_work)");

    ooverlap::system::legacy_peer_buffer_descriptor local_desc{};
    check_oo(
        oo_buffer_export_legacy_descriptor(
            local_buf,
            &local_desc),
        "oo_buffer_export_legacy_descriptor(local)");

    std::vector<ooverlap::system::legacy_peer_buffer_descriptor> all_desc(2);

    group->broker->exchange_data(
        all_desc.data(),
        &local_desc,
        sizeof(local_desc));

    check_oo(
        oo_buffer_import_legacy_descriptor(
            node,
            all_desc[peer_rank],
            &peer_buf),
        "oo_buffer_import_legacy_descriptor(peer)");

    /*
     * Ensure both processes have imported peer data buffers before either rank
     * launches the first collective.
     */
    group->broker->sync();

    for (int iter = 0; iter < iters; ++iter) {
        reset_work_buffer(
            local_work,
            local_src,
            bytes,
            local_device,
            stream);

        /*
         * Conservative host barrier for the smoke test. This makes the test
         * validate the IPC path without relying on overlap/timing behavior.
         */
        group->broker->sync();

        check_oo(
            oo_allreduce(
                node,
                local_buf,
                peer_buf,
                static_cast<size_t>(numel),
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream),
            "oo_allreduce(ipc)");

        system::runtime::sync_stream_on_device(
            local_device,
            stream,
            "sync oo_allreduce(ipc)");

        verify_local_result(
            "ipc oo_allreduce",
            local_work,
            numel,
            local_rank,
            local_device);

        /*
         * Make sure both ranks finish verification before either rank resets
         * local_work for the next iteration.
         */
        group->broker->sync();
    }

    /*
     * Data-buffer cleanup must be ordered:
     *   1. both ranks finish kernels/verifications
     *   2. both ranks close imported peer mappings
     *   3. both ranks are allowed to free local cudaMalloc storage
     */
    group->broker->sync();

    oo_buffer_destroy(peer_buf);
    peer_buf = nullptr;

    group->broker->sync();

    oo_buffer_destroy(local_buf);
    local_buf = nullptr;

    system::runtime::set_device(local_device);
    system::runtime::check_cuda(
        cudaFree(local_work),
        "cudaFree(local_work)");
    local_work = nullptr;

    system::runtime::check_cuda(
        cudaFree(local_src),
        "cudaFree(local_src)");
    local_src = nullptr;

    oo_node_destroy(node);
    node = nullptr;

    oo_group_destroy(group);
    group = nullptr;

    system::runtime::destroy_stream_on_device(
        local_device,
        stream);
    stream = nullptr;

    return true;
}

} // namespace ooverlap
