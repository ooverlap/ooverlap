#pragma once

#include "test/endpoint_persistent_smoke_test.h"

#include "comm/collective/published_tile.h"
#include "comm/endpoint_persistent_kernel.h"
#include "comm/endpoint_runtime.h"
#include "comm/group.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ooverlap {
namespace {

__global__ void publish_single_ready_tile_kernel(
    comm::collective::ReadyTileQueue queue,
    const half* src,
    uint32_t bytes,
    uint64_t tile_id,
    int* status_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const bool ok = comm::collective::device_publish_ready_tile(
        queue,
        tile_id,
        src,
        bytes,
        0,
        0,
        0,
        nullptr);

    *status_out = ok ? 1 : 0;
}

std::vector<half> make_host_pattern(
    int64_t numel) {
    std::vector<half> out(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        const float x = 0.125f + 0.01f * static_cast<float>(i % 97);
        out[static_cast<size_t>(i)] = __float2half_rn(x);
    }
    return out;
}

void expect_half_vectors_close(
    const std::vector<half>& got,
    const std::vector<half>& ref,
    const char* what) {
    if (got.size() != ref.size()) {
        throw std::runtime_error(std::string(what) + ": size mismatch");
    }

    for (size_t i = 0; i < got.size(); ++i) {
        const float g = __half2float(got[i]);
        const float r = __half2float(ref[i]);
        const float err = std::fabs(g - r);
        if (err > 1.0e-3f) {
            throw std::runtime_error(
                std::string(what) +
                ": mismatch at idx=" + std::to_string(i) +
                " got=" + std::to_string(g) +
                " ref=" + std::to_string(r));
        }
    }
}

bool poll_queue_head_until(
    const comm::collective::ReadyTileQueue& q,
    uint64_t target_head,
    int device,
    int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    while (true) {
        uint64_t head_value = 0;
        system::runtime::set_device(device);
        cudaError_t err = cudaMemcpy(
            &head_value,
            q.head,
            sizeof(uint64_t),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpy(queue head) failed: ") +
                cudaGetErrorString(err));
        }

        if (head_value >= target_head) {
            return true;
        }

        auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms > timeout_ms) {
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace

bool endpoint_persistent_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    if (numel <= 0) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: numel must be > 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: dev0 and dev1 must differ");
    }

    comm::Group group{};
    comm::EndpointRuntime runtime{};
    comm::EndpointPersistentControl control{};

    half* src_dev = nullptr;
    half* dst_dev = nullptr;
    int* publish_status_dev = nullptr;
    cudaStream_t producer_stream = nullptr;

    try {
        // Current group_init requires at least 2 devices.
        comm::group_init(
            &group,
            {dev0, dev1},
            comm::kEndpointPersistentChunkBytes);

        // One queue for one tiny smoke path on rank 0.
        comm::endpoint_runtime_init(
            &runtime,
            &group,
            /*rank=*/0,
            /*num_input_queues=*/1,
            /*input_queue_capacity=*/8);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaStreamCreate(&producer_stream),
            "cudaStreamCreate(producer_stream)");

        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
        const auto host_src = make_host_pattern(numel);
        const std::vector<half> host_zero(static_cast<size_t>(numel), __float2half_rn(0.0f));

        system::runtime::check_cuda(cudaMalloc(&src_dev, bytes), "cudaMalloc(src_dev)");
        system::runtime::check_cuda(cudaMalloc(&dst_dev, bytes), "cudaMalloc(dst_dev)");
        system::runtime::check_cuda(cudaMalloc(&publish_status_dev, sizeof(int)), "cudaMalloc(publish_status_dev)");

        system::runtime::check_cuda(
            cudaMemcpy(src_dev, host_src.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(host_src -> src_dev)");
        system::runtime::check_cuda(
            cudaMemcpy(dst_dev, host_zero.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(host_zero -> dst_dev)");
        system::runtime::check_cuda(
            cudaMemset(publish_status_dev, 0, sizeof(int)),
            "cudaMemset(publish_status_dev)");

        // tile_id=0 maps to dst_ptr + 0 * tile_stride_bytes
        comm::endpoint_runtime_configure_queue_operation(
            &runtime,
            /*queue_idx=*/0,
            /*op_id=*/1,
            /*dst_rank=*/0,
            /*dst_ptr=*/dst_dev,
            /*dst_bytes=*/bytes,
            /*tile_stride_bytes=*/bytes,
            /*expected_contributions=*/1,
            comm::exec::ChunkOpKind::kReduceAddNoFtzF16,
            /*epoch=*/1,
            /*enabled=*/true);

        comm::endpoint_persistent_control_init(&control, dev0);
        system::runtime::check_cuda(
            comm::launch_endpoint_persistent_kernel_sm90(
                comm::endpoint_runtime_device_handle(&runtime),
                &control,
                runtime.endpoint.stream),
            "launch_endpoint_persistent_kernel_sm90");

        // Publish one ready tile into queue 0 from a tiny producer kernel.
        publish_single_ready_tile_kernel<<<1, 1, 0, producer_stream>>>(
            runtime.device.input_queues[0],
            src_dev,
            static_cast<uint32_t>(bytes),
            /*tile_id=*/0,
            publish_status_dev);
        system::runtime::check_cuda(
            cudaGetLastError(),
            "publish_single_ready_tile_kernel");

        system::runtime::check_cuda(
            cudaStreamSynchronize(producer_stream),
            "cudaStreamSynchronize(producer_stream)");

        int publish_status = 0;
        system::runtime::check_cuda(
            cudaMemcpy(
                &publish_status,
                publish_status_dev,
                sizeof(int),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(publish_status)");
        if (publish_status != 1) {
            throw std::runtime_error("publish_single_ready_tile_kernel failed to publish tile");
        }

        // Head advances only after scheduler/pipeline finished the current chunk.
        const bool drained =
            poll_queue_head_until(runtime.input_queues_host[0], /*target_head=*/1, dev0, /*timeout_ms=*/5000);
        if (!drained) {
            throw std::runtime_error("timeout waiting for persistent kernel to consume queue head");
        }

        std::vector<half> host_dst(static_cast<size_t>(numel));
        system::runtime::check_cuda(
            cudaMemcpy(host_dst.data(), dst_dev, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(dst_dev -> host_dst)");

        expect_half_vectors_close(host_dst, host_src, "endpoint_persistent_smoke_test");

        comm::endpoint_persistent_control_request_stop(&control);
        system::runtime::check_cuda(
            cudaStreamSynchronize(runtime.endpoint.stream),
            "cudaStreamSynchronize(persistent stream)");

        if (producer_stream != nullptr) {
            cudaStreamDestroy(producer_stream);
            producer_stream = nullptr;
        }
        if (publish_status_dev != nullptr) {
            cudaFree(publish_status_dev);
            publish_status_dev = nullptr;
        }
        if (dst_dev != nullptr) {
            cudaFree(dst_dev);
            dst_dev = nullptr;
        }
        if (src_dev != nullptr) {
            cudaFree(src_dev);
            src_dev = nullptr;
        }

        comm::endpoint_persistent_control_destroy(&control);
        comm::endpoint_runtime_destroy(&runtime);
        comm::group_destroy(&group);
        return true;
    } catch (...) {
        if (producer_stream != nullptr) {
            cudaStreamDestroy(producer_stream);
        }
        if (publish_status_dev != nullptr) {
            cudaFree(publish_status_dev);
        }
        if (dst_dev != nullptr) {
            cudaFree(dst_dev);
        }
        if (src_dev != nullptr) {
            cudaFree(src_dev);
        }

        comm::endpoint_persistent_control_destroy(&control);
        comm::endpoint_runtime_destroy(&runtime);
        comm::group_destroy(&group);
        throw;
    }
}

} // namespace ooverlap
