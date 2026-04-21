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
#include <string>
#include <thread>
#include <vector>

namespace ooverlap {
namespace {

struct PersistentDebugState {
    unsigned long long loop_count = 0;
    unsigned long long no_work_count = 0;
    unsigned long long primed_count = 0;
    unsigned long long advanced_count = 0;
    unsigned long long last_head = 0;
    unsigned long long last_tail = 0;
};

__global__ void publish_single_ready_tile_kernel(
    comm::collective::ReadyTileQueue queue,
    const half* src,
    uint32_t bytes,
    uint64_t tile_id,
    int* status_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
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

__global__ void spin_only_kernel(
    const uint32_t* stop_flag,
    PersistentDebugState* dbg) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    while ((*reinterpret_cast<volatile const uint32_t*>(stop_flag)) == 0u) {
        dbg->loop_count += 1;
#if defined(__CUDA_ARCH__)
        __nanosleep(256);
#endif
    }
}

__global__ void queue_read_only_kernel(
    comm::collective::ReadyTileQueue queue,
    const uint32_t* stop_flag,
    PersistentDebugState* dbg) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    while ((*reinterpret_cast<volatile const uint32_t*>(stop_flag)) == 0u) {
        dbg->loop_count += 1;
        dbg->last_head = *reinterpret_cast<volatile const uint64_t*>(queue.head);
        dbg->last_tail = *reinterpret_cast<volatile const uint64_t*>(queue.tail);
#if defined(__CUDA_ARCH__)
        __nanosleep(256);
#endif
    }
}

__global__ void scheduler_prime_only_kernel(
    comm::DeviceEndpointRuntime runtime,
    const uint32_t* stop_flag,
    PersistentDebugState* dbg) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    while ((*reinterpret_cast<volatile const uint32_t*>(stop_flag)) == 0u) {
        dbg->loop_count += 1;

        if (runtime.num_scheduler_bindings > 0 &&
            runtime.scheduler_bindings != nullptr &&
            runtime.scheduler_bindings[0].queue != nullptr) {
            auto* q = runtime.scheduler_bindings[0].queue;
            dbg->last_head = *reinterpret_cast<volatile const uint64_t*>(q->head);
            dbg->last_tail = *reinterpret_cast<volatile const uint64_t*>(q->tail);
        }

        comm::exec::ChunkScheduler<1> sched{};
        comm::exec::chunk_scheduler_init<1>(
            &sched,
            runtime.scheduler_bindings,
            runtime.num_scheduler_bindings,
            comm::kEndpointPersistentChunkBytes);

        if (!comm::exec::chunk_scheduler_try_prime_current(&sched)) {
            dbg->no_work_count += 1;
        } else {
            dbg->primed_count += 1;
        }

#if defined(__CUDA_ARCH__)
        __nanosleep(256);
#endif
    }
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

void copy_queue_state(
    const comm::collective::ReadyTileQueue& q,
    uint64_t* head_out,
    uint64_t* tail_out) {
    if (head_out == nullptr || tail_out == nullptr) {
        throw std::invalid_argument("copy_queue_state: null output");
    }

    cudaError_t err = cudaMemcpy(
        head_out,
        q.head,
        sizeof(uint64_t),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("copy_queue_state(head) failed: ") +
            cudaGetErrorString(err));
    }

    err = cudaMemcpy(
        tail_out,
        q.tail,
        sizeof(uint64_t),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("copy_queue_state(tail) failed: ") +
            cudaGetErrorString(err));
    }
}

PersistentDebugState copy_debug_state(
    const PersistentDebugState* dbg_dev) {
    PersistentDebugState out{};
    cudaError_t err = cudaMemcpy(
        &out,
        dbg_dev,
        sizeof(PersistentDebugState),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("copy_debug_state failed: ") +
            cudaGetErrorString(err));
    }
    return out;
}

bool wait_event_with_timeout(
    cudaEvent_t ev,
    int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();

    while (true) {
        cudaError_t st = cudaEventQuery(ev);
        if (st == cudaSuccess) {
            return true;
        }
        if (st != cudaErrorNotReady) {
            throw std::runtime_error(
                std::string("cudaEventQuery failed: ") +
                cudaGetErrorString(st));
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms > timeout_ms) {
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void reset_phase_state(
    comm::EndpointRuntime* runtime,
    half* dst_dev,
    size_t bytes,
    int* publish_status_dev,
    PersistentDebugState* dbg_dev,
    const std::vector<half>& host_zero,
    comm::EndpointPersistentControl* control) {
    comm::collective::ready_tile_queue_reset(&runtime->input_queues_host[0]);

    system::runtime::check_cuda(
        cudaMemcpy(dst_dev, host_zero.data(), bytes, cudaMemcpyHostToDevice),
        "reset_phase_state: cudaMemcpy(host_zero -> dst_dev)");

    system::runtime::check_cuda(
        cudaMemset(publish_status_dev, 0, sizeof(int)),
        "reset_phase_state: cudaMemset(publish_status_dev)");

    system::runtime::check_cuda(
        cudaMemset(dbg_dev, 0, sizeof(PersistentDebugState)),
        "reset_phase_state: cudaMemset(dbg_dev)");

    comm::endpoint_persistent_control_reset(control);
}

bool launch_publish_and_wait(
    const comm::collective::ReadyTileQueue& queue,
    const half* src_dev,
    uint32_t bytes,
    int* publish_status_dev,
    cudaStream_t producer_stream,
    int timeout_ms) {
    cudaEvent_t done = nullptr;
    system::runtime::check_cuda(
        cudaEventCreateWithFlags(&done, cudaEventDisableTiming),
        "cudaEventCreateWithFlags(publish done)");

    publish_single_ready_tile_kernel<<<1, 1, 0, producer_stream>>>(
        queue,
        src_dev,
        bytes,
        0,
        publish_status_dev);
    system::runtime::check_cuda(
        cudaGetLastError(),
        "publish_single_ready_tile_kernel");

    system::runtime::check_cuda(
        cudaEventRecord(done, producer_stream),
        "cudaEventRecord(publish done)");

    const bool finished = wait_event_with_timeout(done, timeout_ms);

    cudaError_t err = cudaEventDestroy(done);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaEventDestroy(publish done) failed: ") +
            cudaGetErrorString(err));
    }

    return finished;
}

void check_publish_status(
    int* publish_status_dev,
    const char* phase_name) {
    int publish_status = 0;
    system::runtime::check_cuda(
        cudaMemcpy(
            &publish_status,
            publish_status_dev,
            sizeof(int),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(publish_status)");

    if (publish_status != 1) {
        throw std::runtime_error(
            std::string(phase_name) + ": publish kernel completed but did not publish tile");
    }
}

void stop_and_join_resident(
    comm::EndpointPersistentControl* control,
    cudaStream_t resident_stream) {
    comm::endpoint_persistent_control_request_stop(control);
    system::runtime::check_cuda(
        cudaStreamSynchronize(resident_stream),
        "cudaStreamSynchronize(resident stream)");
}

void run_spin_only_overlap_phase(
    comm::EndpointRuntime* runtime,
    comm::EndpointPersistentControl* control,
    const half* src_dev,
    half* dst_dev,
    size_t bytes,
    int* publish_status_dev,
    PersistentDebugState* dbg_dev,
    cudaStream_t producer_stream,
    cudaStream_t resident_stream,
    const std::vector<half>& host_zero) {
    std::printf("[phase] spin_only begin\n"); std::fflush(stdout);

    reset_phase_state(runtime, dst_dev, bytes, publish_status_dev, dbg_dev, host_zero, control);
    std::printf("[phase] spin_only after reset\n"); std::fflush(stdout);

    spin_only_kernel<<<1, 1, 0, resident_stream>>>(
        control->stop_flag,
        dbg_dev);
    system::runtime::check_cuda(cudaGetLastError(), "spin_only_kernel");
    std::printf("[phase] spin_only after resident launch\n"); std::fflush(stdout);

    const bool publish_finished = launch_publish_and_wait(
        runtime->input_queues_host[0],
        src_dev,
        static_cast<uint32_t>(bytes),
        publish_status_dev,
        producer_stream,
        200);
    std::printf("[phase] spin_only publish_finished=%d\n", int(publish_finished));
    std::fflush(stdout);

    std::printf("[phase] spin_only before stop_and_join\n"); std::fflush(stdout);
    stop_and_join_resident(control, resident_stream);
    std::printf("[phase] spin_only after stop_and_join\n"); std::fflush(stdout);

    const PersistentDebugState dbg = copy_debug_state(dbg_dev);

    if (!publish_finished) {
        throw std::runtime_error(
            "spin_only overlap failed: producer did not complete while dummy resident kernel was alive");
    }

    check_publish_status(publish_status_dev, "spin_only");

    uint64_t head = 0;
    uint64_t tail = 0;
    copy_queue_state(runtime->input_queues_host[0], &head, &tail);

    std::printf(
        "[phase] spin_only ok loops=%llu head=%llu tail=%llu\n",
        dbg.loop_count,
        static_cast<unsigned long long>(head),
        static_cast<unsigned long long>(tail));
    std::fflush(stdout);
}

void run_queue_read_only_overlap_phase(
    comm::EndpointRuntime* runtime,
    comm::EndpointPersistentControl* control,
    const half* src_dev,
    half* dst_dev,
    size_t bytes,
    int* publish_status_dev,
    PersistentDebugState* dbg_dev,
    cudaStream_t producer_stream,
    cudaStream_t resident_stream,
    const std::vector<half>& host_zero) {
    std::printf("[phase] queue_read_only begin\n"); std::fflush(stdout);

    reset_phase_state(runtime, dst_dev, bytes, publish_status_dev, dbg_dev, host_zero, control);
    std::printf("[phase] queue_read_only after reset\n"); std::fflush(stdout);

    queue_read_only_kernel<<<1, 1, 0, resident_stream>>>(
        runtime->input_queues_host[0],
        control->stop_flag,
        dbg_dev);
    system::runtime::check_cuda(cudaGetLastError(), "queue_read_only_kernel");
    std::printf("[phase] queue_read_only after resident launch\n"); std::fflush(stdout);

    const bool publish_finished = launch_publish_and_wait(
        runtime->input_queues_host[0],
        src_dev,
        static_cast<uint32_t>(bytes),
        publish_status_dev,
        producer_stream,
        200);
    std::printf("[phase] queue_read_only publish_finished=%d\n", int(publish_finished));
    std::fflush(stdout);

    std::printf("[phase] queue_read_only before stop_and_join\n"); std::fflush(stdout);
    stop_and_join_resident(control, resident_stream);
    std::printf("[phase] queue_read_only after stop_and_join\n"); std::fflush(stdout);

    const PersistentDebugState dbg = copy_debug_state(dbg_dev);

    if (!publish_finished) {
        throw std::runtime_error(
            "queue_read_only overlap failed: producer did not complete while resident kernel only read queue state");
    }

    check_publish_status(publish_status_dev, "queue_read_only");

    uint64_t head = 0;
    uint64_t tail = 0;
    copy_queue_state(runtime->input_queues_host[0], &head, &tail);

    std::printf(
        "[phase] queue_read_only ok loops=%llu last_head=%llu last_tail=%llu final_head=%llu final_tail=%llu\n",
        dbg.loop_count,
        dbg.last_head,
        dbg.last_tail,
        static_cast<unsigned long long>(head),
        static_cast<unsigned long long>(tail));
    std::fflush(stdout);
}

void run_scheduler_prime_only_overlap_phase(
    comm::EndpointRuntime* runtime,
    comm::EndpointPersistentControl* control,
    const half* src_dev,
    half* dst_dev,
    size_t bytes,
    int* publish_status_dev,
    PersistentDebugState* dbg_dev,
    cudaStream_t producer_stream,
    cudaStream_t resident_stream,
    const std::vector<half>& host_zero) {
    std::printf("[phase] scheduler_prime_only begin\n"); std::fflush(stdout);

    reset_phase_state(runtime, dst_dev, bytes, publish_status_dev, dbg_dev, host_zero, control);
    std::printf("[phase] scheduler_prime_only after reset\n"); std::fflush(stdout);

    scheduler_prime_only_kernel<<<1, 1, 0, resident_stream>>>(
        runtime->device,
        control->stop_flag,
        dbg_dev);
    system::runtime::check_cuda(cudaGetLastError(), "scheduler_prime_only_kernel");
    std::printf("[phase] scheduler_prime_only after resident launch\n"); std::fflush(stdout);

    const bool publish_finished = launch_publish_and_wait(
        runtime->input_queues_host[0],
        src_dev,
        static_cast<uint32_t>(bytes),
        publish_status_dev,
        producer_stream,
        200);
    std::printf("[phase] scheduler_prime_only publish_finished=%d\n", int(publish_finished));
    std::fflush(stdout);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::printf("[phase] scheduler_prime_only before stop_and_join\n"); std::fflush(stdout);
    stop_and_join_resident(control, resident_stream);
    std::printf("[phase] scheduler_prime_only after stop_and_join\n"); std::fflush(stdout);

    const PersistentDebugState dbg = copy_debug_state(dbg_dev);

    if (!publish_finished) {
        throw std::runtime_error(
            "scheduler_prime_only overlap failed: producer did not complete while resident kernel polled scheduler");
    }

    check_publish_status(publish_status_dev, "scheduler_prime_only");

    uint64_t head = 0;
    uint64_t tail = 0;
    copy_queue_state(runtime->input_queues_host[0], &head, &tail);

    std::printf(
        "[phase] scheduler_prime_only ok loops=%llu no_work=%llu primed=%llu final_head=%llu final_tail=%llu\n",
        dbg.loop_count,
        dbg.no_work_count,
        dbg.primed_count,
        static_cast<unsigned long long>(head),
        static_cast<unsigned long long>(tail));
    std::fflush(stdout);
}

void run_full_persistent_direct_phase(
    comm::EndpointRuntime* runtime,
    comm::EndpointPersistentControl* control,
    const half* src_dev,
    half* dst_dev,
    size_t bytes,
    int* publish_status_dev,
    cudaStream_t producer_stream,
    const std::vector<half>& host_zero,
    const std::vector<half>& host_src) {
    std::printf("[phase] full_persistent_direct begin\n"); std::fflush(stdout);

    system::runtime::check_cuda(
        cudaMemcpy(dst_dev, host_zero.data(), bytes, cudaMemcpyHostToDevice),
        "full_persistent_direct: reset dst");
    system::runtime::check_cuda(
        cudaMemset(publish_status_dev, 0, sizeof(int)),
        "full_persistent_direct: reset publish_status");
    comm::collective::ready_tile_queue_reset(&runtime->input_queues_host[0]);
    comm::endpoint_persistent_control_reset(control);

    publish_single_ready_tile_kernel<<<1, 1, 0, producer_stream>>>(
        runtime->input_queues_host[0],
        src_dev,
        static_cast<uint32_t>(bytes),
        0,
        publish_status_dev);
    system::runtime::check_cuda(
        cudaGetLastError(),
        "full_persistent_direct: publish_single_ready_tile_kernel");

    system::runtime::check_cuda(
        cudaStreamSynchronize(producer_stream),
        "full_persistent_direct: cudaStreamSynchronize(producer_stream)");

    check_publish_status(publish_status_dev, "full_persistent_direct");

    system::runtime::check_cuda(
        comm::launch_endpoint_persistent_kernel_sm90(
            comm::endpoint_runtime_device_handle(runtime),
            control,
            runtime->endpoint.stream),
        "launch_endpoint_persistent_kernel_sm90");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop_and_join_resident(control, runtime->endpoint.stream);

    uint64_t head = 0;
    uint64_t tail = 0;
    copy_queue_state(runtime->input_queues_host[0], &head, &tail);

    if (head != 1 || tail != 1) {
        throw std::runtime_error(
            "full_persistent_direct failed: expected queue state head=1 tail=1 after stop");
    }

    std::vector<half> host_dst(static_cast<size_t>(bytes / sizeof(half)));
    system::runtime::check_cuda(
        cudaMemcpy(host_dst.data(), dst_dev, bytes, cudaMemcpyDeviceToHost),
        "full_persistent_direct: cudaMemcpy(dst_dev -> host_dst)");

    expect_half_vectors_close(
        host_dst,
        host_src,
        "full_persistent_direct");

    std::printf(
        "[phase] full_persistent_direct ok head=%llu tail=%llu\n",
        static_cast<unsigned long long>(head),
        static_cast<unsigned long long>(tail));
    std::fflush(stdout);
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
    PersistentDebugState* dbg_dev = nullptr;
    cudaStream_t producer_stream = nullptr;
    cudaStream_t resident_stream = nullptr;

    bool control_initialized = false;

    try {
        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
        const auto host_src = make_host_pattern(numel);
        const std::vector<half> host_zero(
            static_cast<size_t>(numel),
            __float2half_rn(0.0f));

        std::printf("[smoke] before group_init\n"); std::fflush(stdout);
        comm::group_init(&group, {dev0, dev1}, comm::kEndpointPersistentChunkBytes);

        std::printf("[smoke] after group_init\n"); std::fflush(stdout);
        comm::endpoint_runtime_init(&runtime, &group, 0, 1, 8);

        std::printf("[smoke] after endpoint_runtime_init\n"); std::fflush(stdout);

        system::runtime::set_device(dev0);

        cudaDeviceProp prop{};
        system::runtime::check_cuda(
            cudaGetDeviceProperties(&prop, dev0),
            "cudaGetDeviceProperties(dev0)");
        std::printf(
            "[smoke] concurrentKernels=%d multiProcessorCount=%d\n",
            prop.concurrentKernels,
            prop.multiProcessorCount);
        std::fflush(stdout);

        system::runtime::check_cuda(
            cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags(producer_stream)");

        int least_priority = 0;
        int greatest_priority = 0;
        system::runtime::check_cuda(
            cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
            "cudaDeviceGetStreamPriorityRange");
        std::printf(
            "[smoke] stream priorities: least=%d greatest=%d\n",
            least_priority,
            greatest_priority);
        std::fflush(stdout);

        system::runtime::check_cuda(
            cudaStreamCreateWithPriority(
                &resident_stream,
                cudaStreamNonBlocking,
                least_priority),
            "cudaStreamCreateWithPriority(resident_stream)");

        system::runtime::check_cuda(
            cudaMalloc(&src_dev, bytes),
            "cudaMalloc(src_dev)");
        system::runtime::check_cuda(
            cudaMalloc(&dst_dev, bytes),
            "cudaMalloc(dst_dev)");
        system::runtime::check_cuda(
            cudaMalloc(&publish_status_dev, sizeof(int)),
            "cudaMalloc(publish_status_dev)");
        system::runtime::check_cuda(
            cudaMalloc(&dbg_dev, sizeof(PersistentDebugState)),
            "cudaMalloc(dbg_dev)");

        system::runtime::check_cuda(
            cudaMemcpy(src_dev, host_src.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(host_src -> src_dev)");
        system::runtime::check_cuda(
            cudaMemcpy(dst_dev, host_zero.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(host_zero -> dst_dev)");
        system::runtime::check_cuda(
            cudaMemset(publish_status_dev, 0, sizeof(int)),
            "cudaMemset(publish_status_dev)");
        system::runtime::check_cuda(
            cudaMemset(dbg_dev, 0, sizeof(PersistentDebugState)),
            "cudaMemset(dbg_dev)");

        std::printf("[smoke] after device buffer alloc/init\n"); std::fflush(stdout);

        comm::endpoint_runtime_configure_queue_operation(
            &runtime,
            0,
            1,
            0,
            dst_dev,
            bytes,
            bytes,
            1,
            comm::exec::ChunkOpKind::kReduceAddNoFtzF16,
            1,
            true);

        std::printf("[smoke] after configure_queue_operation\n"); std::fflush(stdout);

        comm::endpoint_persistent_control_init(&control, dev0);
        control_initialized = true;

        run_spin_only_overlap_phase(
            &runtime,
            &control,
            src_dev,
            dst_dev,
            bytes,
            publish_status_dev,
            dbg_dev,
            producer_stream,
            resident_stream,
            host_zero);

        run_queue_read_only_overlap_phase(
            &runtime,
            &control,
            src_dev,
            dst_dev,
            bytes,
            publish_status_dev,
            dbg_dev,
            producer_stream,
            resident_stream,
            host_zero);

        run_scheduler_prime_only_overlap_phase(
            &runtime,
            &control,
            src_dev,
            dst_dev,
            bytes,
            publish_status_dev,
            dbg_dev,
            producer_stream,
            resident_stream,
            host_zero);

        run_full_persistent_direct_phase(
            &runtime,
            &control,
            src_dev,
            dst_dev,
            bytes,
            publish_status_dev,
            producer_stream,
            host_zero,
            host_src);

        if (resident_stream != nullptr) {
            cudaStreamDestroy(resident_stream);
            resident_stream = nullptr;
        }
        if (producer_stream != nullptr) {
            cudaStreamDestroy(producer_stream);
            producer_stream = nullptr;
        }
        if (dbg_dev != nullptr) {
            cudaFree(dbg_dev);
            dbg_dev = nullptr;
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

        if (control_initialized) {
            comm::endpoint_persistent_control_destroy(&control);
            control_initialized = false;
        }

        comm::endpoint_runtime_destroy(&runtime);
        comm::group_destroy(&group);
        return true;
    } catch (...) {
        try {
            if (control_initialized) {
                comm::endpoint_persistent_control_request_stop(&control);
                if (resident_stream != nullptr) {
                    cudaStreamSynchronize(resident_stream);
                }
                cudaStreamSynchronize(runtime.endpoint.stream);
            }
        } catch (...) {
        }

        if (resident_stream != nullptr) {
            cudaStreamDestroy(resident_stream);
        }
        if (producer_stream != nullptr) {
            cudaStreamDestroy(producer_stream);
        }
        if (dbg_dev != nullptr) {
            cudaFree(dbg_dev);
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
        if (control_initialized) {
            comm::endpoint_persistent_control_destroy(&control);
        }

        comm::endpoint_runtime_destroy(&runtime);
        comm::group_destroy(&group);
        throw;
    }
}

} // namespace ooverlap
