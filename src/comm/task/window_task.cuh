#pragma once

#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ooverlap {
namespace comm {
namespace task {

constexpr int kWindowTaskMaxCtas = TMA_TWO_GPU_PEER_MAX_CTAS;

using WindowTaskCtaMask = std::uint32_t;
using WindowTaskByteCount = std::uint32_t;

constexpr WindowTaskCtaMask kWindowTaskAllCtas =
    ~WindowTaskCtaMask{0};
constexpr WindowTaskByteCount kWindowTaskMaxBytes =
    ~WindowTaskByteCount{0};

/* OOVERLAP_WINDOW_TASK_UNION_COMPACTION_V1 */
enum class WindowTaskOp : std::uint8_t {
    None = 0,

    /* TMA load from task.src, then reduce/apply into task.dst. */
    ReduceTMA = 1,

    /* TMA copy task.src -> task.dst over the task window range. */
    CopyTMA = 3,

    /* Fast global-memory copy task.src -> task.dst. */
    CopyFast = 4,

    /* Ready-signal operations. */
    ReadyPublish = 8,
    ReadyWait = 9,
    ReadyPublishWait = 10,

    CopyTMAFanout = 11,
    ReduceTMAFanout = 12,
    Barrier = 13,
};

struct WindowTaskWindowPayload {
    const void* src;
    void* dst;
    WindowTaskByteCount total_bytes;
    int begin_window;
    int end_window;
    int window_chunks;
};

struct WindowTaskFanoutPayload {
    const void* src;
    void* fanout_dsts[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS];
    WindowTaskByteCount total_bytes;
    int begin_window;
    int end_window;
    int window_chunks;
    std::uint8_t fanout_dst_count;
    std::uint8_t fanout_reduce_scope
        [TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS];
};

struct WindowTaskReadyPayload {
    int* ready_signal;
    const int* ready_wait_signal;
    int ready_epoch;
    int ready_protocol;
    int ready_wait_epoch;
    int ready_owner_cta;
};

union WindowTaskPayload {
    WindowTaskWindowPayload window;
    WindowTaskFanoutPayload fanout;
    WindowTaskReadyPayload ready;
    std::uint32_t barrier_target;

    __host__ __device__ constexpr WindowTaskPayload()
        : window{} {}
};

struct WindowTask {
    WindowTaskOp op = WindowTaskOp::None;
    bool terminal = false;
    WindowTaskCtaMask cta_mask = kWindowTaskAllCtas;
    WindowTaskPayload payload{};
};

static_assert(
    std::is_trivially_copyable<WindowTask>::value,
    "WindowTask must remain trivially copyable for by-value kernel launch");
static_assert(
    sizeof(WindowTask) <= 80,
    "WindowTask union compaction unexpectedly exceeds 80 bytes");


/*
 * Static CTA assignment for one lowered WindowTask.
 *
 * The lowering layer will set cta_mask later. Until then, the all-ones default
 * preserves the existing behavior: every launched CTA executes every task in
 * its stripe.
 */
__host__ __device__ __forceinline__ bool window_task_runs_on_cta(
    const WindowTask& task,
    int cta_idx) {
    if (cta_idx < 0 || cta_idx >= kWindowTaskMaxCtas) {
        return false;
    }

    const WindowTaskCtaMask cta_bit =
        WindowTaskCtaMask{1} << static_cast<unsigned int>(cta_idx);

    return (task.cta_mask & cta_bit) != 0;
}

__host__ __device__ __forceinline__ WindowTask make_barrier_task(
    std::uint32_t barrier_target) {
    WindowTask task{};
    task.op = WindowTaskOp::Barrier;
    task.payload.barrier_target = barrier_target;
    return task;
}

__host__ __device__ __forceinline__ WindowTask make_window_task(
    WindowTaskOp op,
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    bool terminal = false) {
    WindowTask task{};

    if (total_bytes >
        static_cast<size_t>(kWindowTaskMaxBytes)) {
        return task;
    }

    task.op = op;
    task.terminal = terminal;
    task.payload.window = WindowTaskWindowPayload{
        src,
        dst,
        static_cast<WindowTaskByteCount>(total_bytes),
        begin_window,
        end_window,
        window_chunks,
    };
    return task;
}

__host__ __device__ __forceinline__ WindowTask make_reduce_tma_task(
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    bool terminal = false) {
    return make_window_task(
        WindowTaskOp::ReduceTMA,
        src,
        dst,
        total_bytes,
        begin_window,
        end_window,
        window_chunks,
        terminal);
}

__host__ __device__ __forceinline__ WindowTask make_copy_tma_task(
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    bool terminal = false) {
    return make_window_task(
        WindowTaskOp::CopyTMA,
        src,
        dst,
        total_bytes,
        begin_window,
        end_window,
        window_chunks,
        terminal);
}


/*
 * OOVERLAP_FANOUT_WINDOW_TASK_MAKERS_PATCH:
 *
 * Fanout task constructors only build WindowTask metadata. They do not execute
 * anything; executor/pipeline support is added separately.
 */
__host__ __device__ __forceinline__ WindowTask make_copy_tma_fanout_task(
    const void* src,
    void* const* fanout_dsts,
    int fanout_dst_count,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    bool terminal = false) {
    WindowTask task{};

    if (total_bytes >
        static_cast<size_t>(kWindowTaskMaxBytes)) {
        return task;
    }

    if (fanout_dst_count < 0) {
        fanout_dst_count = 0;
    }

    if (fanout_dst_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS) {
        fanout_dst_count = TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS;
    }

    task.op = WindowTaskOp::CopyTMAFanout;
    task.terminal = terminal;
    task.payload.fanout = WindowTaskFanoutPayload{};
    task.payload.fanout.src = src;
    task.payload.fanout.total_bytes =
        static_cast<WindowTaskByteCount>(total_bytes);
    task.payload.fanout.begin_window = begin_window;
    task.payload.fanout.end_window = end_window;
    task.payload.fanout.window_chunks = window_chunks;
    task.payload.fanout.fanout_dst_count =
        static_cast<std::uint8_t>(fanout_dst_count);

    for (int i = 0; i < fanout_dst_count; ++i) {
        task.payload.fanout.fanout_dsts[i] =
            fanout_dsts != nullptr ? fanout_dsts[i] : nullptr;
    }

    return task;
}

__host__ __device__ __forceinline__ WindowTask make_reduce_tma_fanout_task(
    const void* src,
    void* const* fanout_dsts,
    const int* fanout_reduce_scope,
    int fanout_dst_count,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    bool terminal = false) {
    WindowTask task{};

    if (total_bytes >
        static_cast<size_t>(kWindowTaskMaxBytes)) {
        return task;
    }

    if (fanout_dst_count < 0) {
        fanout_dst_count = 0;
    }

    if (fanout_dst_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS) {
        fanout_dst_count = TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS;
    }

    task.op = WindowTaskOp::ReduceTMAFanout;
    task.terminal = terminal;
    task.payload.fanout = WindowTaskFanoutPayload{};
    task.payload.fanout.src = src;
    task.payload.fanout.total_bytes =
        static_cast<WindowTaskByteCount>(total_bytes);
    task.payload.fanout.begin_window = begin_window;
    task.payload.fanout.end_window = end_window;
    task.payload.fanout.window_chunks = window_chunks;
    task.payload.fanout.fanout_dst_count =
        static_cast<std::uint8_t>(fanout_dst_count);

    for (int i = 0; i < fanout_dst_count; ++i) {
        task.payload.fanout.fanout_dsts[i] =
            fanout_dsts != nullptr ? fanout_dsts[i] : nullptr;
        task.payload.fanout.fanout_reduce_scope[i] =
            static_cast<std::uint8_t>(
                fanout_reduce_scope != nullptr
                    ? fanout_reduce_scope[i]
                    : 0);
    }

    return task;
}


__host__ __device__ __forceinline__ WindowTask make_copy_fast_task(
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    bool terminal = false) {
    return make_window_task(
        WindowTaskOp::CopyFast,
        src,
        dst,
        total_bytes,
        begin_window,
        end_window,
        window_chunks,
        terminal);
}

__host__ __device__ __forceinline__ WindowTask make_ready_publish_task(
    int* ready_signal,
    int epoch,
    int protocol,
    bool terminal = false) {
    WindowTask task{};
    task.op = WindowTaskOp::ReadyPublish;
    task.terminal = terminal;
    task.payload.ready = WindowTaskReadyPayload{};
    task.payload.ready.ready_signal = ready_signal;
    task.payload.ready.ready_epoch = epoch;
    task.payload.ready.ready_protocol = protocol;
    return task;
}

__host__ __device__ __forceinline__ WindowTask make_ready_wait_task(
    const int* ready_signal,
    int epoch,
    bool terminal = false) {
    WindowTask task{};
    task.op = WindowTaskOp::ReadyWait;
    task.terminal = terminal;
    task.payload.ready = WindowTaskReadyPayload{};
    task.payload.ready.ready_signal =
        const_cast<int*>(ready_signal);
    task.payload.ready.ready_epoch = epoch;
    return task;
}

/* OOVERLAP_READY_PUBLISH_WAIT_MERGE_PATCH */
__host__ __device__ __forceinline__ WindowTask make_ready_publish_wait_task(
    int* publish_signal,
    int publish_epoch,
    int publish_protocol,
    const int* wait_signal,
    int wait_epoch,
    int owner_cta,
    bool terminal = false) {
    WindowTask task{};
    task.op = WindowTaskOp::ReadyPublishWait;
    task.terminal = terminal;
    task.payload.ready = WindowTaskReadyPayload{};
    task.payload.ready.ready_signal = publish_signal;
    task.payload.ready.ready_epoch = publish_epoch;
    task.payload.ready.ready_protocol = publish_protocol;
    task.payload.ready.ready_wait_signal = wait_signal;
    task.payload.ready.ready_wait_epoch = wait_epoch;
    task.payload.ready.ready_owner_cta = owner_cta;
    return task;
}

__host__ __device__ __forceinline__ bool window_task_has_work(
    const WindowTask& task) {
    switch (task.op) {
        case WindowTaskOp::ReadyPublishWait:
            return task.payload.ready.ready_signal != nullptr &&
                   task.payload.ready.ready_epoch > 0 &&
                   task.payload.ready.ready_wait_signal != nullptr &&
                   task.payload.ready.ready_wait_epoch > 0;

        case WindowTaskOp::ReadyPublish:
        case WindowTaskOp::ReadyWait:
            return task.payload.ready.ready_signal != nullptr &&
                   task.payload.ready.ready_epoch > 0;

        case WindowTaskOp::CopyTMAFanout:
        case WindowTaskOp::ReduceTMAFanout:
            return task.payload.fanout.src != nullptr &&
                   task.payload.fanout.fanout_dst_count > 0 &&
                   task.payload.fanout.total_bytes > 0 &&
                   task.payload.fanout.window_chunks > 0 &&
                   task.payload.fanout.begin_window <
                       task.payload.fanout.end_window;

        case WindowTaskOp::ReduceTMA:
        case WindowTaskOp::CopyTMA:
        case WindowTaskOp::CopyFast:
            return task.payload.window.src != nullptr &&
                   task.payload.window.dst != nullptr &&
                   task.payload.window.total_bytes > 0 &&
                   task.payload.window.window_chunks > 0 &&
                   task.payload.window.begin_window <
                       task.payload.window.end_window;

        case WindowTaskOp::Barrier:
            return task.payload.barrier_target > 0;

        case WindowTaskOp::None:
        default:
            return false;
    }
}

} // namespace task
} // namespace comm
} // namespace ooverlap
