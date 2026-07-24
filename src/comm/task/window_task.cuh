#pragma once

#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace task {

constexpr int kWindowTaskMaxCtas = 64;

using WindowTaskCtaMask = uint64_t;

constexpr WindowTaskCtaMask kWindowTaskAllCtas = ~WindowTaskCtaMask{0};

/* OOVERLAP_WINDOW_TASK_SIGNAL_COMPACTION_V1 */
enum class WindowTaskOp : uint8_t {
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

struct WindowTask {
    WindowTaskOp op = WindowTaskOp::None;

    WindowTaskCtaMask cta_mask = kWindowTaskAllCtas;

    uint32_t barrier_target = 0;

    const void* src = nullptr;
    void* dst = nullptr;
    
    void* fanout_dsts[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};
    uint8_t fanout_dst_count = 0;
    uint8_t fanout_reduce_scope[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};

    /*
     * total_bytes is the logical span of src/dst. Window indices are absolute
     * over this span:
     *
     *   byte_begin = begin_window * window_chunks * ChunkBytes
     *   byte_end   = end_window   * window_chunks * ChunkBytes
     */
    size_t total_bytes = 0;

    int begin_window = 0;
    int end_window = 0;
    int window_chunks = 0;

    /* Ready-signal state used by ReadyPublish/ReadyWait tasks. */
    int* ready_signal = nullptr;
    const int* ready_wait_signal = nullptr;

    int ready_epoch = 0;
    int ready_protocol = 0;
    int ready_wait_epoch = 0;
    int ready_owner_cta = 0;

    /*
     * terminal means: after this task, the CTA returns and does not execute
     * more tasks in its stripe.
     */
    bool terminal = false;
};

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
    uint32_t barrier_target) {
    WindowTask task{};
    task.op = WindowTaskOp::Barrier;
    task.barrier_target = barrier_target;
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
    task.op = op;
    task.src = src;
    task.dst = dst;
    task.total_bytes = total_bytes;
    task.begin_window = begin_window;
    task.end_window = end_window;
    task.window_chunks = window_chunks;
    task.terminal = terminal;
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
    WindowTask task =
        make_window_task(
            WindowTaskOp::CopyTMAFanout,
            src,
            nullptr,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            terminal);

    if (fanout_dst_count < 0) {
        fanout_dst_count = 0;
    }

    if (fanout_dst_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS) {
        fanout_dst_count = TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS;
    }

    task.fanout_dst_count =
        static_cast<uint8_t>(fanout_dst_count);

    for (int i = 0; i < fanout_dst_count; ++i) {
        task.fanout_dsts[i] =
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
    WindowTask task =
        make_window_task(
            WindowTaskOp::ReduceTMAFanout,
            src,
            nullptr,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            terminal);

    if (fanout_dst_count < 0) {
        fanout_dst_count = 0;
    }

    if (fanout_dst_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS) {
        fanout_dst_count = TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS;
    }

    task.fanout_dst_count =
        static_cast<uint8_t>(fanout_dst_count);

    for (int i = 0; i < fanout_dst_count; ++i) {
        task.fanout_dsts[i] =
            fanout_dsts != nullptr ? fanout_dsts[i] : nullptr;

        /*
         * Keep the scope value as an integer here so window_task.cuh does not
         * need to include tma_reduce.cuh. The fanout pipeline will cast it back
         * to tma::TmaReduceScope when it builds runtime reduce targets.
         */
        task.fanout_reduce_scope[i] =
            static_cast<uint8_t>(
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
    task.ready_signal = ready_signal;
    task.ready_epoch = epoch;
    task.ready_protocol = protocol;
    task.terminal = terminal;
    return task;
}

__host__ __device__ __forceinline__ WindowTask make_ready_wait_task(
    const int* ready_signal,
    int epoch,
    bool terminal = false) {
    WindowTask task{};
    task.op = WindowTaskOp::ReadyWait;
    task.ready_signal = const_cast<int*>(ready_signal);
    task.ready_epoch = epoch;
    task.terminal = terminal;
    return task;
}


/* OOVERLAP_READY_PUBLISH_WAIT_MERGE_PATCH: maker for merged consecutive ReadyPublish + ReadyWait. */
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
    task.ready_signal = publish_signal;
    task.ready_epoch = publish_epoch;
    task.ready_protocol = publish_protocol;
    task.ready_wait_signal = wait_signal;
    task.ready_wait_epoch = wait_epoch;
    task.ready_owner_cta = owner_cta;
    task.terminal = terminal;
    return task;
}

__host__ __device__ __forceinline__ bool window_task_has_work(
    const WindowTask& task) {
    if (task.op == WindowTaskOp::ReadyPublishWait) {
        return task.ready_signal != nullptr &&
               task.ready_epoch > 0 &&
               task.ready_wait_signal != nullptr &&
               task.ready_wait_epoch > 0;
    }

    if (task.op == WindowTaskOp::ReadyPublish ||
        task.op == WindowTaskOp::ReadyWait) {
        return task.ready_signal != nullptr &&
               task.ready_epoch > 0;
    }

    if (task.op == WindowTaskOp::CopyTMAFanout ||
        task.op == WindowTaskOp::ReduceTMAFanout) {
        return task.src != nullptr &&
               task.fanout_dst_count > 0 &&
               task.total_bytes > 0 &&
               task.window_chunks > 0 &&
               task.begin_window < task.end_window;
    }

    return task.op != WindowTaskOp::None &&
           task.src != nullptr &&
           task.dst != nullptr &&
           task.total_bytes > 0 &&
           task.window_chunks > 0 &&
           task.begin_window < task.end_window;
}

} // namespace task
} // namespace comm
} // namespace ooverlap
