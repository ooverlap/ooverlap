#pragma once

#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace task {

enum class WindowTaskOp : uint8_t {
    None = 0,

    /*
     * TMA load from task.src into shared memory, then reduce/apply into
     * task.dst. No inter-CTA window signal.
     */
    ReduceTMA = 1,

    /*
     * Same as ReduceTMA, but publishes task.signal_flags as windows complete.
     * This is the producer side for overlapped copy.
     */
    ReduceTMASignal = 2,

    /*
     * TMA copy task.src -> task.dst over the task window range.
     */
    CopyTMA = 3,

    /*
     * Fast global-memory copy task.src -> task.dst over the task window range.
     */
    CopyFast = 4,

    /*
     * Fast global-memory copy task.src -> task.dst, waiting for the matching
     * per-window producer signal before copying each window.
     */
    CopyFastAfterSignal = 5,

    /*
     * TMA copy task.src -> task.dst and publish task.signal_flags as windows
     * complete.
     */
    CopyTMASignal = 6,

    /*
     * Wait for task.signal_flags window-by-window, then TMA-reduce task.src
     * into task.dst.
     */
    ReduceTMAAfterSignal = 7,
};

struct WindowTask {
    WindowTaskOp op = WindowTaskOp::None;

    const void* src = nullptr;
    void* dst = nullptr;

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

    /*
     * Signal fields are only used by:
     *
     *   ReduceTMASignal
     *   CopyFastAfterSignal
     *
     * signal_base_window is the absolute window index corresponding to
     * signal_flags[0].
     */
    int* signal_flags = nullptr;
    int signal_base_window = 0;

    /*
     * terminal means: after this task, the CTA returns and does not execute
     * more tasks in its stripe.
     */
    bool terminal = false;
};

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

__host__ __device__ __forceinline__ WindowTask make_reduce_tma_signal_task(
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    int* signal_flags,
    int signal_base_window,
    bool terminal = false) {
    WindowTask task =
        make_window_task(
            WindowTaskOp::ReduceTMASignal,
            src,
            dst,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            terminal);

    task.signal_flags = signal_flags;
    task.signal_base_window = signal_base_window;
    return task;
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

__host__ __device__ __forceinline__ WindowTask make_copy_tma_signal_task(
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    int* signal_flags,
    int signal_base_window,
    bool terminal = false) {
    WindowTask task =
        make_window_task(
            WindowTaskOp::CopyTMASignal,
            src,
            dst,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            terminal);

    task.signal_flags = signal_flags;
    task.signal_base_window = signal_base_window;
    return task;
}

__host__ __device__ __forceinline__ WindowTask make_reduce_tma_after_signal_task(
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    const int* signal_flags,
    int signal_base_window,
    bool terminal = false) {
    WindowTask task =
        make_window_task(
            WindowTaskOp::ReduceTMAAfterSignal,
            src,
            dst,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            terminal);

    task.signal_flags = const_cast<int*>(signal_flags);
    task.signal_base_window = signal_base_window;
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

__host__ __device__ __forceinline__ WindowTask make_copy_fast_after_signal_task(
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    const int* signal_flags,
    int signal_base_window,
    bool terminal = false) {
    WindowTask task =
        make_window_task(
            WindowTaskOp::CopyFastAfterSignal,
            src,
            dst,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            terminal);

    /*
     * WindowTask stores non-const signal_flags so one struct can serve both
     * producer and consumer tasks. Consumer execution treats it as const.
     */
    task.signal_flags = const_cast<int*>(signal_flags);
    task.signal_base_window = signal_base_window;
    return task;
}

__host__ __device__ __forceinline__ bool window_task_has_work(
    const WindowTask& task) {
    return task.op != WindowTaskOp::None &&
           task.src != nullptr &&
           task.dst != nullptr &&
           task.total_bytes > 0 &&
           task.window_chunks > 0 &&
           task.begin_window < task.end_window;
}

__host__ __device__ __forceinline__ bool window_task_uses_signal(
    const WindowTask& task) {
    return task.op == WindowTaskOp::ReduceTMASignal ||
           task.op == WindowTaskOp::CopyFastAfterSignal ||
           task.op == WindowTaskOp::CopyTMASignal ||
           task.op == WindowTaskOp::ReduceTMAAfterSignal;
}

} // namespace task
} // namespace comm
} // namespace ooverlap
