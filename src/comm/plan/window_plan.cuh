#pragma once

#include "comm/task/window_task.cuh"

namespace ooverlap {
namespace comm {
namespace plan {

template <int MaxTasks>
struct WindowTaskExecutorPlan {
    static_assert(MaxTasks > 0, "MaxTasks must be > 0");

    int total_tasks = 0;
    int tasks_per_cta = 0;
    task::WindowTask tasks[MaxTasks];
};

template <int MaxTasks>
__host__ __forceinline__ void window_task_executor_plan_clear(
    WindowTaskExecutorPlan<MaxTasks>* plan) {
    if (plan == nullptr) {
        return;
    }

    plan->total_tasks = 0;
    plan->tasks_per_cta = 0;

    for (int i = 0; i < MaxTasks; ++i) {
        plan->tasks[i] = task::WindowTask{};
    }
}

template <int MaxTasks>
__host__ __forceinline__ bool window_task_executor_plan_set(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    int task_idx,
    const task::WindowTask& task) {
    if (plan == nullptr || task_idx < 0 || task_idx >= MaxTasks) {
        return false;
    }

    plan->tasks[task_idx] = task;
    return true;
}

template <int MaxTasks>
__host__ __forceinline__ bool prepend_ready_tasks_to_each_cta(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    int cta_count,
    int* local_ready_signal,
    const int* const* peer_ready_signals,
    int peer_count,
    int epoch,
    int ready_protocol,
    int poll_sleep_cycles) {
    if (plan == nullptr ||
        cta_count <= 0 ||
        peer_count < 0 ||
        epoch <= 0 ||
        local_ready_signal == nullptr) {
        return false;
    }

    if (peer_count > 0 && peer_ready_signals == nullptr) {
        return false;
    }

    const int old_tasks_per_cta =
        plan->tasks_per_cta;

    if (old_tasks_per_cta <= 0 ||
        plan->total_tasks != cta_count * old_tasks_per_cta) {
        return false;
    }

    const int prefix_tasks =
        1 + peer_count;

    const int new_tasks_per_cta =
        old_tasks_per_cta + prefix_tasks;

    const int new_total_tasks =
        cta_count * new_tasks_per_cta;

    if (new_total_tasks > MaxTasks) {
        return false;
    }

    for (int cta = cta_count - 1; cta >= 0; --cta) {
        const int old_base =
            cta * old_tasks_per_cta;

        const int new_base =
            cta * new_tasks_per_cta;

        for (int i = old_tasks_per_cta - 1; i >= 0; --i) {
            plan->tasks[new_base + prefix_tasks + i] =
                plan->tasks[old_base + i];
        }

        plan->tasks[new_base] =
            task::make_ready_publish_task(
                local_ready_signal,
                epoch,
                ready_protocol);

        for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
            plan->tasks[new_base + 1 + peer_idx] =
                task::make_ready_wait_task(
                    peer_ready_signals[peer_idx],
                    epoch,
                    poll_sleep_cycles);
        }
    }

    plan->tasks_per_cta = new_tasks_per_cta;
    plan->total_tasks = new_total_tasks;
    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
