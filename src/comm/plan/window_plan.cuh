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

} // namespace plan
} // namespace comm
} // namespace ooverlap
