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


/* OOVERLAP_ALL_COLLECTIVES_PLAN_BY_VALUE_V1 */
template <int DstMaxTasks, int SrcMaxTasks>
__host__ __forceinline__ bool window_task_executor_plan_pack(
    const WindowTaskExecutorPlan<SrcMaxTasks>& source,
    WindowTaskExecutorPlan<DstMaxTasks>* destination) {

    destination->total_tasks = source.total_tasks;
    destination->tasks_per_cta = source.tasks_per_cta;

   
    for (int task_idx = 0;
         task_idx < source.total_tasks;
         ++task_idx) {
        const task::WindowTask& source_task =
            source.tasks[task_idx];
    
        task::WindowTask& destination_task =
            destination->tasks[task_idx];
    
        if (source_task.op == task::WindowTaskOp::None) {
            /*
             * Do not copy the complete WindowTask for an empty slot.
             *
             * destination is reused between launches, so we cannot simply
             * continue: an old non-None task may still be stored here.
             */
            destination_task.op = task::WindowTaskOp::None;
            destination_task.cta_mask = 0;
            destination_task.terminal = false;
            continue;
        }
    
        destination_task = source_task;
    }

    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
