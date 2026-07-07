#pragma once

/*
 * Umbrella include for logical TMA multi-GPU transfer planners.
 *
 * Common pointer-free planning helpers live in:
 *   comm/plan/tma_multi_gpu_plan_utils.cuh
 *
 * Collective-specific planners live in:
 *   comm/plan/tma_multi_gpu_allreduce_plan.cuh
 *   comm/plan/tma_multi_gpu_reduce_scatter_plan.cuh
 *   comm/plan/tma_multi_gpu_all_gather_plan.cuh
 */

#include "comm/plan/tma_multi_gpu_allreduce_plan.cuh"
#include "comm/plan/tma_multi_gpu_reduce_scatter_plan.cuh"
#include "comm/plan/tma_multi_gpu_all_gather_plan.cuh"
