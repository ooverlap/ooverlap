#pragma once

#include "comm/params.h"

namespace ooverlap {
namespace comm {

enum class CollectivePlanFor : int {
    AllReduce = 0,
    ReduceScatter = 1,
    AllGather = 2,
};

enum class AllReducePlanKind : int {
    TmaCopy = 0,
    SeqFastCopyGmem = 1,
    OverlapFastCopyGmem = 2,
};

enum class ReduceScatterPlanKind : int {
    TmaReduce = 0,
    SeqFastAddGmem = 1,
};

enum class AllGatherPlanKind : int {
    TmaCopy = 0,
    SeqFastCopyGmem = 1,
};

union CollectivePlanKind {
    AllReducePlanKind allreduce;
    ReduceScatterPlanKind reduce_scatter;
    AllGatherPlanKind all_gather;

    __host__ __device__ constexpr CollectivePlanKind()
        : allreduce(AllReducePlanKind::TmaCopy) {}

    __host__ __device__ constexpr explicit CollectivePlanKind(
        AllReducePlanKind value)
        : allreduce(value) {}

    __host__ __device__ constexpr explicit CollectivePlanKind(
        ReduceScatterPlanKind value)
        : reduce_scatter(value) {}

    __host__ __device__ constexpr explicit CollectivePlanKind(
        AllGatherPlanKind value)
        : all_gather(value) {}
};

struct LaunchConfig {
    int threads = TMA_TWO_GPU_PEER_DEFAULT_THREADS;
    int max_ctas = TMA_TWO_GPU_PEER_DEFAULT_MAX_CTAS;

    /* CTA group size used when independent reduction tasks are split. */
    int max_ctas_per_reduce_task = 8;

    int window_chunks = TMA_TWO_GPU_PEER_DEFAULT_WINDOW_CHUNKS;

    int chunk_bytes = TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES;
    int stage_depth = TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH;

    CollectivePlanFor plan_for = CollectivePlanFor::AllReduce;
    CollectivePlanKind plan{};
};

__host__ __device__ __forceinline__ LaunchConfig default_launch_config() {
    return LaunchConfig{};
}

__host__ __device__ __forceinline__ LaunchConfig make_allreduce_launch_config(
    AllReducePlanKind plan = AllReducePlanKind::TmaCopy) {
    LaunchConfig config{};
    config.plan_for = CollectivePlanFor::AllReduce;
    config.plan = CollectivePlanKind(plan);
    return config;
}

__host__ __device__ __forceinline__ LaunchConfig make_reduce_scatter_launch_config(
    ReduceScatterPlanKind plan = ReduceScatterPlanKind::TmaReduce) {
    LaunchConfig config{};
    config.plan_for = CollectivePlanFor::ReduceScatter;
    config.plan = CollectivePlanKind(plan);
    return config;
}

__host__ __device__ __forceinline__ LaunchConfig make_all_gather_launch_config(
    AllGatherPlanKind plan = AllGatherPlanKind::TmaCopy) {
    LaunchConfig config{};
    config.plan_for = CollectivePlanFor::AllGather;
    config.plan = CollectivePlanKind(plan);
    return config;
}

__host__ __device__ __forceinline__ void set_allreduce_plan(
    LaunchConfig* config,
    AllReducePlanKind plan) {
    if (config == nullptr) {
        return;
    }

    config->plan_for = CollectivePlanFor::AllReduce;
    config->plan = CollectivePlanKind(plan);
}

__host__ __device__ __forceinline__ void set_reduce_scatter_plan(
    LaunchConfig* config,
    ReduceScatterPlanKind plan) {
    if (config == nullptr) {
        return;
    }

    config->plan_for = CollectivePlanFor::ReduceScatter;
    config->plan = CollectivePlanKind(plan);
}

__host__ __device__ __forceinline__ void set_all_gather_plan(
    LaunchConfig* config,
    AllGatherPlanKind plan) {
    if (config == nullptr) {
        return;
    }

    config->plan_for = CollectivePlanFor::AllGather;
    config->plan = CollectivePlanKind(plan);
}

__host__ __device__ __forceinline__ AllReducePlanKind allreduce_plan(
    LaunchConfig config) {
    return config.plan.allreduce;
}

__host__ __device__ __forceinline__ ReduceScatterPlanKind reduce_scatter_plan(
    LaunchConfig config) {
    return config.plan.reduce_scatter;
}

__host__ __device__ __forceinline__ AllGatherPlanKind all_gather_plan(
    LaunchConfig config) {
    return config.plan.all_gather;
}

__host__ __device__ __forceinline__ bool launch_config_valid_common(
    LaunchConfig config) {
    if (config.threads <= 0 || config.threads > 1024) {
        return false;
    }

    if ((config.threads % 32) != 0) {
        return false;
    }

    if (config.max_ctas <= 0 ||
        config.max_ctas > TMA_TWO_GPU_PEER_MAX_CTAS) {
        return false;
    }

    if (config.max_ctas_per_reduce_task <= 0 ||
        config.max_ctas_per_reduce_task >
            TMA_TWO_GPU_PEER_MAX_CTAS) {
        return false;
    }

    if (config.window_chunks <= 0) {
        return false;
    }

    if (config.chunk_bytes < 16) {
        return false;
    }

    if ((config.chunk_bytes % 16) != 0) {
        return false;
    }

    if (config.stage_depth < 2) {
        return false;
    }

    if ((config.stage_depth % 2) != 0) {
        return false;
    }

    return true;
}

__host__ __device__ __forceinline__ bool launch_config_valid_plan(
    LaunchConfig config) {
    switch (config.plan_for) {
        case CollectivePlanFor::AllReduce:
            switch (config.plan.allreduce) {
                case AllReducePlanKind::TmaCopy:
                case AllReducePlanKind::SeqFastCopyGmem:
                    return true;

                case AllReducePlanKind::OverlapFastCopyGmem:
                    return config.max_ctas >= 2;

                default:
                    return false;
            }

        case CollectivePlanFor::ReduceScatter:
            switch (config.plan.reduce_scatter) {
                case ReduceScatterPlanKind::TmaReduce:
                case ReduceScatterPlanKind::SeqFastAddGmem:
                    return true;

                default:
                    return false;
            }

        case CollectivePlanFor::AllGather:
            switch (config.plan.all_gather) {
                case AllGatherPlanKind::TmaCopy:
                case AllGatherPlanKind::SeqFastCopyGmem:
                    return true;

                default:
                    return false;
            }

        default:
            return false;
    }
}

__host__ __device__ __forceinline__ bool launch_config_valid(
    LaunchConfig config) {
    return launch_config_valid_common(config) &&
           launch_config_valid_plan(config);
}

} // namespace comm
} // namespace ooverlap
