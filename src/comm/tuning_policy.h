#pragma once

#include "comm/launch_config.h"

#include "ooverlap/comm.h"

#include <cstddef>

namespace ooverlap {
namespace comm {

enum class TuningPreference : int {
    BestPerformance = 0,
    BestEfficiency = 1,
};

TuningPreference tuning_preference_from_public(
    oo_tuning_mode_t mode);

LaunchConfig select_launch_config_for_allreduce(
    size_t bytes_per_rank,
    TuningPreference preference);

} // namespace comm
} // namespace ooverlap
