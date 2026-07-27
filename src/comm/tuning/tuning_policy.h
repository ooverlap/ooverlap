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

/*
 * Select a CTA-only launch policy for the nearest measured message size.
 *
 * The compact policy is keyed only by:
 *   - collective
 *   - world_size
 *   - full logical message bytes
 *
 * Each point stores a performance-ranked list of exact pairs:
 *   [max_ctas, max_ctas_per_reduce_task]
 *
 * List order is the performance order; execution times are intentionally not
 * stored or parsed by the runtime policy loader.
 */
LaunchConfig select_launch_config_for_collective(
    CollectivePlanFor collective,
    int world_size,
    size_t bytes,
    TuningPreference preference);

LaunchConfig select_launch_config_for_allreduce(
    int world_size,
    size_t bytes,
    TuningPreference preference);

LaunchConfig select_launch_config_for_reduce_scatter(
    int world_size,
    size_t bytes,
    TuningPreference preference);

LaunchConfig select_launch_config_for_all_gather(
    int world_size,
    size_t bytes,
    TuningPreference preference);

/*
 * Compatibility overloads for older internal callers. They intentionally use
 * no world-size-specific policy and therefore normally fall back to defaults.
 * New collective call sites should use the overloads above.
 */
LaunchConfig select_launch_config_for_collective(
    CollectivePlanFor collective,
    size_t bytes,
    TuningPreference preference);

LaunchConfig select_launch_config_for_allreduce(
    size_t bytes,
    TuningPreference preference);

LaunchConfig select_launch_config_for_reduce_scatter(
    size_t bytes,
    TuningPreference preference);

LaunchConfig select_launch_config_for_all_gather(
    size_t bytes,
    TuningPreference preference);

} // namespace comm
} // namespace ooverlap
