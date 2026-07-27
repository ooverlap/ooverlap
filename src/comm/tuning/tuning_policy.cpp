#include "comm/tuning/tuning_policy.h"
// OOVERLAP_POLICY_WITHOUT_DTYPE_AND_TIMINGS_V1

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ooverlap {
namespace comm {
namespace {

using json = nlohmann::json;

constexpr const char* kEnvPolicyPath = "OOVERLAP_TUNING_POLICY";
constexpr const char* kDefaultPolicyPath = "./tma_collective_policy.json";
constexpr const char* kEnvMaxCtas = "OOVERLAP_MAX_CTAS";
constexpr const char* kEnvMaxCtasPerReduceTask =
    "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK";

struct RankedCtaCandidate {
    int max_ctas = 0;
    int max_ctas_per_reduce_task = 0;
};

struct PolicyPoint {
    CollectivePlanFor collective = CollectivePlanFor::AllReduce;
    int world_size = 0;
    size_t bytes = 0;
    std::vector<RankedCtaCandidate> ranked;
};

struct LoadedPolicy {
    bool attempted = false;
    bool loaded = false;
    std::vector<PolicyPoint> points;
};

struct SelectionCacheKey {
    int collective = 0;
    int world_size = 0;
    size_t bytes = 0;
    int preference = 0;

    bool has_max_ctas = false;
    int max_ctas = 0;

    bool has_max_ctas_per_reduce_task = false;
    int max_ctas_per_reduce_task = 0;
};

struct SelectionCacheKeyHash {
    size_t operator()(const SelectionCacheKey& key) const {
        size_t h = std::hash<int>{}(key.collective);

        auto mix = [&](size_t value) {
            h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };

        mix(std::hash<int>{}(key.world_size));
        mix(std::hash<size_t>{}(key.bytes));
        mix(std::hash<int>{}(key.preference));
        mix(std::hash<bool>{}(key.has_max_ctas));
        mix(std::hash<int>{}(key.max_ctas));
        mix(std::hash<bool>{}(key.has_max_ctas_per_reduce_task));
        mix(std::hash<int>{}(key.max_ctas_per_reduce_task));
        return h;
    }
};

struct SelectionCacheKeyEqual {
    bool operator()(
        const SelectionCacheKey& a,
        const SelectionCacheKey& b) const {
        return a.collective == b.collective &&
               a.world_size == b.world_size &&
               a.bytes == b.bytes &&
               a.preference == b.preference &&
               a.has_max_ctas == b.has_max_ctas &&
               a.max_ctas == b.max_ctas &&
               a.has_max_ctas_per_reduce_task ==
                   b.has_max_ctas_per_reduce_task &&
               a.max_ctas_per_reduce_task ==
                   b.max_ctas_per_reduce_task;
    }
};

using SelectionCache =
    std::unordered_map<
        SelectionCacheKey,
        LaunchConfig,
        SelectionCacheKeyHash,
        SelectionCacheKeyEqual>;

LoadedPolicy& global_policy() {
    static LoadedPolicy policy;
    return policy;
}

std::mutex& global_policy_mutex() {
    static std::mutex mutex;
    return mutex;
}

SelectionCache& selection_cache() {
    static SelectionCache cache;
    return cache;
}

std::mutex& selection_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

bool parse_positive_int_env(
    const char* name,
    int* out) {
    if (name == nullptr || out == nullptr) {
        return false;
    }

    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);

    if (errno != 0 ||
        end == text ||
        *end != '\0' ||
        value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }

    *out = static_cast<int>(value);
    return true;
}

bool json_number_as_double(
    const json& value,
    double* out) {
    if (out == nullptr) {
        return false;
    }

    if (value.is_number()) {
        const double parsed = value.get<double>();
        if (!std::isfinite(parsed)) {
            return false;
        }
        *out = parsed;
        return true;
    }

    if (!value.is_string()) {
        return false;
    }

    const std::string text = value.get<std::string>();
    if (text.empty()) {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);

    if (errno != 0 ||
        end == text.c_str() ||
        *end != '\0' ||
        !std::isfinite(parsed)) {
        return false;
    }

    *out = parsed;
    return true;
}

bool json_value_as_int(
    const json& value,
    int* out) {
    double parsed = 0.0;
    if (out == nullptr || !json_number_as_double(value, &parsed)) {
        return false;
    }

    if (parsed < static_cast<double>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(parsed) != parsed) {
        return false;
    }

    *out = static_cast<int>(parsed);
    return true;
}

bool json_value_as_size(
    const json& value,
    size_t* out) {
    double parsed = 0.0;
    if (out == nullptr || !json_number_as_double(value, &parsed)) {
        return false;
    }

    if (parsed < 0.0 ||
        parsed > static_cast<double>(std::numeric_limits<size_t>::max()) ||
        std::floor(parsed) != parsed) {
        return false;
    }

    *out = static_cast<size_t>(parsed);
    return true;
}

bool json_get_int(
    const json& object,
    const char* key,
    int* out) {
    return object.is_object() &&
           key != nullptr &&
           object.contains(key) &&
           json_value_as_int(object.at(key), out);
}

bool json_get_size(
    const json& object,
    const char* key,
    size_t* out) {
    return object.is_object() &&
           key != nullptr &&
           object.contains(key) &&
           json_value_as_size(object.at(key), out);
}

bool json_get_string(
    const json& object,
    const char* key,
    std::string* out) {
    if (out == nullptr ||
        !object.is_object() ||
        key == nullptr ||
        !object.contains(key) ||
        !object.at(key).is_string()) {
        return false;
    }

    *out = object.at(key).get<std::string>();
    return true;
}

bool collective_name_to_plan_for(
    const std::string& name,
    CollectivePlanFor* out) {
    if (out == nullptr) {
        return false;
    }

    if (name == "allreduce" ||
        name == "all_reduce" ||
        name == "all-reduce" ||
        name == "ar") {
        *out = CollectivePlanFor::AllReduce;
        return true;
    }

    if (name == "reduce_scatter" ||
        name == "reduce-scatter" ||
        name == "reducescatter" ||
        name == "rs") {
        *out = CollectivePlanFor::ReduceScatter;
        return true;
    }

    if (name == "all_gather" ||
        name == "all-gather" ||
        name == "allgather" ||
        name == "ag") {
        *out = CollectivePlanFor::AllGather;
        return true;
    }

    return false;
}

bool valid_candidate(
    const RankedCtaCandidate& candidate) {
    return candidate.max_ctas > 0 &&
           candidate.max_ctas <= TMA_TWO_GPU_PEER_MAX_CTAS &&
           candidate.max_ctas_per_reduce_task > 0 &&
           candidate.max_ctas_per_reduce_task <= candidate.max_ctas &&
           candidate.max_ctas_per_reduce_task <=
               TMA_TWO_GPU_PEER_MAX_CTAS;
}

bool parse_ranked_candidate(
    const json& value,
    RankedCtaCandidate* out) {
    if (out == nullptr || !value.is_array() || value.size() != 2) {
        return false;
    }

    RankedCtaCandidate candidate{};

    if (!json_value_as_int(value.at(0), &candidate.max_ctas) ||
        !json_value_as_int(
            value.at(1),
            &candidate.max_ctas_per_reduce_task) ||
        !valid_candidate(candidate)) {
        return false;
    }

    *out = candidate;
    return true;
}

bool same_candidate(
    const RankedCtaCandidate& a,
    const RankedCtaCandidate& b) {
    return a.max_ctas == b.max_ctas &&
           a.max_ctas_per_reduce_task ==
               b.max_ctas_per_reduce_task;
}

bool parse_policy_point(
    const json& value,
    PolicyPoint* out) {
    if (out == nullptr || !value.is_object()) {
        return false;
    }

    PolicyPoint point{};
    std::string collective_name;

    if (!json_get_int(value, "world_size", &point.world_size) ||
        point.world_size <= 0 ||
        !json_get_string(value, "collective", &collective_name) ||
        !collective_name_to_plan_for(collective_name, &point.collective) ||
        (!json_get_size(value, "bytes", &point.bytes) &&
         !json_get_size(value, "bytes_per_rank", &point.bytes)) ||
        point.bytes == 0) {
        return false;
    }

    const json* ranked = nullptr;
    if (value.contains("ranked")) {
        ranked = &value.at("ranked");
    } else if (value.contains("candidates")) {
        ranked = &value.at("candidates");
    }

    if (ranked == nullptr || !ranked->is_array()) {
        return false;
    }

    for (const json& candidate_json : *ranked) {
        RankedCtaCandidate candidate{};
        if (!parse_ranked_candidate(candidate_json, &candidate)) {
            continue;
        }

        bool duplicate = false;
        for (const RankedCtaCandidate& existing : point.ranked) {
            if (same_candidate(existing, candidate)) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            point.ranked.push_back(candidate);
        }
    }

    if (point.ranked.empty()) {
        return false;
    }

    *out = std::move(point);
    return true;
}

void parse_policy_points(
    const json& values,
    std::vector<PolicyPoint>* points) {
    if (points == nullptr) {
        return;
    }

    if (values.is_array()) {
        for (const json& value : values) {
            PolicyPoint point{};
            if (parse_policy_point(value, &point)) {
                points->push_back(std::move(point));
            }
        }
        return;
    }

    PolicyPoint point{};
    if (parse_policy_point(values, &point)) {
        points->push_back(std::move(point));
    }
}

bool load_policy_file(
    const char* path,
    std::vector<PolicyPoint>* points) {
    if (path == nullptr || path[0] == '\0' || points == nullptr) {
        return false;
    }

    std::ifstream file(path);
    if (!file) {
        return false;
    }

    json root;
    try {
        file >> root;
    } catch (...) {
        return false;
    }

    const size_t before = points->size();

    if (root.is_object() && root.contains("entries")) {
        parse_policy_points(root.at("entries"), points);
    } else if (root.is_object() && root.contains("results")) {
        parse_policy_points(root.at("results"), points);
    } else {
        parse_policy_points(root, points);
    }

    return points->size() > before;
}

void ensure_policy_loaded() {
    LoadedPolicy& policy = global_policy();
    std::lock_guard<std::mutex> lock(global_policy_mutex());

    if (policy.attempted) {
        return;
    }

    policy.attempted = true;
    policy.loaded = false;
    policy.points.clear();

    const char* env_path = std::getenv(kEnvPolicyPath);

    if (load_policy_file(env_path, &policy.points)) {
        policy.loaded = true;
    } else {
        policy.points.clear();
        policy.loaded =
            load_policy_file(kDefaultPolicyPath, &policy.points);
    }

    if (!policy.loaded) {
        return;
    }

    std::sort(
        policy.points.begin(),
        policy.points.end(),
        [](const PolicyPoint& a, const PolicyPoint& b) {
            if (a.world_size != b.world_size) {
                return a.world_size < b.world_size;
            }
            if (a.collective != b.collective) {
                return static_cast<int>(a.collective) <
                       static_cast<int>(b.collective);
            }
            return a.bytes < b.bytes;
        });
}

LaunchConfig default_config_for_collective(
    CollectivePlanFor collective) {
    switch (collective) {
        case CollectivePlanFor::AllReduce:
            return make_allreduce_launch_config(
                AllReducePlanKind::TmaCopy);
        case CollectivePlanFor::ReduceScatter:
            return make_reduce_scatter_launch_config(
                ReduceScatterPlanKind::TmaReduce);
        case CollectivePlanFor::AllGather:
            return make_all_gather_launch_config(
                AllGatherPlanKind::TmaCopy);
        default:
            return LaunchConfig{};
    }
}

void apply_runtime_caps(
    LaunchConfig* config,
    bool has_max_ctas,
    int max_ctas,
    bool has_max_ctas_per_reduce_task,
    int max_ctas_per_reduce_task) {
    if (config == nullptr) {
        return;
    }

    if (has_max_ctas) {
        config->max_ctas =
            std::min(
                config->max_ctas,
                std::min(max_ctas, TMA_TWO_GPU_PEER_MAX_CTAS));
    }

    if (has_max_ctas_per_reduce_task) {
        config->max_ctas_per_reduce_task =
            std::min(
                config->max_ctas_per_reduce_task,
                std::min(
                    max_ctas_per_reduce_task,
                    TMA_TWO_GPU_PEER_MAX_CTAS));
    }

    config->max_ctas_per_reduce_task =
        std::min(
            config->max_ctas_per_reduce_task,
            config->max_ctas);
}

LaunchConfig fallback_config(
    CollectivePlanFor collective,
    bool has_max_ctas,
    int max_ctas,
    bool has_max_ctas_per_reduce_task,
    int max_ctas_per_reduce_task) {
    LaunchConfig config = default_config_for_collective(collective);

    apply_runtime_caps(
        &config,
        has_max_ctas,
        max_ctas,
        has_max_ctas_per_reduce_task,
        max_ctas_per_reduce_task);

    if (!launch_config_valid(config)) {
        return default_config_for_collective(collective);
    }

    return config;
}

const PolicyPoint* select_nearest_policy_point(
    const std::vector<PolicyPoint>& points,
    CollectivePlanFor collective,
    int world_size,
    size_t requested_bytes) {
    const PolicyPoint* previous = nullptr;
    const PolicyPoint* next = nullptr;

    for (const PolicyPoint& point : points) {
        if (point.collective != collective ||
            point.world_size != world_size) {
            continue;
        }

        if (point.bytes == requested_bytes) {
            return &point;
        }

        if (point.bytes < requested_bytes) {
            if (previous == nullptr || point.bytes > previous->bytes) {
                previous = &point;
            }
        } else if (next == nullptr || point.bytes < next->bytes) {
            next = &point;
        }
    }

    if (previous == nullptr) {
        return next;
    }
    if (next == nullptr) {
        return previous;
    }

    const long double previous_ratio =
        static_cast<long double>(requested_bytes) /
        static_cast<long double>(previous->bytes);
    const long double next_ratio =
        static_cast<long double>(next->bytes) /
        static_cast<long double>(requested_bytes);

    return previous_ratio <= next_ratio ? previous : next;
}

bool candidate_fits_caps(
    const RankedCtaCandidate& candidate,
    bool has_max_ctas,
    int max_ctas,
    bool has_max_ctas_per_reduce_task,
    int max_ctas_per_reduce_task) {
    if (has_max_ctas && candidate.max_ctas > max_ctas) {
        return false;
    }

    if (has_max_ctas_per_reduce_task &&
        candidate.max_ctas_per_reduce_task >
            max_ctas_per_reduce_task) {
        return false;
    }

    return true;
}

LaunchConfig config_from_policy_point(
    const PolicyPoint& point,
    bool has_max_ctas,
    int max_ctas,
    bool has_max_ctas_per_reduce_task,
    int max_ctas_per_reduce_task) {
    const RankedCtaCandidate* selected = nullptr;

    for (const RankedCtaCandidate& candidate : point.ranked) {
        if (candidate_fits_caps(
                candidate,
                has_max_ctas,
                max_ctas,
                has_max_ctas_per_reduce_task,
                max_ctas_per_reduce_task)) {
            selected = &candidate;
            break;
        }
    }

    /*
     * Prefer an actually measured tuple. If every measured tuple exceeds a
     * runtime cap, clamp the best-ranked tuple as a deterministic fallback.
     */
    if (selected == nullptr && !point.ranked.empty()) {
        selected = &point.ranked.front();
    }

    LaunchConfig config =
        default_config_for_collective(point.collective);

    if (selected != nullptr) {
        config.max_ctas = selected->max_ctas;
        config.max_ctas_per_reduce_task =
            selected->max_ctas_per_reduce_task;
    }

    apply_runtime_caps(
        &config,
        has_max_ctas,
        max_ctas,
        has_max_ctas_per_reduce_task,
        max_ctas_per_reduce_task);

    return config;
}

SelectionCacheKey make_selection_cache_key(
    CollectivePlanFor collective,
    int world_size,
    size_t bytes,
    TuningPreference preference,
    bool has_max_ctas,
    int max_ctas,
    bool has_max_ctas_per_reduce_task,
    int max_ctas_per_reduce_task) {
    SelectionCacheKey key{};
    key.collective = static_cast<int>(collective);
    key.world_size = world_size;
    key.bytes = bytes;
    key.preference = static_cast<int>(preference);
    key.has_max_ctas = has_max_ctas;
    key.max_ctas = has_max_ctas ? max_ctas : 0;
    key.has_max_ctas_per_reduce_task =
        has_max_ctas_per_reduce_task;
    key.max_ctas_per_reduce_task =
        has_max_ctas_per_reduce_task
            ? max_ctas_per_reduce_task
            : 0;
    return key;
}

bool selection_cache_lookup(
    const SelectionCacheKey& key,
    LaunchConfig* out) {
    if (out == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(selection_cache_mutex());
    const auto it = selection_cache().find(key);
    if (it == selection_cache().end()) {
        return false;
    }

    *out = it->second;
    return true;
}

void selection_cache_store(
    const SelectionCacheKey& key,
    LaunchConfig config) {
    std::lock_guard<std::mutex> lock(selection_cache_mutex());
    selection_cache()[key] = config;
}

} // namespace

TuningPreference tuning_preference_from_public(
    oo_tuning_mode_t mode) {
    switch (mode) {
        case OO_TUNING_BEST_EFFICIENCY:
            return TuningPreference::BestEfficiency;
        case OO_TUNING_BEST_PERFORMANCE:
        default:
            return TuningPreference::BestPerformance;
    }
}

LaunchConfig select_launch_config_for_collective(
    CollectivePlanFor collective,
    int world_size,
    size_t bytes,
    TuningPreference preference) {
    int env_max_ctas = 0;
    const bool has_max_ctas =
        parse_positive_int_env(kEnvMaxCtas, &env_max_ctas);

    int env_max_ctas_per_reduce_task = 0;
    const bool has_max_ctas_per_reduce_task =
        parse_positive_int_env(
            kEnvMaxCtasPerReduceTask,
            &env_max_ctas_per_reduce_task);

    const SelectionCacheKey cache_key =
        make_selection_cache_key(
            collective,
            world_size,
            bytes,
            preference,
            has_max_ctas,
            env_max_ctas,
            has_max_ctas_per_reduce_task,
            env_max_ctas_per_reduce_task);

    LaunchConfig cached{};
    if (selection_cache_lookup(cache_key, &cached)) {
        return cached;
    }

    ensure_policy_loaded();
    LoadedPolicy& policy = global_policy();

    LaunchConfig selected =
        fallback_config(
            collective,
            has_max_ctas,
            env_max_ctas,
            has_max_ctas_per_reduce_task,
            env_max_ctas_per_reduce_task);

    if (policy.loaded &&
        !policy.points.empty() &&
        world_size > 0 &&
        bytes > 0) {
        const PolicyPoint* point =
            select_nearest_policy_point(
                policy.points,
                collective,
                world_size,
                bytes);

        if (point != nullptr) {
            LaunchConfig candidate =
                config_from_policy_point(
                    *point,
                    has_max_ctas,
                    env_max_ctas,
                    has_max_ctas_per_reduce_task,
                    env_max_ctas_per_reduce_task);

            if (launch_config_valid(candidate)) {
                selected = candidate;
            }
        }
    }

    selection_cache_store(cache_key, selected);
    return selected;
}

LaunchConfig select_launch_config_for_allreduce(
    int world_size,
    size_t bytes,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::AllReduce,
        world_size,
        bytes,
        preference);
}

LaunchConfig select_launch_config_for_reduce_scatter(
    int world_size,
    size_t bytes,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::ReduceScatter,
        world_size,
        bytes,
        preference);
}

LaunchConfig select_launch_config_for_all_gather(
    int world_size,
    size_t bytes,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::AllGather,
        world_size,
        bytes,
        preference);
}

LaunchConfig select_launch_config_for_collective(
    CollectivePlanFor collective,
    size_t bytes,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        collective,
        0,
        bytes,
        preference);
}

LaunchConfig select_launch_config_for_allreduce(
    size_t bytes,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::AllReduce,
        bytes,
        preference);
}

LaunchConfig select_launch_config_for_reduce_scatter(
    size_t bytes,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::ReduceScatter,
        bytes,
        preference);
}

LaunchConfig select_launch_config_for_all_gather(
    size_t bytes,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::AllGather,
        bytes,
        preference);
}

} // namespace comm
} // namespace ooverlap
