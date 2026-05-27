#include "comm/tuning/tuning_policy.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
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

struct PolicyEntry {
    CollectivePlanFor collective = CollectivePlanFor::AllReduce;
    size_t bytes_per_rank = 0;
    double avg_ms = 0.0;
    LaunchConfig config{};
};

struct LoadedPolicy {
    bool attempted = false;
    bool loaded = false;
    std::vector<PolicyEntry> entries;
};

struct SelectionCacheKey {
    int collective = 0;
    size_t bytes_per_rank = 0;
    int preference = 0;

    bool has_max_ctas = false;
    int max_ctas = 0;

    bool has_max_threads = false;
    int max_threads = 0;

    long long tolerance_ppm = 50000;
};

struct SelectionCacheKeyHash {
    size_t operator()(const SelectionCacheKey& key) const {
        size_t h = std::hash<int>{}(key.collective);

        auto mix = [&](size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };

        mix(std::hash<size_t>{}(key.bytes_per_rank));
        mix(std::hash<int>{}(key.preference));
        mix(std::hash<bool>{}(key.has_max_ctas));
        mix(std::hash<int>{}(key.max_ctas));
        mix(std::hash<bool>{}(key.has_max_threads));
        mix(std::hash<int>{}(key.max_threads));
        mix(std::hash<long long>{}(key.tolerance_ppm));

        return h;
    }
};

struct SelectionCacheKeyEqual {
    bool operator()(
        const SelectionCacheKey& a,
        const SelectionCacheKey& b) const {
        return a.collective == b.collective &&
               a.bytes_per_rank == b.bytes_per_rank &&
               a.preference == b.preference &&
               a.has_max_ctas == b.has_max_ctas &&
               a.max_ctas == b.max_ctas &&
               a.has_max_threads == b.has_max_threads &&
               a.max_threads == b.max_threads &&
               a.tolerance_ppm == b.tolerance_ppm;
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
    if (out == nullptr) {
        return false;
    }

    const char* text = std::getenv(name);

    if (text == nullptr || text[0] == '\0') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }

    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        return false;
    }

    *out = static_cast<int>(value);
    return true;
}

bool parse_double_env(
    const char* name,
    double* out) {
    if (out == nullptr) {
        return false;
    }

    const char* text = std::getenv(name);

    if (text == nullptr || text[0] == '\0') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text, &end);

    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }

    if (!(value >= 0.0)) {
        return false;
    }

    *out = value;
    return true;
}

long long tolerance_ppm_from_fraction(double tolerance) {
    if (!(tolerance >= 0.0)) {
        tolerance = 0.05;
    }

    const double scaled = tolerance * 1000000.0;

    if (scaled <= 0.0) {
        return 0;
    }

    if (scaled >= static_cast<double>(std::numeric_limits<long long>::max())) {
        return std::numeric_limits<long long>::max();
    }

    return static_cast<long long>(scaled + 0.5);
}

double tolerance_fraction_from_ppm(long long tolerance_ppm) {
    if (tolerance_ppm <= 0) {
        return 0.0;
    }

    return static_cast<double>(tolerance_ppm) / 1000000.0;
}

long long runtime_tolerance_ppm() {
    double value = 0.10;
    double env_value = 0.0;

    if (parse_double_env("OOVERLAP_TUNING_TOLERANCE", &env_value)) {
        value = env_value;
    }

    return tolerance_ppm_from_fraction(value);
}

bool json_get_string(
    const json& obj,
    const char* key,
    std::string* out) {
    if (out == nullptr || !obj.is_object() || !obj.contains(key)) {
        return false;
    }

    const json& value = obj.at(key);

    if (!value.is_string()) {
        return false;
    }

    *out = value.get<std::string>();
    return true;
}

bool parse_string_double(
    const std::string& text,
    double* out) {
    if (out == nullptr || text.empty()) {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);

    if (errno != 0 || end == text.c_str() || *end != '\0') {
        return false;
    }

    *out = value;
    return true;
}

bool json_get_double(
    const json& obj,
    const char* key,
    double* out) {
    if (out == nullptr || !obj.is_object() || !obj.contains(key)) {
        return false;
    }

    const json& value = obj.at(key);

    if (value.is_number()) {
        *out = value.get<double>();
        return true;
    }

    if (value.is_string()) {
        return parse_string_double(value.get<std::string>(), out);
    }

    return false;
}

bool json_get_int(
    const json& obj,
    const char* key,
    int* out) {
    double value = 0.0;

    if (!json_get_double(obj, key, &value)) {
        return false;
    }

    if (value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value > static_cast<double>(std::numeric_limits<int>::max())) {
        return false;
    }

    *out = static_cast<int>(value);
    return true;
}

bool json_get_size(
    const json& obj,
    const char* key,
    size_t* out) {
    double value = 0.0;

    if (!json_get_double(obj, key, &value)) {
        return false;
    }

    if (value < 0.0 ||
        value > static_cast<double>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    *out = static_cast<size_t>(value);
    return true;
}

bool row_status_ok(const json& row) {
    std::string status;

    if (!json_get_string(row, "status", &status)) {
        return true;
    }

    return status == "ok";
}

bool row_backend_ok(const json& row) {
    std::string backend;

    if (!json_get_string(row, "backend", &backend)) {
        return true;
    }

    return backend == "ooverlap" || backend == "oo";
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

bool kernel_name_to_plan(
    CollectivePlanFor collective,
    const std::string& kernel,
    LaunchConfig* config) {
    if (config == nullptr || kernel == "nccl") {
        return false;
    }

    if (collective == CollectivePlanFor::AllReduce) {
        config->plan_for = CollectivePlanFor::AllReduce;

        if (kernel == "tma_copy" ||
            kernel == "normal" ||
            kernel == "tma") {
            config->plan = CollectivePlanKind(AllReducePlanKind::TmaCopy);
            return true;
        }

        if (kernel == "seq_fast_copy_gmem" ||
            kernel == "seq_fast_gmem" ||
            kernel == "not_fused" ||
            kernel == "fast_gmem_seq") {
            config->plan =
                CollectivePlanKind(AllReducePlanKind::SeqFastCopyGmem);
            return true;
        }

        if (kernel == "overlap_fast_copy_gmem" ||
            kernel == "overlap_fast_gmem" ||
            kernel == "fused" ||
            kernel == "fast_gmem_overlap") {
            config->plan =
                CollectivePlanKind(AllReducePlanKind::OverlapFastCopyGmem);
            return true;
        }
    }

    if (collective == CollectivePlanFor::ReduceScatter) {
        config->plan_for = CollectivePlanFor::ReduceScatter;

        if (kernel == "tma_reduce" ||
            kernel == "normal" ||
            kernel == "tma") {
            config->plan =
                CollectivePlanKind(ReduceScatterPlanKind::TmaReduce);
            return true;
        }

        if (kernel == "seq_fast_add_gmem" ||
            kernel == "fast_add_gmem" ||
            kernel == "seq_fast_gmem" ||
            kernel == "not_fused" ||
            kernel == "fast_gmem_seq") {
            config->plan =
                CollectivePlanKind(ReduceScatterPlanKind::SeqFastAddGmem);
            return true;
        }
    }

    if (collective == CollectivePlanFor::AllGather) {
        config->plan_for = CollectivePlanFor::AllGather;

        if (kernel == "tma_copy" ||
            kernel == "normal" ||
            kernel == "tma") {
            config->plan =
                CollectivePlanKind(AllGatherPlanKind::TmaCopy);
            return true;
        }

        if (kernel == "seq_fast_copy_gmem" ||
            kernel == "seq_fast_gmem" ||
            kernel == "not_fused" ||
            kernel == "fast_gmem_seq") {
            config->plan =
                CollectivePlanKind(AllGatherPlanKind::SeqFastCopyGmem);
            return true;
        }
    }

    return false;
}

bool row_avg_ms(
    const json& row,
    double* out) {
    if (json_get_double(row, "avg_ms", out)) {
        return *out > 0.0;
    }

    double total_ms = 0.0;
    int iters = 0;

    if (json_get_double(row, "total_ms", &total_ms) &&
        json_get_int(row, "iters", &iters) &&
        iters > 0) {
        *out = total_ms / static_cast<double>(iters);
        return *out > 0.0;
    }

    double latency_us = 0.0;

    if (json_get_double(row, "latency_us", &latency_us)) {
        *out = latency_us * 1.0e-3;
        return *out > 0.0;
    }

    return false;
}

bool parse_policy_row(
    const json& row,
    PolicyEntry* out) {
    if (out == nullptr || !row.is_object()) {
        return false;
    }

    if (!row_status_ok(row) || !row_backend_ok(row)) {
        return false;
    }

    std::string collective_name;
    if (!json_get_string(row, "collective", &collective_name)) {
        return false;
    }

    CollectivePlanFor collective{};
    if (!collective_name_to_plan_for(collective_name, &collective)) {
        return false;
    }

    std::string kernel;
    if (!json_get_string(row, "kernel", &kernel)) {
        return false;
    }

    PolicyEntry entry{};
    entry.collective = collective;
    entry.config = LaunchConfig{};

    if (!kernel_name_to_plan(collective, kernel, &entry.config)) {
        return false;
    }

    if (!json_get_size(row, "bytes_per_rank", &entry.bytes_per_rank) &&
        !json_get_size(row, "bytes", &entry.bytes_per_rank)) {
        return false;
    }

    if (!row_avg_ms(row, &entry.avg_ms)) {
        return false;
    }

    json_get_int(row, "max_ctas", &entry.config.max_ctas);
    json_get_int(row, "threads", &entry.config.threads);
    json_get_int(row, "window_chunks", &entry.config.window_chunks);

    if (!json_get_int(row, "chunk_bytes", &entry.config.chunk_bytes)) {
        json_get_int(row, "compile_chunk_bytes", &entry.config.chunk_bytes);
    }

    if (!json_get_int(row, "stage_depth", &entry.config.stage_depth)) {
        if (!json_get_int(row, "compile_stage_depth", &entry.config.stage_depth)) {
            json_get_int(
                row,
                "compile_reduce_stage_depth",
                &entry.config.stage_depth);
        }
    }

    if (!launch_config_valid(entry.config)) {
        return false;
    }

    *out = entry;
    return true;
}

void parse_policy_json_rows(
    const json& rows,
    std::vector<PolicyEntry>* entries) {
    if (entries == nullptr) {
        return;
    }

    if (rows.is_array()) {
        for (const json& row : rows) {
            PolicyEntry entry{};

            if (parse_policy_row(row, &entry)) {
                entries->push_back(entry);
            }
        }

        return;
    }

    PolicyEntry entry{};

    if (parse_policy_row(rows, &entry)) {
        entries->push_back(entry);
    }
}

bool load_policy_file(
    const char* path,
    std::vector<PolicyEntry>* entries) {
    if (path == nullptr || path[0] == '\0' || entries == nullptr) {
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

    const size_t before = entries->size();

    if (root.is_object() && root.contains("results")) {
        parse_policy_json_rows(root.at("results"), entries);
    } else {
        parse_policy_json_rows(root, entries);
    }

    return entries->size() > before;
}

void ensure_policy_loaded() {
    LoadedPolicy& policy = global_policy();

    std::lock_guard<std::mutex> lock(global_policy_mutex());

    if (policy.attempted) {
        return;
    }

    policy.attempted = true;
    policy.entries.clear();

    const char* env_path = std::getenv(kEnvPolicyPath);

    if (load_policy_file(env_path, &policy.entries)) {
        policy.loaded = true;
    } else {
        policy.entries.clear();
        policy.loaded = load_policy_file(kDefaultPolicyPath, &policy.entries);
    }

    if (!policy.loaded) {
        return;
    }

    std::sort(
        policy.entries.begin(),
        policy.entries.end(),
        [](const PolicyEntry& a, const PolicyEntry& b) {
            if (a.collective != b.collective) {
                return static_cast<int>(a.collective) <
                       static_cast<int>(b.collective);
            }

            if (a.bytes_per_rank != b.bytes_per_rank) {
                return a.bytes_per_rank < b.bytes_per_rank;
            }

            if (a.config.max_ctas != b.config.max_ctas) {
                return a.config.max_ctas < b.config.max_ctas;
            }

            if (a.avg_ms != b.avg_ms) {
                return a.avg_ms < b.avg_ms;
            }

            if (a.config.threads != b.config.threads) {
                return a.config.threads < b.config.threads;
            }

            return static_cast<int>(a.config.plan_for) <
                   static_cast<int>(b.config.plan_for);
        });
}

size_t select_policy_size(
    const std::vector<PolicyEntry>& entries,
    CollectivePlanFor collective,
    size_t requested_bytes) {
    bool has_prev = false;
    bool has_next = false;

    size_t prev = 0;
    size_t next = 0;

    for (const PolicyEntry& entry : entries) {
        if (entry.collective != collective) {
            continue;
        }

        const size_t b = entry.bytes_per_rank;

        if (b == requested_bytes) {
            return b;
        }

        if (b < requested_bytes) {
            prev = b;
            has_prev = true;
            continue;
        }

        next = b;
        has_next = true;
        break;
    }

    if (!has_prev && !has_next) {
        return requested_bytes;
    }

    if (!has_prev) {
        return next;
    }

    if (!has_next) {
        return prev;
    }

    const long double prev_ratio =
        static_cast<long double>(requested_bytes) /
        static_cast<long double>(prev);

    const long double next_ratio =
        static_cast<long double>(next) /
        static_cast<long double>(requested_bytes);

    return (prev_ratio <= next_ratio) ? prev : next;
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

LaunchConfig fallback_config(
    CollectivePlanFor collective) {
    LaunchConfig config =
        default_config_for_collective(collective);

    int max_ctas = 0;
    if (parse_positive_int_env("OOVERLAP_MAX_CTAS", &max_ctas)) {
        config.max_ctas = std::min(config.max_ctas, max_ctas);
    }

    int max_threads = 0;
    if (parse_positive_int_env("OOVERLAP_MAX_THREADS", &max_threads)) {
        config.threads = std::min(config.threads, max_threads);
        config.threads = (config.threads / 32) * 32;

        if (config.threads < 32) {
            config.threads = 32;
        }
    }

    if (!launch_config_valid(config)) {
        return default_config_for_collective(collective);
    }

    return config;
}

bool env_filter_accepts(
    const PolicyEntry& entry,
    bool has_max_ctas,
    int env_max_ctas,
    bool has_max_threads,
    int env_max_threads) {
    if (has_max_ctas && entry.config.max_ctas > env_max_ctas) {
        return false;
    }

    if (has_max_threads && entry.config.threads > env_max_threads) {
        return false;
    }

    return true;
}

int plan_index(LaunchConfig config) {
    switch (config.plan_for) {
        case CollectivePlanFor::AllReduce:
            return static_cast<int>(config.plan.allreduce);

        case CollectivePlanFor::ReduceScatter:
            return static_cast<int>(config.plan.reduce_scatter);

        case CollectivePlanFor::AllGather:
            return static_cast<int>(config.plan.all_gather);

        default:
            return 0;
    }
}

bool better_efficiency_choice(
    const PolicyEntry& candidate,
    const PolicyEntry& current) {
    if (candidate.config.max_ctas != current.config.max_ctas) {
        return candidate.config.max_ctas < current.config.max_ctas;
    }

    if (candidate.config.threads != current.config.threads) {
        return candidate.config.threads < current.config.threads;
    }

    if (candidate.avg_ms != current.avg_ms) {
        return candidate.avg_ms < current.avg_ms;
    }

    return plan_index(candidate.config) < plan_index(current.config);
}

bool select_from_candidates(
    const std::vector<PolicyEntry>& candidates,
    TuningPreference preference,
    long long tolerance_ppm,
    PolicyEntry* selected) {
    if (selected == nullptr || candidates.empty()) {
        return false;
    }

    const PolicyEntry* best = &candidates[0];

    for (const PolicyEntry& entry : candidates) {
        if (entry.avg_ms < best->avg_ms) {
            best = &entry;
        }
    }

    if (preference == TuningPreference::BestPerformance) {
        *selected = *best;
        return true;
    }

    const double tolerance =
        tolerance_fraction_from_ppm(tolerance_ppm);

    const double max_allowed =
        best->avg_ms * (1.0 + tolerance);

    const PolicyEntry* efficient = nullptr;

    for (const PolicyEntry& entry : candidates) {
        if (entry.avg_ms > max_allowed) {
            continue;
        }

        if (efficient == nullptr ||
            better_efficiency_choice(entry, *efficient)) {
            efficient = &entry;
        }
    }

    *selected = efficient != nullptr ? *efficient : *best;
    return true;
}

SelectionCacheKey make_selection_cache_key(
    CollectivePlanFor collective,
    size_t bytes_per_rank,
    TuningPreference preference,
    bool has_max_ctas,
    int env_max_ctas,
    bool has_max_threads,
    int env_max_threads,
    long long tolerance_ppm) {
    SelectionCacheKey key{};
    key.collective = static_cast<int>(collective);
    key.bytes_per_rank = bytes_per_rank;
    key.preference = static_cast<int>(preference);
    key.has_max_ctas = has_max_ctas;
    key.max_ctas = has_max_ctas ? env_max_ctas : 0;
    key.has_max_threads = has_max_threads;
    key.max_threads = has_max_threads ? env_max_threads : 0;
    key.tolerance_ppm =
        preference == TuningPreference::BestEfficiency ? tolerance_ppm : 0;
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
    size_t bytes_per_rank,
    TuningPreference preference) {
    int env_max_ctas = 0;
    const bool has_max_ctas =
        parse_positive_int_env("OOVERLAP_MAX_CTAS", &env_max_ctas);

    int env_max_threads = 0;
    const bool has_max_threads =
        parse_positive_int_env("OOVERLAP_MAX_THREADS", &env_max_threads);

    const long long tolerance_ppm =
        runtime_tolerance_ppm();

    const SelectionCacheKey cache_key =
        make_selection_cache_key(
            collective,
            bytes_per_rank,
            preference,
            has_max_ctas,
            env_max_ctas,
            has_max_threads,
            env_max_threads,
            tolerance_ppm);

    LaunchConfig cached{};
    if (selection_cache_lookup(cache_key, &cached)) {
        return cached;
    }

    ensure_policy_loaded();

    LoadedPolicy& policy =
        global_policy();

    LaunchConfig selected_config{};

    if (!policy.loaded || policy.entries.empty()) {
        selected_config = fallback_config(collective);
        selection_cache_store(cache_key, selected_config);
        return selected_config;
    }

    const size_t selected_size =
        select_policy_size(
            policy.entries,
            collective,
            bytes_per_rank);

    std::vector<PolicyEntry> candidates;
    candidates.reserve(64);

    for (const PolicyEntry& entry : policy.entries) {
        if (entry.collective != collective) {
            continue;
        }

        if (entry.bytes_per_rank != selected_size) {
            continue;
        }

        if (!env_filter_accepts(
                entry,
                has_max_ctas,
                env_max_ctas,
                has_max_threads,
                env_max_threads)) {
            continue;
        }

        candidates.push_back(entry);
    }

    PolicyEntry selected{};

    if (!select_from_candidates(
            candidates,
            preference,
            tolerance_ppm,
            &selected)) {
        selected_config = fallback_config(collective);
    } else {
        selected_config = selected.config;
    }

    selection_cache_store(cache_key, selected_config);
    return selected_config;
}

LaunchConfig select_launch_config_for_allreduce(
    size_t bytes_per_rank,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::AllReduce,
        bytes_per_rank,
        preference);
}

LaunchConfig select_launch_config_for_reduce_scatter(
    size_t bytes_per_rank,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::ReduceScatter,
        bytes_per_rank,
        preference);
}

LaunchConfig select_launch_config_for_all_gather(
    size_t bytes_per_rank,
    TuningPreference preference) {
    return select_launch_config_for_collective(
        CollectivePlanFor::AllGather,
        bytes_per_rank,
        preference);
}

} // namespace comm
} // namespace ooverlap
