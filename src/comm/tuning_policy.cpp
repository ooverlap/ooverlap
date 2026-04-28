#include "comm/tuning_policy.h"

#include "comm/params.h"

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

struct PolicyEntry {
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
    size_t bytes_per_rank = 0;

    int preference = 0;

    bool has_max_ctas = false;
    int max_ctas = 0;

    bool has_max_threads = false;
    int max_threads = 0;

    /*
     * OOVERLAP_TUNING_TOLERANCE is represented in parts-per-million so that
     * the cache key stays integer-only.
     *
     * 0.05 -> 50000.
     */
    long long tolerance_ppm = 50000;
};

struct SelectionCacheKeyHash {
    size_t operator()(const SelectionCacheKey& key) const {
        size_t h = std::hash<size_t>{}(key.bytes_per_rank);

        auto mix = [&](size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };

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
        return a.bytes_per_rank == b.bytes_per_rank &&
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
    long value = std::strtol(text, &end, 10);

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
    double value = std::strtod(text, &end);

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
    double value = 0.05;

    /*
     * Optional runtime override.
     *
     * Example:
     *   OOVERLAP_TUNING_TOLERANCE=0.03
     */
    double env_value = 0.0;
    if (parse_double_env("OOVERLAP_TUNING_TOLERANCE", &env_value)) {
        value = env_value;
    }

    return tolerance_ppm_from_fraction(value);
}

bool find_json_key(
    const std::string& line,
    const char* key,
    size_t* value_pos) {
    if (value_pos == nullptr) {
        return false;
    }

    const std::string needle =
        std::string("\"") + key + "\"";

    const size_t key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }

    const size_t colon_pos = line.find(':', key_pos + needle.size());
    if (colon_pos == std::string::npos) {
        return false;
    }

    size_t pos = colon_pos + 1;
    while (pos < line.size() &&
           (line[pos] == ' ' || line[pos] == '\t')) {
        ++pos;
    }

    if (pos >= line.size()) {
        return false;
    }

    *value_pos = pos;
    return true;
}

bool json_get_string(
    const std::string& line,
    const char* key,
    std::string* out) {
    if (out == nullptr) {
        return false;
    }

    size_t pos = 0;
    if (!find_json_key(line, key, &pos)) {
        return false;
    }

    if (line.compare(pos, 4, "null") == 0) {
        return false;
    }

    if (line[pos] != '"') {
        return false;
    }

    ++pos;
    std::string value;

    while (pos < line.size()) {
        const char c = line[pos++];

        if (c == '"') {
            *out = value;
            return true;
        }

        if (c == '\\' && pos < line.size()) {
            value.push_back(line[pos++]);
            continue;
        }

        value.push_back(c);
    }

    return false;
}

bool json_get_double(
    const std::string& line,
    const char* key,
    double* out) {
    if (out == nullptr) {
        return false;
    }

    size_t pos = 0;
    if (!find_json_key(line, key, &pos)) {
        return false;
    }

    if (line.compare(pos, 4, "null") == 0) {
        return false;
    }

    const char* begin = line.c_str() + pos;
    char* end = nullptr;

    errno = 0;
    const double value = std::strtod(begin, &end);

    if (errno != 0 || end == begin) {
        return false;
    }

    *out = value;
    return true;
}

bool json_get_int(
    const std::string& line,
    const char* key,
    int* out) {
    double value = 0.0;
    if (!json_get_double(line, key, &value)) {
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
    const std::string& line,
    const char* key,
    size_t* out) {
    double value = 0.0;
    if (!json_get_double(line, key, &value)) {
        return false;
    }

    if (value < 0.0 ||
        value > static_cast<double>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    *out = static_cast<size_t>(value);
    return true;
}

bool kernel_name_to_plan_kind(
    const std::string& name,
    AllreducePlanKind* out) {
    if (out == nullptr) {
        return false;
    }

    if (name == "nccl") {
        return false;
    }

    if (name == "tma_copy" ||
        name == "normal" ||
        name == "tma") {
        *out = AllreducePlanKind::TmaCopy;
        return true;
    }

    if (name == "seq_fast_gmem" ||
        name == "not_fused" ||
        name == "fast_gmem_seq") {
        *out = AllreducePlanKind::SeqFastGmem;
        return true;
    }

    if (name == "overlap_fast_gmem" ||
        name == "fused" ||
        name == "fast_gmem_overlap") {
        *out = AllreducePlanKind::OverlapFastGmem;
        return true;
    }

    return false;
}

bool parse_policy_line(
    const std::string& line,
    PolicyEntry* out) {
    if (out == nullptr || line.empty()) {
        return false;
    }

    std::string kernel;
    if (!json_get_string(line, "kernel", &kernel)) {
        return false;
    }

    AllreducePlanKind kind{};
    if (!kernel_name_to_plan_kind(kernel, &kind)) {
        return false;
    }

    PolicyEntry entry{};
    entry.config.plan_kind = kind;

    if (!json_get_size(line, "bytes_per_rank", &entry.bytes_per_rank)) {
        return false;
    }

    if (!json_get_double(line, "avg_ms", &entry.avg_ms)) {
        return false;
    }

    if (!(entry.avg_ms > 0.0)) {
        return false;
    }

    if (!json_get_int(line, "max_ctas", &entry.config.max_ctas)) {
        return false;
    }

    if (!json_get_int(line, "threads", &entry.config.threads)) {
        return false;
    }

    if (!json_get_int(line, "window_chunks", &entry.config.window_chunks)) {
        return false;
    }

    if (!json_get_int(line, "chunk_bytes", &entry.config.chunk_bytes)) {
        json_get_int(
            line,
            "compile_chunk_bytes",
            &entry.config.chunk_bytes);
    }

    if (!json_get_int(line, "stage_depth", &entry.config.stage_depth)) {
        json_get_int(
            line,
            "compile_reduce_stage_depth",
            &entry.config.stage_depth);
    }

    if (!launch_config_valid(entry.config)) {
        return false;
    }

    if (entry.config.plan_kind == AllreducePlanKind::OverlapFastGmem &&
        !launch_config_valid_for_overlap(entry.config)) {
        return false;
    }

    *out = entry;
    return true;
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

    std::string line;
    size_t loaded = 0;

    while (std::getline(file, line)) {
        PolicyEntry entry{};

        if (parse_policy_line(line, &entry)) {
            entries->push_back(entry);
            ++loaded;
        }
    }

    return loaded > 0;
}

void ensure_policy_loaded() {
    LoadedPolicy& policy = global_policy();

    std::lock_guard<std::mutex> lock(global_policy_mutex());

    if (policy.attempted) {
        return;
    }

    policy.attempted = true;

    const char* env_path = std::getenv("OOVERLAP_TUNING_POLICY");

    if (load_policy_file(env_path, &policy.entries)) {
        policy.loaded = true;
    } else if (load_policy_file(
                   "results/tma_allreduce_policy.jsonl",
                   &policy.entries)) {
        policy.loaded = true;
    } else if (load_policy_file(
                   "results/tma_allreduce_best_configs.jsonl",
                   &policy.entries)) {
        policy.loaded = true;
    } else if (load_policy_file(
                   "results/tma_allreduce_sweep.jsonl",
                   &policy.entries)) {
        policy.loaded = true;
    } else {
        policy.loaded = false;
    }

    if (policy.loaded) {
        std::sort(
            policy.entries.begin(),
            policy.entries.end(),
            [](const PolicyEntry& a, const PolicyEntry& b) {
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
                return static_cast<int>(a.config.plan_kind) <
                       static_cast<int>(b.config.plan_kind);
            });
    }
}

size_t select_policy_size(
    const std::vector<PolicyEntry>& entries,
    size_t requested_bytes) {
    if (entries.empty()) {
        return requested_bytes;
    }

    size_t prev = 0;
    size_t next = 0;

    for (const PolicyEntry& entry : entries) {
        const size_t b = entry.bytes_per_rank;

        if (b == requested_bytes) {
            return b;
        }

        if (b < requested_bytes) {
            prev = b;
            continue;
        }

        next = b;
        break;
    }

    if (prev == 0) {
        return entries.front().bytes_per_rank;
    }

    if (next == 0) {
        return entries.back().bytes_per_rank;
    }

    /*
     * Choose nearest size in multiplicative/log distance.
     *
     * This avoids pathological jumps:
     *
     *   requested = 32.35 MiB
     *
     * should map to 32 MiB, not 64 MiB.
     */
    const long double prev_ratio =
        static_cast<long double>(requested_bytes) /
        static_cast<long double>(prev);

    const long double next_ratio =
        static_cast<long double>(next) /
        static_cast<long double>(requested_bytes);

    if (prev_ratio <= next_ratio) {
        return prev;
    }

    return next;
}

LaunchConfig fallback_config() {
    LaunchConfig config{};

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
        return LaunchConfig{};
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

    return static_cast<int>(candidate.config.plan_kind) <
           static_cast<int>(current.config.plan_kind);
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

    const double tolerance = tolerance_fraction_from_ppm(tolerance_ppm);
    const double max_allowed = best->avg_ms * (1.0 + tolerance);

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

    if (efficient == nullptr) {
        *selected = *best;
    } else {
        *selected = *efficient;
    }

    return true;
}

SelectionCacheKey make_selection_cache_key(
    size_t bytes_per_rank,
    TuningPreference preference,
    bool has_max_ctas,
    int env_max_ctas,
    bool has_max_threads,
    int env_max_threads,
    long long tolerance_ppm) {
    SelectionCacheKey key{};
    key.bytes_per_rank = bytes_per_rank;
    key.preference = static_cast<int>(preference);
    key.has_max_ctas = has_max_ctas;
    key.max_ctas = has_max_ctas ? env_max_ctas : 0;
    key.has_max_threads = has_max_threads;
    key.max_threads = has_max_threads ? env_max_threads : 0;
    key.tolerance_ppm =
        (preference == TuningPreference::BestEfficiency) ? tolerance_ppm : 0;
    return key;
}

bool selection_cache_lookup(
    const SelectionCacheKey& key,
    LaunchConfig* out) {
    if (out == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(selection_cache_mutex());

    const SelectionCache& cache = selection_cache();
    const auto it = cache.find(key);

    if (it == cache.end()) {
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

LaunchConfig select_launch_config_for_allreduce(
    size_t bytes_per_rank,
    TuningPreference preference) {
    int env_max_ctas = 0;
    const bool has_max_ctas =
        parse_positive_int_env("OOVERLAP_MAX_CTAS", &env_max_ctas);

    int env_max_threads = 0;
    const bool has_max_threads =
        parse_positive_int_env("OOVERLAP_MAX_THREADS", &env_max_threads);

    const long long tolerance_ppm = runtime_tolerance_ppm();

    const SelectionCacheKey cache_key =
        make_selection_cache_key(
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

    LoadedPolicy& policy = global_policy();

    LaunchConfig selected_config{};

    if (!policy.loaded || policy.entries.empty()) {
        selected_config = fallback_config();
        selection_cache_store(cache_key, selected_config);
        return selected_config;
    }

    const size_t selected_size =
        select_policy_size(policy.entries, bytes_per_rank);

    std::vector<PolicyEntry> candidates;
    candidates.reserve(64);

    for (const PolicyEntry& entry : policy.entries) {
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
        selected_config = fallback_config();
    } else {
        selected_config = selected.config;
    }

    selection_cache_store(cache_key, selected_config);

    return selected_config;
}

} // namespace comm
} // namespace ooverlap
