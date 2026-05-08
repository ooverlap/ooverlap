#include "test/tma_collective_sweep_2gpu.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ooverlap {
namespace {

using json = nlohmann::json;

constexpr double kDefaultEfficiencyMinBestRatio = 0.95;

std::string key_piece(
    const json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }

    if (value.is_number_integer()) {
        return std::to_string(value.get<int64_t>());
    }

    if (value.is_number_unsigned()) {
        return std::to_string(value.get<uint64_t>());
    }

    return value.dump();
}

std::string make_group_key(
    const json& id,
    bool include_variant) {
    std::ostringstream oss;

    oss << key_piece(id.at("collective"))
        << "|"
        << key_piece(id.at("kernel"))
        << "|"
        << key_piece(id.at("numel"));

    if (include_variant) {
        oss << "|"
            << key_piece(id.at("variant"));
    }

    return oss.str();
}

std::string make_nccl_key(
    const json& id) {
    std::ostringstream oss;

    oss << key_piece(id.at("collective"))
        << "|"
        << key_piece(id.at("numel"));

    return oss.str();
}

std::vector<std::string> read_string_list(
    const json& root,
    const char* key,
    std::vector<std::string> fallback) {
    if (!root.contains(key) || root.at(key).is_null()) {
        return fallback;
    }

    const json& value = root.at(key);

    if (value.is_string()) {
        return {value.get<std::string>()};
    }

    if (!value.is_array()) {
        throw std::invalid_argument(std::string(key) + " must be a string or array");
    }

    std::vector<std::string> out;
    out.reserve(value.size());

    for (const json& item : value) {
        if (!item.is_string()) {
            throw std::invalid_argument(std::string(key) + " entries must be strings");
        }

        out.push_back(item.get<std::string>());
    }

    if (out.empty()) {
        throw std::invalid_argument(std::string(key) + " must not be empty");
    }

    return out;
}

std::vector<int64_t> read_i64_list_required(
    const json& root,
    const char* key) {
    if (!root.contains(key) || root.at(key).is_null()) {
        throw std::invalid_argument(std::string("request must contain ") + key);
    }

    const json& value = root.at(key);
    std::vector<int64_t> out;

    if (value.is_number_integer() || value.is_number_unsigned()) {
        out.push_back(value.get<int64_t>());
    } else if (value.is_array()) {
        out.reserve(value.size());

        for (const json& item : value) {
            if (!item.is_number_integer() && !item.is_number_unsigned()) {
                throw std::invalid_argument(std::string(key) + " entries must be integers");
            }

            out.push_back(item.get<int64_t>());
        }
    } else {
        throw std::invalid_argument(std::string(key) + " must be an integer or array");
    }

    if (out.empty()) {
        throw std::invalid_argument(std::string(key) + " must not be empty");
    }

    for (int64_t x : out) {
        if (x <= 0) {
            throw std::invalid_argument(std::string(key) + " entries must be > 0");
        }
    }

    return out;
}

std::vector<int> read_cta_list(
    const json& root) {
    if (root.contains("ctas") && !root.at("ctas").is_null()) {
        const json& value = root.at("ctas");

        if (!value.is_array()) {
            throw std::invalid_argument("ctas must be an array");
        }

        std::vector<int> out;
        out.reserve(value.size());

        for (const json& item : value) {
            if (!item.is_number_integer() && !item.is_number_unsigned()) {
                throw std::invalid_argument("ctas entries must be integers");
            }

            const int cta = item.get<int>();

            if (cta <= 0) {
                throw std::invalid_argument("ctas entries must be > 0");
            }

            out.push_back(cta);
        }

        if (out.empty()) {
            throw std::invalid_argument("ctas must not be empty");
        }

        return out;
    }

    if (root.contains("max_ctas") && !root.at("max_ctas").is_null()) {
        const int max_ctas = root.at("max_ctas").get<int>();

        if (max_ctas <= 0) {
            throw std::invalid_argument("max_ctas must be > 0");
        }

        return {max_ctas};
    }

    return {1, 2, 4, 8, 16};
}

int read_int_or(
    const json& root,
    const char* key,
    int fallback) {
    if (!root.contains(key) || root.at(key).is_null()) {
        return fallback;
    }

    return root.at(key).get<int>();
}

int read_int_variant_or_root_or(
    const json& variant,
    const json& root,
    const char* key,
    int fallback) {
    if (variant.contains(key) && !variant.at(key).is_null()) {
        return variant.at(key).get<int>();
    }

    return read_int_or(root, key, fallback);
}

json default_variants() {
    return json::array({
        {
            {"name", "2k_x64"},
            {"chunk_bytes", 2 * 1024},
            {"stage_depth", 64},
        },
        {
            {"name", "4k_x32"},
            {"chunk_bytes", 4 * 1024},
            {"stage_depth", 32},
        },
        {
            {"name", "8k_x16"},
            {"chunk_bytes", 8 * 1024},
            {"stage_depth", 16},
        },
        {
            {"name", "16k_x8"},
            {"chunk_bytes", 16 * 1024},
            {"stage_depth", 8},
        },
        {
            {"name", "32k_x4"},
            {"chunk_bytes", 32 * 1024},
            {"stage_depth", 4},
        },
        {
            {"name", "64k_x2"},
            {"chunk_bytes", 64 * 1024},
            {"stage_depth", 2},
        },
        {
            {"name", "100k_x2"},
            {"chunk_bytes", 100 * 1024},
            {"stage_depth", 2},
        },
    });
}

json read_variants(
    const json& root) {
    if (root.contains("variants") && !root.at("variants").is_null()) {
        const json& variants = root.at("variants");

        if (!variants.is_array() || variants.empty()) {
            throw std::invalid_argument("variants must be a non-empty array");
        }

        json out = json::array();

        for (size_t i = 0; i < variants.size(); ++i) {
            json v = variants.at(i);

            if (!v.is_object()) {
                throw std::invalid_argument("variants entries must be objects");
            }

            if (!v.contains("chunk_bytes") || !v.contains("stage_depth")) {
                throw std::invalid_argument(
                    "each variant must contain chunk_bytes and stage_depth");
            }

            if (!v.contains("name") || v.at("name").is_null()) {
                v["name"] =
                    std::string("variant_") + std::to_string(i);
            }

            out.push_back(v);
        }

        return out;
    }

    if (root.contains("chunk_bytes") || root.contains("stage_depth")) {
        const int chunk_bytes =
            read_int_or(root, "chunk_bytes", 8 * 1024);

        const int stage_depth =
            read_int_or(root, "stage_depth", 16);

        return json::array({
            {
                {"name", "requested"},
                {"chunk_bytes", chunk_bytes},
                {"stage_depth", stage_depth},
            },
        });
    }

    return default_variants();
}

bool json_bool_or(
    const json& root,
    const char* key,
    bool fallback) {
    if (!root.contains(key) || root.at(key).is_null()) {
        return fallback;
    }

    return root.at(key).get<bool>();
}

double json_double_or(
    const json& root,
    const char* key,
    double fallback) {
    if (!root.contains(key) || root.at(key).is_null()) {
        return fallback;
    }

    return root.at(key).get<double>();
}

json make_sweep_request(
    const json& request) {
    const std::vector<int64_t> numels =
        read_i64_list_required(request, "numels");

    const std::vector<std::string> collectives =
        read_string_list(
            request,
            "collectives",
            {"allreduce", "reduce_scatter", "all_gather"});

    const std::vector<std::string> kernels =
        read_string_list(
            request,
            "kernels",
            {"tma_copy"});

    const std::vector<int> ctas =
        read_cta_list(request);

    const json variants =
        read_variants(request);

    const bool include_nccl =
        json_bool_or(request, "include_nccl", true);

    json sweep;
    sweep["iters"] = read_int_or(request, "iters", 100);
    sweep["warmup"] = read_int_or(request, "warmup", 20);
    sweep["dev0"] = read_int_or(request, "dev0", 0);
    sweep["dev1"] = read_int_or(request, "dev1", 1);

    json scenarios = json::array();

    for (const std::string& collective : collectives) {
        for (int64_t numel : numels) {
            if (include_nccl) {
                scenarios.push_back({
                    {
                        "id",
                        {
                            {"kind", "nccl_baseline"},
                            {"collective", collective},
                            {"numel", numel},
                        },
                    },
                    {"backend", "nccl"},
                    {"collective", collective},
                    {"numel", numel},
                });
            }

            for (const std::string& kernel : kernels) {
                for (const json& variant : variants) {
                    const std::string variant_name =
                        variant.at("name").get<std::string>();

                    const int chunk_bytes =
                        variant.at("chunk_bytes").get<int>();

                    const int stage_depth =
                        variant.at("stage_depth").get<int>();

                    const int threads =
                        read_int_variant_or_root_or(
                            variant,
                            request,
                            "threads",
                            32);

                    const int window_chunks =
                        read_int_variant_or_root_or(
                            variant,
                            request,
                            "window_chunks",
                            128);

                    for (int max_ctas : ctas) {
                        scenarios.push_back({
                            {
                                "id",
                                {
                                    {"kind", "ooverlap_candidate"},
                                    {"collective", collective},
                                    {"kernel", kernel},
                                    {"numel", numel},
                                    {"variant", variant_name},
                                    {"max_ctas", max_ctas},
                                    {"chunk_bytes", chunk_bytes},
                                    {"stage_depth", stage_depth},
                                    {"threads", threads},
                                    {"window_chunks", window_chunks},
                                },
                            },
                            {"backend", "ooverlap"},
                            {"collective", collective},
                            {"kernel", kernel},
                            {"numel", numel},
                            {"threads", threads},
                            {"max_ctas", max_ctas},
                            {"window_chunks", window_chunks},
                            {"chunk_bytes", chunk_bytes},
                            {"stage_depth", stage_depth},
                        });
                    }
                }
            }
        }
    }

    sweep["scenarios"] = std::move(scenarios);
    return sweep;
}

bool row_ok(
    const json& row) {
    return row.is_object() &&
           row.value("status", std::string()) == "ok";
}

bool is_ooverlap_candidate(
    const json& row) {
    if (!row_ok(row)) {
        return false;
    }

    if (row.value("backend", std::string()) != "ooverlap") {
        return false;
    }

    if (!row.contains("id") || !row.at("id").is_object()) {
        return false;
    }

    return row.at("id").value("kind", std::string()) ==
           "ooverlap_candidate";
}

bool is_nccl_baseline(
    const json& row) {
    if (!row_ok(row)) {
        return false;
    }

    if (row.value("backend", std::string()) != "nccl") {
        return false;
    }

    if (!row.contains("id") || !row.at("id").is_object()) {
        return false;
    }

    return row.at("id").value("kind", std::string()) ==
           "nccl_baseline";
}

double row_gbps(
    const json& row) {
    return row.value("effective_gbps_aggregate_2gpu", 0.0);
}

double row_latency_us(
    const json& row) {
    return row.value("latency_us", std::numeric_limits<double>::infinity());
}

double row_avg_ms(
    const json& row) {
    return row.value("avg_ms", std::numeric_limits<double>::infinity());
}

int row_ctas(
    const json& row) {
    const json& id = row.at("id");

    if (id.contains("max_ctas")) {
        return id.at("max_ctas").get<int>();
    }

    if (row.contains("max_ctas")) {
        return row.at("max_ctas").get<int>();
    }

    return std::numeric_limits<int>::max();
}

double row_efficiency_score(
    const json& row) {
    const int ctas = row_ctas(row);

    if (ctas <= 0 || ctas == std::numeric_limits<int>::max()) {
        return 0.0;
    }

    return row_gbps(row) / static_cast<double>(ctas);
}

bool better_performance(
    const json& a,
    const json& b) {
    const double ga = row_gbps(a);
    const double gb = row_gbps(b);

    if (ga != gb) {
        return ga > gb;
    }

    return row_latency_us(a) < row_latency_us(b);
}

bool better_efficiency_score(
    const json& a,
    const json& b) {
    const double ea = row_efficiency_score(a);
    const double eb = row_efficiency_score(b);

    if (ea != eb) {
        return ea > eb;
    }

    return better_performance(a, b);
}

bool better_near_best_efficiency(
    const json& a,
    const json& b) {
    const int ca = row_ctas(a);
    const int cb = row_ctas(b);

    if (ca != cb) {
        return ca < cb;
    }

    return better_performance(a, b);
}

json compact_row(
    const json& row,
    const json* nccl_row) {
    json out;

    out["id"] = row.value("id", json(nullptr));
    out["avg_ms"] = row.value("avg_ms", json(nullptr));
    out["latency_us"] = row.value("latency_us", json(nullptr));
    out["effective_gbps_per_rank"] =
        row.value("effective_gbps_per_rank", json(nullptr));
    out["effective_gbps_aggregate_2gpu"] =
        row.value("effective_gbps_aggregate_2gpu", json(nullptr));

    if (row.contains("id") && row.at("id").is_object()) {
        const json& id = row.at("id");

        out["variant"] = id.value("variant", std::string());
        out["max_ctas"] = id.value("max_ctas", -1);
        out["chunk_bytes"] = id.value("chunk_bytes", -1);
        out["stage_depth"] = id.value("stage_depth", -1);
        out["threads"] = id.value("threads", -1);
        out["window_chunks"] = id.value("window_chunks", -1);
    }

    out["efficiency_gbps_per_cta"] = row_efficiency_score(row);

    if (nccl_row != nullptr) {
        const double nccl_ms = row_avg_ms(*nccl_row);
        const double this_ms = row_avg_ms(row);

        out["speedup_over_nccl"] =
            (std::isfinite(nccl_ms) && std::isfinite(this_ms) && this_ms > 0.0)
                ? nccl_ms / this_ms
                : 0.0;
    }

    return out;
}

json make_summary_for_groups(
    const json& results,
    bool include_variant,
    double efficiency_min_best_ratio) {
    std::map<std::string, std::vector<json>> groups;
    std::map<std::string, json> nccl_by_key;

    for (const json& row : results) {
        if (is_nccl_baseline(row)) {
            const json& id = row.at("id");
            nccl_by_key[make_nccl_key(id)] = row;
        }
    }

    for (const json& row : results) {
        if (!is_ooverlap_candidate(row)) {
            continue;
        }

        const json& id = row.at("id");
        groups[make_group_key(id, include_variant)].push_back(row);
    }

    json summaries = json::array();

    for (const auto& kv : groups) {
        const std::vector<json>& rows = kv.second;

        if (rows.empty()) {
            continue;
        }

        json best_perf = rows.front();

        for (const json& row : rows) {
            if (better_performance(row, best_perf)) {
                best_perf = row;
            }
        }

        json best_eff_score = rows.front();

        for (const json& row : rows) {
            if (better_efficiency_score(row, best_eff_score)) {
                best_eff_score = row;
            }
        }

        const double best_gbps =
            row_gbps(best_perf);

        const double threshold =
            best_gbps * efficiency_min_best_ratio;

        bool have_near_best = false;
        json near_best_eff;

        for (const json& row : rows) {
            if (row_gbps(row) < threshold) {
                continue;
            }

            if (!have_near_best ||
                better_near_best_efficiency(row, near_best_eff)) {
                near_best_eff = row;
                have_near_best = true;
            }
        }

        if (!have_near_best) {
            near_best_eff = best_perf;
        }

        const json& id = best_perf.at("id");

        const std::string nccl_key =
            make_nccl_key(id);

        const json* nccl_row = nullptr;
        auto nccl_it = nccl_by_key.find(nccl_key);

        if (nccl_it != nccl_by_key.end()) {
            nccl_row = &nccl_it->second;
        }

        std::set<int> ctas;
        std::set<std::string> variants;

        for (const json& row : rows) {
            const json& rid = row.at("id");
            ctas.insert(rid.value("max_ctas", -1));
            variants.insert(rid.value("variant", std::string()));
        }

        json ctas_json = json::array();

        for (int cta : ctas) {
            ctas_json.push_back(cta);
        }

        json variants_json = json::array();

        for (const std::string& variant : variants) {
            variants_json.push_back(variant);
        }

        json summary;

        summary["collective"] = id.at("collective");
        summary["kernel"] = id.at("kernel");
        summary["numel"] = id.at("numel");
        summary["bytes_per_rank"] =
            best_perf.value("bytes_per_rank", json(nullptr));
        summary["summary_scope"] =
            include_variant ? "per_variant" : "across_variants";
        summary["candidate_count"] = rows.size();
        summary["ctas_considered"] = std::move(ctas_json);
        summary["variants_considered"] = std::move(variants_json);
        summary["efficiency_min_best_ratio"] =
            efficiency_min_best_ratio;

        if (include_variant) {
            summary["variant"] = id.at("variant");
        }

        summary["best_performance"] =
            compact_row(best_perf, nccl_row);

        summary["efficiency_near_best"] =
            compact_row(near_best_eff, nccl_row);

        summary["best_efficiency_score"] =
            compact_row(best_eff_score, nccl_row);

        if (nccl_row != nullptr) {
            summary["nccl"] =
                compact_row(*nccl_row, nullptr);
        } else {
            summary["nccl"] = nullptr;
        }

        const double perf_ms = row_avg_ms(best_perf);
        const double near_ms = row_avg_ms(near_best_eff);
        const double perf_gbps = row_gbps(best_perf);
        const double near_gbps = row_gbps(near_best_eff);
        const int perf_ctas = row_ctas(best_perf);
        const int near_ctas = row_ctas(near_best_eff);

        summary["near_best_perf_ratio"] =
            perf_gbps > 0.0 ? near_gbps / perf_gbps : 0.0;

        summary["near_best_latency_delta_pct"] =
            perf_ms > 0.0
                ? 100.0 * (near_ms - perf_ms) / perf_ms
                : 0.0;

        summary["near_best_cta_savings_pct"] =
            perf_ctas > 0
                ? 100.0 *
                      static_cast<double>(perf_ctas - near_ctas) /
                      static_cast<double>(perf_ctas)
                : 0.0;

        summaries.push_back(std::move(summary));
    }

    return summaries;
}

json make_request_metadata(
    const json& request,
    const json& sweep_request) {
    json meta;

    meta["num_scenarios"] =
        sweep_request.at("scenarios").size();

    meta["efficiency_definition"] =
        "lowest max_ctas candidate whose aggregate bandwidth is at least "
        "efficiency_min_best_ratio times best_performance bandwidth";

    meta["best_performance_definition"] =
        "candidate with highest aggregate 2-GPU GB/s, tie-broken by lower latency";

    meta["best_efficiency_score_definition"] =
        "candidate with highest aggregate 2-GPU GB/s divided by max_ctas";

    meta["efficiency_min_best_ratio"] =
        json_double_or(
            request,
            "efficiency_min_best_ratio",
            kDefaultEfficiencyMinBestRatio);

    return meta;
}

} // namespace

std::string benchmark_tma_efficiency_vs_best_2gpu_json(
    const std::string& request_json) {
    json out;

    try {
        const json request =
            json::parse(request_json);

        const double efficiency_min_best_ratio =
            json_double_or(
                request,
                "efficiency_min_best_ratio",
                kDefaultEfficiencyMinBestRatio);

        if (efficiency_min_best_ratio <= 0.0 ||
            efficiency_min_best_ratio > 1.0) {
            throw std::invalid_argument(
                "efficiency_min_best_ratio must be in (0, 1]");
        }

        const json sweep_request =
            make_sweep_request(request);

        const std::string sweep_output_string =
            benchmark_tma_two_gpu_collective_sweep_json(
                sweep_request.dump());

        const json sweep_output =
            json::parse(sweep_output_string);

        out["ok"] =
            sweep_output.value("ok", false);

        out["metadata"] =
            make_request_metadata(request, sweep_request);

        out["raw_results"] =
            sweep_output.value("results", json::array());

        out["sweep_errors"] =
            sweep_output.value("errors", json::array());

        out["summary_across_variants"] =
            make_summary_for_groups(
                out["raw_results"],
                false,
                efficiency_min_best_ratio);

        out["summary_by_variant"] =
            make_summary_for_groups(
                out["raw_results"],
                true,
                efficiency_min_best_ratio);

        if (json_bool_or(request, "include_generated_sweep_request", false)) {
            out["generated_sweep_request"] = sweep_request;
        }

        return out.dump(2);
    } catch (const std::exception& e) {
        out["ok"] = false;
        out["raw_results"] = json::array();
        out["summary_across_variants"] = json::array();
        out["summary_by_variant"] = json::array();
        out["sweep_errors"] = json::array();
        out["errors"] = json::array({
            {
                {"message", e.what()},
            },
        });

        return out.dump(2);
    }
}

} // namespace ooverlap
