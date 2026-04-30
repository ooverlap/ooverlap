#!/usr/bin/env python3
"""
Generate SM90 CUTLASS-3 GEMM-with-signal instances for ooverlap.

This version supports exact stage-count keys and explicit tile schedulers:

  TileM, TileN, TileK,
  ClusterM, ClusterN, ClusterK,
  Stages,
  Mainloop,
  Epilogue,
  Scheduler

Schedulers:
  normal   -> default CUTLASS GemmUniversal scheduler, encoded as C++ type void
  stream_k -> cutlass::gemm::StreamKScheduler

Useful Stream-K workflow:

  # First run gen_config_sm90.py once so *_failed.json contains unsupported_stream_k rows.
  python tool/gen_config_sm90.py \
    --m 16384 --n 8192 --k 8192 \
    --csv ~/csv_out_h100/m16384n8192k8192.gemm.csv \
    --layout packed \
    --top-csv 10 \
    --match-stages exact \
    --timing-mode eager \
    --csv-c-layout any --csv-d-layout any

  # Then generate Stream-K instances from that failed file.
  python tool/generate_instances_sm90.py \
    --from-failed-json configs/m16384n8192k8192_nvidia_h100_nvl_packed_sm90_failed.json \
    --top-failed 20 \
    --keep-curated

  cd build && make -j

Then test direct algo ids, or rerun gen_config_sm90.py with --allow-stream-k.
"""

import argparse
import json
import re
from pathlib import Path

try:
    import torch
except Exception:
    torch = None


MAINLOOP_TYPES = {
    "ws": "cutlass::gemm::KernelTmaWarpSpecialized",
    "pingpong": "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
    "cooperative": "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
}

EPILOGUE_TYPES = {
    "auto": "cutlass::epilogue::collective::EpilogueScheduleAuto",
}

SCHEDULER_TYPES = {
    "normal": "void",
    "stream_k": "cutlass::gemm::StreamKScheduler",
}


def root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def cluster_cpp(cluster):
    cm, cn, ck = [int(x) for x in cluster]
    return f"cute::Shape<cute::_{cm}, cute::_{cn}, cute::_{ck}>"


def stage_cpp(stages):
    if stages is None or str(stages).lower() == "auto" or int(stages) < 0:
        return "cutlass::gemm::collective::StageCountAuto"
    return f"cutlass::gemm::collective::StageCount<{int(stages)}>"


def normalize_scheduler(x):
    if x is None:
        return "normal"
    s = str(x).strip().lower().replace("-", "_")
    if s in ("streamk", "stream_k"):
        return "stream_k"
    return "normal"


def infer_scheduler_from_row(row):
    if "scheduler" in row:
        return normalize_scheduler(row.get("scheduler"))
    if "csv_scheduler" in row:
        return normalize_scheduler(row.get("csv_scheduler"))
    if bool(row.get("is_stream_k", False)):
        return "stream_k"
    op = str(row.get("operation", "")).lower().replace("-", "_")
    err = str(row.get("error", "")).lower().replace("-", "_")
    if re.search(r"stream_?k", op) or re.search(r"stream_?k", err):
        return "stream_k"
    return "normal"


def combo_key(combo):
    stages = combo.get("stages", "auto")
    if stages is None or str(stages).lower() == "auto" or int(stages) < 0:
        stages_key = "auto"
    else:
        stages_key = int(stages)

    return (
        int(combo["tile_m"]),
        int(combo["tile_n"]),
        int(combo["tile_k"]),
        int(combo["cluster"][0]),
        int(combo["cluster"][1]),
        int(combo["cluster"][2]),
        stages_key,
        str(combo["mainloop"]),
        str(combo.get("epilogue", "auto")),
        normalize_scheduler(combo.get("scheduler", "normal")),
    )


def combo_dict_from_key(key):
    return {
        "tile_m": int(key[0]),
        "tile_n": int(key[1]),
        "tile_k": int(key[2]),
        "cluster": [int(key[3]), int(key[4]), int(key[5])],
        "stages": key[6],
        "mainloop": str(key[7]),
        "epilogue": str(key[8]),
        "scheduler": str(key[9]),
    }


def canonical_combo(x):
    scheduler = normalize_scheduler(x.get("scheduler", "normal"))
    if scheduler not in SCHEDULER_TYPES:
        raise ValueError(f"Unsupported scheduler={scheduler}")

    mainloop = str(x["mainloop"])
    if mainloop not in MAINLOOP_TYPES:
        raise ValueError(f"Unsupported mainloop={mainloop}")

    epilogue = str(x.get("epilogue", "auto"))
    if epilogue not in EPILOGUE_TYPES:
        raise ValueError(f"Unsupported epilogue={epilogue}")

    stages = x.get("stages", "auto")
    stages_out = "auto" if stages is None or str(stages).lower() == "auto" or int(stages) < 0 else int(stages)

    return {
        "tile_m": int(x["tile_m"]),
        "tile_n": int(x["tile_n"]),
        "tile_k": int(x["tile_k"]),
        "cluster": [int(x["cluster"][0]), int(x["cluster"][1]), int(x["cluster"][2])],
        "stages": stages_out,
        "mainloop": mainloop,
        "epilogue": epilogue,
        "scheduler": scheduler,
    }


def default_base_ws_combos():
    combos = []
    for tm in [64, 128, 256]:
        for tn in [64, 128, 256]:
            if tm == 64 and tn == 64:
                continue
            if tm == 256 and tn == 256:
                continue
            for tk in [32, 64, 128]:
                for cluster in [(1, 1, 1), (1, 2, 1), (2, 1, 1)]:
                    combos.append({
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "cluster": list(cluster),
                        "stages": "auto",
                        "mainloop": "ws",
                        "epilogue": "auto",
                        "scheduler": "normal",
                    })
    return combos


def curated_sm90_combos(include_pingpong: bool, include_cooperative: bool, include_stream_k: bool):
    combos = []

    for tm, tn, tk, stages in [
        (64, 256, 64, 5),
        (128, 128, 64, 7),
        (128, 256, 64, 4),
        (256, 128, 64, 4),
    ]:
        for cluster in [(1, 1, 1), (1, 2, 1), (2, 1, 1)]:
            combos.append({
                "tile_m": tm, "tile_n": tn, "tile_k": tk,
                "cluster": list(cluster),
                "stages": stages,
                "mainloop": "ws",
                "epilogue": "auto",
                "scheduler": "normal",
            })

    if include_pingpong:
        for tm, tn, tk, stages in [
            (64, 256, 64, 5),
            (128, 128, 64, 6),
            (128, 128, 64, 7),
        ]:
            for cluster in [(1, 2, 1), (2, 1, 1)]:
                combos.append({
                    "tile_m": tm, "tile_n": tn, "tile_k": tk,
                    "cluster": list(cluster),
                    "stages": stages,
                    "mainloop": "pingpong",
                    "epilogue": "auto",
                    "scheduler": "normal",
                })

    if include_cooperative:
        coop_shapes = [
            (128, 128, 64, 6),
            (128, 256, 64, 4),
            (256, 128, 64, 4),
        ]
        for tm, tn, tk, stages in coop_shapes:
            for cluster in [(1, 2, 1), (2, 1, 1)]:
                combos.append({
                    "tile_m": tm, "tile_n": tn, "tile_k": tk,
                    "cluster": list(cluster),
                    "stages": stages,
                    "mainloop": "cooperative",
                    "epilogue": "auto",
                    "scheduler": "normal",
                })

                if include_stream_k:
                    combos.append({
                        "tile_m": tm, "tile_n": tn, "tile_k": tk,
                        "cluster": list(cluster),
                        "stages": stages,
                        "mainloop": "cooperative",
                        "epilogue": "auto",
                        "scheduler": "stream_k",
                    })

    return combos


def _rows_from_json_obj(data, preferred_key):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if preferred_key in data:
            return data.get(preferred_key, [])
        if "failed" in data:
            return data.get("failed", [])
        if "missing" in data:
            return data.get("missing", [])
    return []


def load_rows_json(path: Path, top_n: int, allow_slow: bool, preferred_key: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = _rows_from_json_obj(data, preferred_key)
    out = []
    for r in rows:
        if "tile_m" not in r:
            continue

        op = str(r.get("operation", ""))
        if (not allow_slow) and "_f32_f32_" in op:
            continue

        mainloop = str(r.get("mainloop", "ws"))
        if mainloop not in MAINLOOP_TYPES:
            continue

        stages = r.get("stages", None)
        if stages is None:
            continue

        out.append(canonical_combo({
            "tile_m": r["tile_m"],
            "tile_n": r["tile_n"],
            "tile_k": r["tile_k"],
            "cluster": r["cluster"],
            "stages": stages,
            "mainloop": mainloop,
            "epilogue": r.get("epilogue", "auto"),
            "scheduler": infer_scheduler_from_row(r),
        }))

        if len(out) >= top_n:
            break

    return out


def dedupe(combos):
    by_key = {}
    for c in combos:
        cc = canonical_combo(c)
        by_key[combo_key(cc)] = cc

    out = list(by_key.values())
    out.sort(key=lambda c: (
        str(c["mainloop"]),
        str(c.get("scheduler", "normal")),
        int(c["tile_m"]),
        int(c["tile_n"]),
        int(c["tile_k"]),
        int(c["cluster"][0]),
        int(c["cluster"][1]),
        int(c["cluster"][2]),
        str(c["stages"]),
        str(c["epilogue"]),
    ))
    return out


def cpp_template_args(combo):
    tm = int(combo["tile_m"])
    tn = int(combo["tile_n"])
    tk = int(combo["tile_k"])
    stage = stage_cpp(combo["stages"])
    cluster = cluster_cpp(combo["cluster"])
    mainloop = MAINLOOP_TYPES[combo["mainloop"]]
    epilogue = EPILOGUE_TYPES[combo["epilogue"]]
    scheduler = SCHEDULER_TYPES[normalize_scheduler(combo.get("scheduler", "normal"))]
    return f"{tm}, {tn}, {tk}, {stage}, {cluster}, {mainloop}, {epilogue}, {scheduler}"


def write_algo_dicts(root: Path, candidates):
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    index_dict = {
        combo_key(combo): idx
        for idx, combo in enumerate(candidates)
    }

    if torch is not None:
        torch.save(index_dict, config_dir / "AlgoDictSm90.pt")
    else:
        print("[WARN] torch is not importable; skipping AlgoDictSm90.pt")

    json_items = []
    for combo in candidates:
        key = combo_key(combo)
        json_items.append({
            "algo": index_dict[key],
            **combo_dict_from_key(key),
        })

    with open(config_dir / "AlgoDictSm90.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "SM90/H100 CUTLASS-3 GEMM-with-signal algorithm dictionary",
                "key_format": [
                    "TileM",
                    "TileN",
                    "TileK",
                    "ClusterM",
                    "ClusterN",
                    "ClusterK",
                    "Stages",
                    "Mainloop",
                    "Epilogue",
                    "Scheduler",
                ],
                "algorithms": json_items,
            },
            f,
            indent=2,
        )


def write_signal_instances(root: Path, candidates):
    inc_dir = root / "src" / "inc"
    tiling_dir = root / "src" / "tiling"

    inc_dir.mkdir(parents=True, exist_ok=True)
    tiling_dir.mkdir(parents=True, exist_ok=True)

    inc_path = inc_dir / "signal_instances_sm90.inc"
    table_path = tiling_dir / "signal_tiling_sm90.cuh"

    with open(inc_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by tool/generate_instances_sm90.py\n")
        f.write("// Do not edit by hand.\n\n")

        for combo in candidates:
            args = cpp_template_args(combo)
            f.write(
                "template void cutlass_gemm_signal_sm90<\n"
                f"  {args}\n"
                ">(\n"
                "  int M, int N, int K,\n"
                "  int ReLDN,\n"
                "  int num_segments,\n"
                "  int* CommThr,\n"
                "  half* A, half* B, half* D,\n"
                "  int* MM, int* RA,\n"
                "  bool Monitor,\n"
                "  cudaStream_t stream\n"
                ");\n\n"
            )

    with open(table_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by tool/generate_instances_sm90.py\n")
        f.write("// Do not edit by hand.\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <cuda_runtime.h>\n")
        f.write("#include <cuda_fp16.h>\n")
        f.write("#include <cstdint>\n\n")

        f.write("namespace ooverlap {\n\n")

        f.write(
            "using SignalSm90FuncPtr = void (*)(\n"
            "    int M, int N, int K,\n"
            "    int ReLDN,\n"
            "    int num_segments,\n"
            "    int* CommThr,\n"
            "    half* A, half* B, half* D,\n"
            "    int* MM, int* RA,\n"
            "    bool Monitor,\n"
            "    cudaStream_t stream);\n\n"
        )

        f.write("struct SignalSm90AlgoMeta {\n")
        f.write("  int tile_m;\n")
        f.write("  int tile_n;\n")
        f.write("  int tile_k;\n")
        f.write("  int cluster_m;\n")
        f.write("  int cluster_n;\n")
        f.write("  int cluster_k;\n")
        f.write("  int stages;\n")
        f.write("  const char* mainloop;\n")
        f.write("  const char* epilogue;\n")
        f.write("  const char* scheduler;\n")
        f.write("};\n\n")

        f.write("static SignalSm90FuncPtr signal_sm90_func_table[] = {\n")
        for combo in candidates:
            args = cpp_template_args(combo)
            f.write(f"  &::cutlass_gemm_signal_sm90<{args}>,\n")
        f.write("};\n\n")

        f.write(f"static constexpr int signal_sm90_func_count = {len(candidates)};\n\n")

        f.write("static SignalSm90AlgoMeta signal_sm90_algo_meta[] = {\n")
        for combo in candidates:
            cm, cn, ck = combo["cluster"]
            stages = -1 if str(combo["stages"]).lower() == "auto" else int(combo["stages"])
            scheduler = normalize_scheduler(combo.get("scheduler", "normal"))
            f.write(
                "  {"
                f"{int(combo['tile_m'])}, "
                f"{int(combo['tile_n'])}, "
                f"{int(combo['tile_k'])}, "
                f"{int(cm)}, {int(cn)}, {int(ck)}, "
                f"{stages}, "
                f"\"{combo['mainloop']}\", "
                f"\"{combo['epilogue']}\", "
                f"\"{scheduler}\""
                "},\n"
            )
        f.write("};\n\n")

        f.write("} // namespace ooverlap\n")


def print_summary(candidates):
    print("Generated SM90 signal instances:")
    print(f"  count = {len(candidates)}")
    print("")
    for idx, combo in enumerate(candidates):
        cm, cn, ck = combo["cluster"]
        print(
            f"  algo={idx:03d} "
            f"tile={combo['tile_m']}x{combo['tile_n']}x{combo['tile_k']} "
            f"cluster={cm}x{cn}x{ck} "
            f"stages={combo['stages']} "
            f"mainloop={combo['mainloop']} "
            f"epilogue={combo['epilogue']} "
            f"scheduler={normalize_scheduler(combo.get('scheduler', 'normal'))}"
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default=None)

    ap.add_argument(
        "--preset",
        choices=["curated", "base-ws-only"],
        default="curated",
    )

    ap.add_argument("--include-pingpong", action="store_true", default=True)
    ap.add_argument("--no-include-pingpong", dest="include_pingpong", action="store_false")

    ap.add_argument("--include-cooperative", action="store_true", default=True)
    ap.add_argument("--no-include-cooperative", dest="include_cooperative", action="store_false")

    ap.add_argument("--include-stream-k", action="store_true")
    ap.add_argument("--stream-k-only", action="store_true")

    ap.add_argument("--from-missing-json", type=str, default=None)
    ap.add_argument("--top-missing", type=int, default=20)

    ap.add_argument("--from-failed-json", type=str, default=None)
    ap.add_argument("--top-failed", type=int, default=20)

    ap.add_argument("--keep-base-ws", action="store_true")
    ap.add_argument("--keep-curated", action="store_true")
    ap.add_argument("--allow-slow-or-f32-output-rows", action="store_true")

    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root is not None else root_from_script()

    candidates = []

    if args.from_missing_json is not None:
        candidates.extend(load_rows_json(
            Path(args.from_missing_json).expanduser().resolve(),
            args.top_missing,
            args.allow_slow_or_f32_output_rows,
            preferred_key="missing",
        ))

    if args.from_failed_json is not None:
        candidates.extend(load_rows_json(
            Path(args.from_failed_json).expanduser().resolve(),
            args.top_failed,
            args.allow_slow_or_f32_output_rows,
            preferred_key="failed",
        ))

    if args.keep_base_ws:
        candidates.extend(default_base_ws_combos())

    if args.keep_curated or (args.from_missing_json is None and args.from_failed_json is None):
        if args.preset == "base-ws-only":
            candidates.extend(default_base_ws_combos())
        else:
            candidates.extend(curated_sm90_combos(
                include_pingpong=args.include_pingpong,
                include_cooperative=args.include_cooperative,
                include_stream_k=args.include_stream_k,
            ))

    if args.stream_k_only:
        candidates = [c for c in candidates if normalize_scheduler(c.get("scheduler", "normal")) == "stream_k"]

    candidates = dedupe(candidates)

    if not candidates:
        raise RuntimeError("No SM90 candidates generated.")

    write_algo_dicts(root, candidates)
    write_signal_instances(root, candidates)
    print_summary(candidates)

    print("")
    print("Wrote:")
    print(f"  {root / 'configs' / 'AlgoDictSm90.pt'}")
    print(f"  {root / 'configs' / 'AlgoDictSm90.json'}")
    print(f"  {root / 'src' / 'inc' / 'signal_instances_sm90.inc'}")
    print(f"  {root / 'src' / 'tiling' / 'signal_tiling_sm90.cuh'}")


if __name__ == "__main__":
    main()
