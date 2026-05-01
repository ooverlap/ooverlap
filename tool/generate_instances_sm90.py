#!/usr/bin/env python3
"""
Generate SM90 CUTLASS-3 GEMM instances for ooverlap.

This generator now emits ONE shared algorithm dictionary and TWO generated
dispatch tables:

  configs/AlgoDictSm90.{json,pt}

  signal path:
    src/inc/signal_instances_sm90.inc
    src/tiling/signal_tiling_sm90.cuh

  plain GEMM path:
    src/inc/plain_instances_sm90.inc
    src/tiling/plain_tiling_sm90.cuh

The important bit: plain GEMM and signal/reorder GEMM use the same algo ids.
So an algo picked by tool/gen_config_plain_sm90.py can later be reused by the
signal/reorder path without translating ids.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def cluster_cpp(cluster: Iterable[int]) -> str:
    cm, cn, ck = [int(x) for x in cluster]
    return f"cute::Shape<cute::_{cm}, cute::_{cn}, cute::_{ck}>"


def stage_cpp(stages: Any) -> str:
    if stages is None or str(stages).lower() == "auto" or int(stages) < 0:
        return "cutlass::gemm::collective::StageCountAuto"
    return f"cutlass::gemm::collective::StageCount<{int(stages)}>"


def normalize_scheduler(x: Any) -> str:
    if x is None:
        return "normal"
    s = str(x).strip().lower().replace("-", "_")
    if s in ("streamk", "stream_k", "stream"):
        return "stream_k"
    return "normal"


def infer_scheduler_from_row(row: Dict[str, Any]) -> str:
    for key in ("scheduler", "csv_scheduler"):
        if key in row:
            return normalize_scheduler(row.get(key))

    if bool(row.get("is_stream_k", False)):
        return "stream_k"

    op = str(row.get("operation", "")).lower().replace("-", "_")
    err = str(row.get("error", "")).lower().replace("-", "_")
    match = str(row.get("match", "")).lower().replace("-", "_")
    reason = str(row.get("reason", "")).lower().replace("-", "_")

    haystack = " ".join([op, err, match, reason])
    if re.search(r"stream_?k", haystack):
        return "stream_k"

    return "normal"


def normalize_stages(stages: Any) -> Any:
    if stages is None or str(stages).lower() == "auto" or int(stages) < 0:
        return "auto"
    return int(stages)


def combo_key(combo: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        int(combo["tile_m"]),
        int(combo["tile_n"]),
        int(combo["tile_k"]),
        int(combo["cluster"][0]),
        int(combo["cluster"][1]),
        int(combo["cluster"][2]),
        normalize_stages(combo.get("stages", "auto")),
        str(combo["mainloop"]),
        str(combo.get("epilogue", "auto")),
        normalize_scheduler(combo.get("scheduler", "normal")),
    )


def combo_dict_from_key(key: Tuple[Any, ...]) -> Dict[str, Any]:
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


def canonical_combo(x: Dict[str, Any]) -> Dict[str, Any]:
    scheduler = normalize_scheduler(x.get("scheduler", "normal"))
    if scheduler not in SCHEDULER_TYPES:
        raise ValueError(f"Unsupported scheduler={scheduler}")

    mainloop = str(x["mainloop"])
    if mainloop not in MAINLOOP_TYPES:
        raise ValueError(f"Unsupported mainloop={mainloop}")

    epilogue = str(x.get("epilogue", "auto"))
    if epilogue not in EPILOGUE_TYPES:
        raise ValueError(f"Unsupported epilogue={epilogue}")

    return {
        "tile_m": int(x["tile_m"]),
        "tile_n": int(x["tile_n"]),
        "tile_k": int(x["tile_k"]),
        "cluster": [int(x["cluster"][0]), int(x["cluster"][1]), int(x["cluster"][2])],
        "stages": normalize_stages(x.get("stages", "auto")),
        "mainloop": mainloop,
        "epilogue": epilogue,
        "scheduler": scheduler,
    }


def default_base_ws_combos() -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
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


def curated_sm90_combos(
    include_pingpong: bool,
    include_cooperative: bool,
    include_stream_k: bool,
) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []

    # Fixed-stage WS sanity set.
    for tm, tn, tk, stages in [
        (64, 256, 64, 5),
        (128, 128, 64, 7),
        (128, 256, 64, 4),
        (256, 128, 64, 4),
    ]:
        for cluster in [(1, 1, 1), (1, 2, 1), (2, 1, 1)]:
            combos.append({
                "tile_m": tm,
                "tile_n": tn,
                "tile_k": tk,
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
                    "tile_m": tm,
                    "tile_n": tn,
                    "tile_k": tk,
                    "cluster": list(cluster),
                    "stages": stages,
                    "mainloop": "pingpong",
                    "epilogue": "auto",
                    "scheduler": "normal",
                })

    if include_cooperative:
        for tm, tn, tk, stages in [
            (128, 128, 64, 6),
            (128, 256, 64, 4),
            (256, 128, 64, 4),
        ]:
            for cluster in [(1, 2, 1), (2, 1, 1)]:
                combos.append({
                    "tile_m": tm,
                    "tile_n": tn,
                    "tile_k": tk,
                    "cluster": list(cluster),
                    "stages": stages,
                    "mainloop": "cooperative",
                    "epilogue": "auto",
                    "scheduler": "normal",
                })

                if include_stream_k:
                    combos.append({
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "cluster": list(cluster),
                        "stages": stages,
                        "mainloop": "cooperative",
                        "epilogue": "auto",
                        "scheduler": "stream_k",
                    })

    return combos


def _rows_from_json_obj(data: Any, preferred_key: str) -> List[Dict[str, Any]]:
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


def load_rows_json(
    path: Path,
    top_n: int,
    allow_slow: bool,
    preferred_key: str,
) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = _rows_from_json_obj(data, preferred_key)
    out: List[Dict[str, Any]] = []

    for row in rows:
        if "tile_m" not in row:
            continue

        op = str(row.get("operation", ""))
        if (not allow_slow) and "_f32_f32_" in op:
            continue

        mainloop = str(row.get("mainloop", "ws"))
        if mainloop not in MAINLOOP_TYPES:
            continue

        stages = row.get("stages", None)
        if stages is None:
            continue

        out.append(canonical_combo({
            "tile_m": row["tile_m"],
            "tile_n": row["tile_n"],
            "tile_k": row["tile_k"],
            "cluster": row["cluster"],
            "stages": stages,
            "mainloop": mainloop,
            "epilogue": row.get("epilogue", "auto"),
            "scheduler": infer_scheduler_from_row(row),
        }))

        if len(out) >= top_n:
            break

    return out


def dedupe(combos: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for combo in combos:
        cc = canonical_combo(combo)
        by_key[combo_key(cc)] = cc

    out = list(by_key.values())
    out.sort(key=lambda c: (
        str(c["mainloop"]),
        normalize_scheduler(c.get("scheduler", "normal")),
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


def cpp_template_args(combo: Dict[str, Any]) -> str:
    tm = int(combo["tile_m"])
    tn = int(combo["tile_n"])
    tk = int(combo["tile_k"])
    stage = stage_cpp(combo["stages"])
    cluster = cluster_cpp(combo["cluster"])
    mainloop = MAINLOOP_TYPES[combo["mainloop"]]
    epilogue = EPILOGUE_TYPES[combo["epilogue"]]
    scheduler = SCHEDULER_TYPES[normalize_scheduler(combo.get("scheduler", "normal"))]
    return f"{tm}, {tn}, {tk}, {stage}, {cluster}, {mainloop}, {epilogue}, {scheduler}"


def write_algo_dicts(root: Path, candidates: List[Dict[str, Any]]) -> None:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    index_dict = {combo_key(combo): idx for idx, combo in enumerate(candidates)}

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
                "description": "SM90/H100 CUTLASS-3 GEMM algorithm dictionary shared by plain and signal paths",
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


def write_signal_instances(root: Path, candidates: List[Dict[str, Any]]) -> None:
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
            f.write(f"  &::cutlass_gemm_signal_sm90<{cpp_template_args(combo)}>,\n")
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


def write_plain_instances(root: Path, candidates: List[Dict[str, Any]]) -> None:
    inc_dir = root / "src" / "inc"
    tiling_dir = root / "src" / "tiling"
    inc_dir.mkdir(parents=True, exist_ok=True)
    tiling_dir.mkdir(parents=True, exist_ok=True)

    inc_path = inc_dir / "plain_instances_sm90.inc"
    table_path = tiling_dir / "plain_tiling_sm90.cuh"

    with open(inc_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by tool/generate_instances_sm90.py\n")
        f.write("// Do not edit by hand.\n\n")

        for combo in candidates:
            args = cpp_template_args(combo)
            f.write(
                "template bool cutlass_gemm_plain_sm90<\n"
                f"  {args}\n"
                ">(\n"
                "  int M, int N, int K,\n"
                "  half* A, half* B_col, half* D_col,\n"
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
            "using PlainSm90FuncPtr = bool (*)(\n"
            "    int M, int N, int K,\n"
            "    half* A, half* B_col, half* D_col,\n"
            "    cudaStream_t stream);\n\n"
        )

        f.write("struct PlainSm90AlgoMeta {\n")
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

        f.write("static PlainSm90FuncPtr plain_sm90_func_table[] = {\n")
        for combo in candidates:
            f.write(f"  &::cutlass_gemm_plain_sm90<{cpp_template_args(combo)}>,\n")
        f.write("};\n\n")
        f.write(f"static constexpr int plain_sm90_func_count = {len(candidates)};\n\n")

        f.write("static PlainSm90AlgoMeta plain_sm90_algo_meta[] = {\n")
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


def print_summary(candidates: List[Dict[str, Any]]) -> None:
    print("Generated SM90 instances:")
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


def main() -> None:
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

    candidates: List[Dict[str, Any]] = []

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
        candidates = [
            c for c in candidates
            if normalize_scheduler(c.get("scheduler", "normal")) == "stream_k"
        ]

    candidates = dedupe(candidates)

    if not candidates:
        raise RuntimeError("No SM90 candidates generated.")

    write_algo_dicts(root, candidates)
    write_signal_instances(root, candidates)
    write_plain_instances(root, candidates)
    print_summary(candidates)

    print("")
    print("Wrote:")
    print(f"  {root / 'configs' / 'AlgoDictSm90.pt'}")
    print(f"  {root / 'configs' / 'AlgoDictSm90.json'}")
    print(f"  {root / 'src' / 'inc' / 'signal_instances_sm90.inc'}")
    print(f"  {root / 'src' / 'tiling' / 'signal_tiling_sm90.cuh'}")
    print(f"  {root / 'src' / 'inc' / 'plain_instances_sm90.inc'}")
    print(f"  {root / 'src' / 'tiling' / 'plain_tiling_sm90.cuh'}")


if __name__ == "__main__":
    main()
