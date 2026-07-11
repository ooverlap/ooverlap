#!/usr/bin/env python3
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

CLUSTERS = [
    (1, 1, 1),
    (1, 2, 1),
    (2, 1, 1),
]

BASE_TILE_M = [64, 128, 256]
BASE_TILE_N = [64, 128, 256]
BASE_TILE_K = [32, 64, 128]
BASE_SKIP_MN = {
    (64, 64),
    (256, 256),
}

CURATED_SHAPES = [
    # mainloop, tile_m, tile_n, tile_k, stages
    ("ws", 64, 256, 64, 5),
    ("ws", 128, 128, 64, 7),
    ("ws", 128, 256, 64, 4),
    ("ws", 256, 128, 64, 4),

    ("pingpong", 64, 256, 64, 5),
    ("pingpong", 128, 128, 64, 6),
    ("pingpong", 128, 128, 64, 7),

    ("cooperative", 128, 128, 64, 6),
    ("cooperative", 128, 256, 64, 4),
    ("cooperative", 256, 128, 64, 4),
    ("cooperative", 256, 256, 64, 3),
]

SCHEDULERS = [
    "normal",
    "stream_k",
]


def root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_scheduler(x: Any) -> str:
    s = "normal" if x is None else str(x).strip().lower().replace("-", "_")
    if s in ("streamk", "stream_k", "stream"):
        return "stream_k"
    return "normal"


def normalize_stages(x: Any) -> Any:
    if x is None:
        return "auto"
    s = str(x).strip().lower()
    if s in ("auto", "-1"):
        return "auto"
    return int(float(s))


def normalize_split_k(x: Any) -> int:
    if x is None:
        return 1
    s = str(x).strip()
    return int(float(s)) if s else 1


def is_supported_combo(mainloop: str, scheduler: str) -> bool:
    mainloop = str(mainloop).strip().lower()
    scheduler = normalize_scheduler(scheduler)

    # CUTLASS rejects Stream-K with normal warp-specialized WS:
    # "TMA warp-specialized kernel does not support specializing the tile scheduler."
    #
    # CUTLASS also rejects Stream-K with pingpong:
    # "Ping-pong kernel does not currently support stream-K scheduler."
    #
    # So in this generator, Stream-K should only be emitted for cooperative.
    if scheduler == "stream_k" and mainloop != "cooperative":
        return False

    return True


def infer_scheduler(row: Dict[str, Any]) -> str:
    if "scheduler" in row:
        return normalize_scheduler(row["scheduler"])
    if "csv_scheduler" in row:
        return normalize_scheduler(row["csv_scheduler"])
    if bool(row.get("is_stream_k", False)):
        return "stream_k"

    haystack = " ".join(
        str(row.get(k, ""))
        for k in ("operation", "error", "match", "reason")
    ).lower().replace("-", "_")

    return "stream_k" if re.search(r"stream_?k", haystack) else "normal"


def is_f16_row(row: Dict[str, Any]) -> bool:
    text = " ".join(str(row.get(k, "")) for k in row).lower()
    if "f32" in text or "float32" in text or "tf32" in text:
        return False
    return True


def cluster_cpp(cluster: Iterable[int]) -> str:
    cm, cn, ck = [int(x) for x in cluster]
    return f"cute::Shape<cute::_{cm}, cute::_{cn}, cute::_{ck}>"


def stage_cpp(stages: Any) -> str:
    stages = normalize_stages(stages)
    if stages == "auto":
        return "cutlass::gemm::collective::StageCountAuto"
    return f"cutlass::gemm::collective::StageCount<{int(stages)}>"


def combo_key(c: Dict[str, Any]) -> Tuple[Any, ...]:
    cluster = c["cluster"]
    return (
        int(c["tile_m"]),
        int(c["tile_n"]),
        int(c["tile_k"]),
        int(cluster[0]),
        int(cluster[1]),
        int(cluster[2]),
        normalize_stages(c.get("stages", "auto")),
        str(c.get("mainloop", "ws")),
        str(c.get("epilogue", "auto")),
        normalize_scheduler(c.get("scheduler", "normal")),
        normalize_split_k(c.get("split_k", 1)),
    )


def combo_from_key(k: Tuple[Any, ...]) -> Dict[str, Any]:
    return {
        "tile_m": int(k[0]),
        "tile_n": int(k[1]),
        "tile_k": int(k[2]),
        "cluster": [int(k[3]), int(k[4]), int(k[5])],
        "stages": k[6],
        "mainloop": str(k[7]),
        "epilogue": str(k[8]),
        "scheduler": str(k[9]),
        "split_k": int(k[10]),
    }


def canonical_combo(x: Dict[str, Any]) -> Dict[str, Any]:
    mainloop = str(x.get("mainloop", "ws"))
    epilogue = str(x.get("epilogue", "auto"))
    scheduler = normalize_scheduler(x.get("scheduler", "normal"))

    if mainloop not in MAINLOOP_TYPES:
        raise ValueError(f"Unsupported mainloop={mainloop}")
    if epilogue not in EPILOGUE_TYPES:
        raise ValueError(f"Unsupported epilogue={epilogue}")
    if scheduler not in SCHEDULER_TYPES:
        raise ValueError(f"Unsupported scheduler={scheduler}")
    if not is_supported_combo(mainloop, scheduler):
        raise ValueError(f"Unsupported combo: mainloop={mainloop}, scheduler={scheduler}")

    return {
        "tile_m": int(x["tile_m"]),
        "tile_n": int(x["tile_n"]),
        "tile_k": int(x["tile_k"]),
        "cluster": [
            int(x["cluster"][0]),
            int(x["cluster"][1]),
            int(x["cluster"][2]),
        ],
        "stages": normalize_stages(x.get("stages", "auto")),
        "mainloop": mainloop,
        "epilogue": epilogue,
        "scheduler": scheduler,
        "split_k": normalize_split_k(x.get("split_k", 1)),
    }


def builtin_combos() -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []

    for tm in BASE_TILE_M:
        for tn in BASE_TILE_N:
            if (tm, tn) in BASE_SKIP_MN:
                continue
            for tk in BASE_TILE_K:
                for cluster in CLUSTERS:
                    combos.append({
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "cluster": list(cluster),
                        "stages": "auto",
                        "mainloop": "ws",
                        "epilogue": "auto",
                        "scheduler": "normal",
                        "split_k": 1,
                    })

    for mainloop, tm, tn, tk, stages in CURATED_SHAPES:
        for cluster in CLUSTERS:
            for scheduler in SCHEDULERS:
                if not is_supported_combo(mainloop, scheduler):
                    continue

                combos.append({
                    "tile_m": tm,
                    "tile_n": tn,
                    "tile_k": tk,
                    "cluster": list(cluster),
                    "stages": stages,
                    "mainloop": mainloop,
                    "epilogue": "auto",
                    "scheduler": scheduler,
                    "split_k": 1,
                })

    return combos


def rows_from_json_obj(data: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if key in data:
        return data.get(key, [])
    if "failed" in data:
        return data.get("failed", [])
    if "missing" in data:
        return data.get("missing", [])
    if "top" in data:
        return data.get("top", [])
    if "all_profiled" in data:
        return data.get("all_profiled", [])
    return []


def load_rows_json(path: Path, top_n: int, key: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = rows_from_json_obj(data, key)
    out: List[Dict[str, Any]] = []

    for row in rows:
        if "tile_m" not in row or not is_f16_row(row):
            continue

        mainloop = row.get("mainloop", "ws")
        scheduler = infer_scheduler(row)

        if not is_supported_combo(str(mainloop), scheduler):
            continue

        out.append(canonical_combo({
            "tile_m": row["tile_m"],
            "tile_n": row["tile_n"],
            "tile_k": row["tile_k"],
            "cluster": row["cluster"],
            "stages": row.get("stages", "auto"),
            "mainloop": mainloop,
            "epilogue": row.get("epilogue", "auto"),
            "scheduler": scheduler,
            "split_k": row.get("split_k", row.get("split_k_slices", 1)),
        }))

        if len(out) >= top_n:
            break

    return out


def dedupe(combos: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for c in combos:
        try:
            cc = canonical_combo(c)
        except ValueError as e:
            print(f"[skip] {e}")
            continue

        by_key[combo_key(cc)] = cc

    out = list(by_key.values())
    out.sort(key=lambda c: (
        str(c["mainloop"]),
        str(c["scheduler"]),
        int(c["tile_m"]),
        int(c["tile_n"]),
        int(c["tile_k"]),
        int(c["cluster"][0]),
        int(c["cluster"][1]),
        int(c["cluster"][2]),
        str(c["stages"]),
        int(c["split_k"]),
    ))
    return out


def cpp_template_args(c: Dict[str, Any]) -> str:
    return (
        f"{int(c['tile_m'])}, "
        f"{int(c['tile_n'])}, "
        f"{int(c['tile_k'])}, "
        f"{stage_cpp(c['stages'])}, "
        f"{cluster_cpp(c['cluster'])}, "
        f"{MAINLOOP_TYPES[c['mainloop']]}, "
        f"{EPILOGUE_TYPES[c['epilogue']]}, "
        f"{SCHEDULER_TYPES[normalize_scheduler(c['scheduler'])]}"
    )


def write_algo_dicts(root: Path, candidates: List[Dict[str, Any]]) -> None:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    index = {combo_key(c): i for i, c in enumerate(candidates)}

    if torch is not None:
        torch.save(index, config_dir / "AlgoDictSm90.pt")
    else:
        print("[WARN] torch is not importable; skipping AlgoDictSm90.pt")

    algorithms = [
        {
            "algo": index[combo_key(c)],
            **combo_from_key(combo_key(c)),
        }
        for c in candidates
    ]

    (config_dir / "AlgoDictSm90.json").write_text(
        json.dumps(
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
                    "SplitK",
                ],
                "algorithms": algorithms,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def write_signal_instances(root: Path, candidates: List[Dict[str, Any]]) -> None:
    inc_dir = root / "src" / "inc"
    tiling_dir = root / "src" / "tiling"
    inc_dir.mkdir(parents=True, exist_ok=True)
    tiling_dir.mkdir(parents=True, exist_ok=True)

    inc = ["// Auto-generated by tool/generate_instances_sm90.py\n", "// Do not edit by hand.\n\n"]
    for c in candidates:
        inc.append(
            "template void cutlass_gemm_signal_sm90<\n"
            f"  {cpp_template_args(c)}\n"
            ">(\n"
            "  int M, int N, int K,\n"
            "  int ReLDN,\n"
            "  int num_segments,\n"
            "  int* CommThr,\n"
            "  half* A, half* B, half* D,\n"
            "  int* MM, int* RA,\n"
            "  int active_sm_count,\n"
            "  bool Monitor,\n"
            "  cudaStream_t stream\n"
            ");\n\n"
        )
    (inc_dir / "signal_instances_sm90.inc").write_text("".join(inc), encoding="utf-8")

    h = [
        "// Auto-generated by tool/generate_instances_sm90.py\n",
        "// Do not edit by hand.\n\n",
        "#pragma once\n\n",
        "#include <cuda_runtime.h>\n",
        "#include <cuda_fp16.h>\n",
        "#include <cstdint>\n\n",
        "namespace ooverlap {\n\n",
        "using SignalSm90FuncPtr = void (*)(\n",
        "    int M, int N, int K,\n",
        "    int ReLDN,\n",
        "    int num_segments,\n",
        "    int* CommThr,\n",
        "    half* A, half* B, half* D,\n",
        "    int* MM, int* RA,\n",
        "    int active_sm_count,\n",
        "    bool Monitor,\n",
        "    cudaStream_t stream);\n\n",
        "struct SignalSm90AlgoMeta {\n",
        "  int tile_m;\n",
        "  int tile_n;\n",
        "  int tile_k;\n",
        "  int cluster_m;\n",
        "  int cluster_n;\n",
        "  int cluster_k;\n",
        "  int stages;\n",
        "  const char* mainloop;\n",
        "  const char* epilogue;\n",
        "  const char* scheduler;\n",
        "  int split_k;\n",
        "};\n\n",
        "static SignalSm90FuncPtr signal_sm90_func_table[] = {\n",
    ]

    for c in candidates:
        h.append(f"  &::cutlass_gemm_signal_sm90<{cpp_template_args(c)}>,\n")

    h += [
        "};\n\n",
        f"static constexpr int signal_sm90_func_count = {len(candidates)};\n\n",
        "static SignalSm90AlgoMeta signal_sm90_algo_meta[] = {\n",
    ]

    for c in candidates:
        cm, cn, ck = c["cluster"]
        stages = -1 if str(c["stages"]).lower() == "auto" else int(c["stages"])
        h.append(
            "  {"
            f"{int(c['tile_m'])}, "
            f"{int(c['tile_n'])}, "
            f"{int(c['tile_k'])}, "
            f"{int(cm)}, {int(cn)}, {int(ck)}, "
            f"{stages}, "
            f"\"{c['mainloop']}\", "
            f"\"{c['epilogue']}\", "
            f"\"{normalize_scheduler(c['scheduler'])}\", "
            f"{normalize_split_k(c.get('split_k', 1))}"
            "},\n"
        )

    h += ["};\n\n", "} // namespace ooverlap\n"]
    (tiling_dir / "signal_tiling_sm90.cuh").write_text("".join(h), encoding="utf-8")


def write_plain_instances(root: Path, candidates: List[Dict[str, Any]]) -> None:
    inc_dir = root / "src" / "inc"
    tiling_dir = root / "src" / "tiling"
    inc_dir.mkdir(parents=True, exist_ok=True)
    tiling_dir.mkdir(parents=True, exist_ok=True)

    inc = ["// Auto-generated by tool/generate_instances_sm90.py\n", "// Do not edit by hand.\n\n"]
    for c in candidates:
        inc.append(
            "template bool cutlass_gemm_plain_sm90<\n"
            f"  {cpp_template_args(c)}\n"
            ">(\n"
            "  int M, int N, int K,\n"
            "  half* A, half* B_col, half* D_col,\n"
            "  cudaStream_t stream\n"
            ");\n\n"
        )
    (inc_dir / "plain_instances_sm90.inc").write_text("".join(inc), encoding="utf-8")

    h = [
        "// Auto-generated by tool/generate_instances_sm90.py\n",
        "// Do not edit by hand.\n\n",
        "#pragma once\n\n",
        "#include <cuda_runtime.h>\n",
        "#include <cuda_fp16.h>\n",
        "#include <cstdint>\n\n",
        "namespace ooverlap {\n\n",
        "using PlainSm90FuncPtr = bool (*)(\n",
        "    int M, int N, int K,\n",
        "    half* A, half* B_col, half* D_col,\n",
        "    cudaStream_t stream);\n\n",
        "struct PlainSm90AlgoMeta {\n",
        "  int tile_m;\n",
        "  int tile_n;\n",
        "  int tile_k;\n",
        "  int cluster_m;\n",
        "  int cluster_n;\n",
        "  int cluster_k;\n",
        "  int stages;\n",
        "  const char* mainloop;\n",
        "  const char* epilogue;\n",
        "  const char* scheduler;\n",
        "  int split_k;\n",
        "};\n\n",
        "static PlainSm90FuncPtr plain_sm90_func_table[] = {\n",
    ]

    for c in candidates:
        h.append(f"  &::cutlass_gemm_plain_sm90<{cpp_template_args(c)}>,\n")

    h += [
        "};\n\n",
        f"static constexpr int plain_sm90_func_count = {len(candidates)};\n\n",
        "static PlainSm90AlgoMeta plain_sm90_algo_meta[] = {\n",
    ]

    for c in candidates:
        cm, cn, ck = c["cluster"]
        stages = -1 if str(c["stages"]).lower() == "auto" else int(c["stages"])
        h.append(
            "  {"
            f"{int(c['tile_m'])}, "
            f"{int(c['tile_n'])}, "
            f"{int(c['tile_k'])}, "
            f"{int(cm)}, {int(cn)}, {int(ck)}, "
            f"{stages}, "
            f"\"{c['mainloop']}\", "
            f"\"{c['epilogue']}\", "
            f"\"{normalize_scheduler(c['scheduler'])}\", "
            f"{normalize_split_k(c.get('split_k', 1))}"
            "},\n"
        )

    h += ["};\n\n", "} // namespace ooverlap\n"]
    (tiling_dir / "plain_tiling_sm90.cuh").write_text("".join(h), encoding="utf-8")


def print_summary(candidates: List[Dict[str, Any]]) -> None:
    print("Generated SM90 instances:")
    print(f"  count = {len(candidates)}")
    for i, c in enumerate(candidates):
        cm, cn, ck = c["cluster"]
        print(
            f"  algo={i:03d} "
            f"tile={c['tile_m']}x{c['tile_n']}x{c['tile_k']} "
            f"cluster={cm}x{cn}x{ck} "
            f"stages={c['stages']} "
            f"mainloop={c['mainloop']} "
            f"epilogue={c['epilogue']} "
            f"scheduler={normalize_scheduler(c['scheduler'])} "
            f"split_k={normalize_split_k(c.get('split_k', 1))}"
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--from-missing-json", type=str, default=None)
    ap.add_argument("--top-missing", type=int, default=20)
    ap.add_argument("--from-failed-json", type=str, default=None)
    ap.add_argument("--top-failed", type=int, default=20)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else root_from_script()

    candidates = builtin_combos()

    if args.from_missing_json:
        candidates.extend(load_rows_json(
            Path(args.from_missing_json).expanduser().resolve(),
            args.top_missing,
            "missing",
        ))

    if args.from_failed_json:
        candidates.extend(load_rows_json(
            Path(args.from_failed_json).expanduser().resolve(),
            args.top_failed,
            "failed",
        ))

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
