#!/usr/bin/env python3
"""
Generate SM90 CUTLASS-3 GEMM-with-signal instances for ooverlap.

This version supports exact stage-count keys:

  TileM, TileN, TileK,
  ClusterM, ClusterN, ClusterK,
  Stages,
  Mainloop,
  Epilogue

Recommended workflow after a profiling miss:

  python tool/generate_instances_sm90.py \
    --from-missing-json configs/m4096n2048k1024_nvidia_h100_nvl_normal_sm90_missing.json \
    --top-missing 20 \
    --keep-base-ws

  cd build && make -j

Then rerun gen_config_sm90.py with:

  --match-stages exact
"""

import argparse
import json
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


def root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def cluster_cpp(cluster):
    cm, cn, ck = cluster
    return f"cute::Shape<cute::_{cm}, cute::_{cn}, cute::_{ck}>"


def stage_cpp(stages):
    if stages is None or str(stages).lower() == "auto" or int(stages) < 0:
        return "cutlass::gemm::collective::StageCountAuto"
    return f"cutlass::gemm::collective::StageCount<{int(stages)}>"


def combo_key(combo):
    return (
        int(combo["tile_m"]),
        int(combo["tile_n"]),
        int(combo["tile_k"]),
        int(combo["cluster"][0]),
        int(combo["cluster"][1]),
        int(combo["cluster"][2]),
        int(combo["stages"]) if combo.get("stages") and combo["stages"] != "auto" is not None else "auto",
        str(combo["mainloop"]),
        str(combo["epilogue"]),
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
    }


def canonical_combo(x):
    return {
        "tile_m": int(x["tile_m"]),
        "tile_n": int(x["tile_n"]),
        "tile_k": int(x["tile_k"]),
        "cluster": [int(x["cluster"][0]), int(x["cluster"][1]), int(x["cluster"][2])],
        "stages": int(x["stages"]) if x.get("stages") not in (None, "auto") else "auto",
        "mainloop": str(x["mainloop"]),
        "epilogue": str(x.get("epilogue", "auto")),
    }


def default_base_ws_combos():
    """
    Compact fallback set. These are not meant to beat the profiler; they keep
    old algo coverage available while you add exact profiler misses.
    """
    combos = []
    for tm in [64, 128, 256]:
        for tn in [64, 128, 256]:
            if tm == 64 and tn == 64:
                continue
            if tm == 256 and tn == 256:
                continue
            for tk in [32, 64, 128]:
                # StageCountAuto is useful as a fallback but exact profiling should
                # use concrete stages from missing JSON.
                for cluster in [(1, 1, 1), (1, 2, 1), (2, 1, 1)]:
                    combos.append({
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "cluster": list(cluster),
                        "stages": "auto",
                        "mainloop": "ws",
                        "epilogue": "auto",
                    })
    return combos


def curated_sm90_combos(include_pingpong: bool, include_cooperative: bool):
    """
    Small H100-focused set based on the top profiler patterns we have seen:
      - ws:          64x256/128x128/128x256/256x128
      - pingpong:    64x256 stage 5, 128x128 stage 6
      - cooperative: 128x128 stage 6, 128x256/256x128 stage 4
    """
    combos = []

    # WS exact-ish stage candidates.
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
            })

    if include_pingpong:
        for tm, tn, tk, stages in [
            (64, 256, 64, 5),
            (128, 128, 64, 6),
        ]:
            for cluster in [(1, 2, 1), (2, 1, 1)]:
                combos.append({
                    "tile_m": tm, "tile_n": tn, "tile_k": tk,
                    "cluster": list(cluster),
                    "stages": stages,
                    "mainloop": "pingpong",
                    "epilogue": "auto",
                })

    if include_cooperative:
        for tm, tn, tk, stages in [
            (128, 128, 64, 6),
            (128, 256, 64, 4),
            (256, 128, 64, 4),
        ]:
            for cluster in [(1, 2, 1), (2, 1, 1)]:
                combos.append({
                    "tile_m": tm, "tile_n": tn, "tile_k": tk,
                    "cluster": list(cluster),
                    "stages": stages,
                    "mainloop": "cooperative",
                    "epilogue": "auto",
                })

    return combos


def load_missing_json(path: Path, top_miss: int, allow_slow: bool):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("missing", data)
    if isinstance(rows, dict):
        rows = rows.get("missing", [])

    out = []
    for r in rows:
        if "tile_m" not in r:
            continue

        # Skip obvious f32-output rows unless requested. Our wrapper is D=f16.
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
        }))

        if len(out) >= top_miss:
            break

    return out


def dedupe(combos):
    by_key = {}
    for c in combos:
        key = combo_key(c)
        by_key[key] = canonical_combo(c)

    out = list(by_key.values())
    out.sort(key=lambda c: (
        str(c["mainloop"]),
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
    return f"{tm}, {tn}, {tk}, {stage}, {cluster}, {mainloop}, {epilogue}"


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
            f.write(
                "  {"
                f"{int(combo['tile_m'])}, "
                f"{int(combo['tile_n'])}, "
                f"{int(combo['tile_k'])}, "
                f"{int(cm)}, {int(cn)}, {int(ck)}, "
                f"{stages}, "
                f"\"{combo['mainloop']}\", "
                f"\"{combo['epilogue']}\""
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
            f"epilogue={combo['epilogue']}"
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

    ap.add_argument("--from-missing-json", type=str, default=None)
    ap.add_argument("--top-missing", type=int, default=20)
    ap.add_argument("--keep-base-ws", action="store_true")
    ap.add_argument("--allow-slow-or-f32-output-rows", action="store_true")

    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root is not None else root_from_script()

    candidates = []

    if args.from_missing_json is not None:
        candidates.extend(load_missing_json(
            Path(args.from_missing_json).expanduser().resolve(),
            args.top_missing,
            args.allow_slow_or_f32_output_rows,
        ))
        if args.keep_base_ws:
            candidates.extend(default_base_ws_combos())
    else:
        if args.preset == "base-ws-only":
            candidates.extend(default_base_ws_combos())
        else:
            candidates.extend(curated_sm90_combos(
                include_pingpong=args.include_pingpong,
                include_cooperative=args.include_cooperative,
            ))

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
