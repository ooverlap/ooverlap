#!/usr/bin/env python3
"""
Generate SM90 CUTLASS-3 GEMM-with-signal instances for ooverlap.

This is the SM90/H100 analogue of FlashOverlap's tool/generate_instances.py.

FlashOverlap generated CUTLASS-2/SM80-style configs:
  ThreadblockM/N/K, WarpM/N/K, InstructionM/N/K, NumStages, SwizzleSize, SplitK

For our CUTLASS-3/SM90 path, the relevant parameters are:
  TileM, TileN, TileK
  ClusterM, ClusterN, ClusterK
  MainloopSchedule
  EpilogueSchedule

Generated files:
  configs/AlgoDictSm90.pt
  configs/AlgoDictSm90.json
  src/inc/signal_instances_sm90.inc
  src/tiling/signal_tiling_sm90.cuh

Integration target:
  src/overlap/gemm_signal_sm90.cu

Important:
  By default this script only emits KernelTmaWarpSpecialized ("ws") kernels.

  Pingpong/cooperative are supported by this generator, but should not be enabled
  until GemmSignalSm90::initialize() supports their schedule-specific
  GemmKernel::Arguments constructor shapes.
"""

import argparse
import itertools
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


def cluster_cpp(cluster):
    cm, cn, ck = cluster
    return f"cute::Shape<cute::_{cm}, cute::_{cn}, cute::_{ck}>"


def combo_key(combo):
    """
    Stable Python/JSON key representation.

    Tuple layout:
      TileM, TileN, TileK,
      ClusterM, ClusterN, ClusterK,
      MainloopName,
      EpilogueName
    """
    return (
        combo["tile_m"],
        combo["tile_n"],
        combo["tile_k"],
        combo["cluster"][0],
        combo["cluster"][1],
        combo["cluster"][2],
        combo["mainloop"],
        combo["epilogue"],
    )

def sm90_two_stage_smem_ok(combo):
    tm = combo["tile_m"]
    tn = combo["tile_n"]
    tk = combo["tile_k"]

    # Conservative fp16 A/B mainloop smem estimate.
    # One stage stores one A tile and one B tile.
    bytes_per_element = 2
    bytes_per_stage = (tm * tk + tn * tk) * bytes_per_element

    # SM90 TMA warp-specialized kernels require at least 2 stages.
    # Use a conservative budget because CUTLASS also needs barriers,
    # descriptors, epilogue/pipeline storage, alignment, etc.
    min_required = 2 * bytes_per_stage
    conservative_budget = 220 * 1024

    return min_required <= conservative_budget


def combo_dict_from_key(key):
    return {
        "tile_m": key[0],
        "tile_n": key[1],
        "tile_k": key[2],
        "cluster": [key[3], key[4], key[5]],
        "mainloop": key[6],
        "epilogue": key[7],
    }


def is_reasonable_combo(combo, preset):
    tm = combo["tile_m"]
    tn = combo["tile_n"]
    tk = combo["tile_k"]
    cm, cn, ck = combo["cluster"]
    mainloop = combo["mainloop"]

    # Basic sanity.
    if ck != 1:
        return False

    # We use WGMMA/TMA kernels. Very tiny CTA tiles are usually not useful here.
    if tm == 64 and tn == 64:
        return False

    if (tm, tn) in [
        (256, 256),
        (256, 128),
        (128, 256),
    ]:
        return False

    # Keep 256x256 out of the safe/default set. It can be useful for pure GEMM,
    # but for FlashOverlap-style overlap it gives fewer output tiles/signals.
    if preset == "safe" and tm == 256 and tn == 256:
        return False

    # For small tile dimensions, clustered launch is usually not our first choice.
    if preset == "safe" and (tm < 128 or tn < 128) and (cm != 1 or cn != 1):
        return False

    # Keep the safe preset compact.
    if preset == "safe":
        if cm * cn > 2:
            return False

    # Pingpong/cooperative are experimental for our wrapper right now.
    # Keep their cluster space conservative.
    if mainloop in ("pingpong", "cooperative"):
        if (cm, cn, ck) != (1, 1, 1):
            return False

    # Avoid too many huge experimental kernels unless explicitly requested.
    if preset != "extended":
        if tm == 256 and tn == 256:
            return False

    if not sm90_two_stage_smem_ok(combo):
        return False

    return True


def build_candidates(args):
    if args.preset == "minimal":
        tile_m = [128]
        tile_n = [128]
        tile_k = [32, 64, 128]
        clusters = [(1, 1, 1)]
    elif args.preset == "safe":
        tile_m = [64, 128, 256]
        tile_n = [64, 128, 256]
        tile_k = [32, 64, 128]
        clusters = [
            (1, 1, 1),
            (1, 2, 1),
            (2, 1, 1),
        ]
    elif args.preset == "extended":
        tile_m = [64, 128, 256]
        tile_n = [64, 128, 256]
        tile_k = [32, 64, 128]
        clusters = [
            (1, 1, 1),
            (1, 2, 1),
            (2, 1, 1),
            (2, 2, 1),
        ]
    else:
        raise ValueError(f"unknown preset={args.preset}")

    mainloops = ["ws"]

    if args.include_pingpong:
        mainloops.append("pingpong")

    if args.include_cooperative:
        mainloops.append("cooperative")

    epilogues = ["auto"]

    candidates = []
    for tm, tn, tk, cluster, mainloop, epilogue in itertools.product(
        tile_m,
        tile_n,
        tile_k,
        clusters,
        mainloops,
        epilogues,
    ):
        combo = {
            "tile_m": tm,
            "tile_n": tn,
            "tile_k": tk,
            "cluster": cluster,
            "mainloop": mainloop,
            "epilogue": epilogue,
        }

        if is_reasonable_combo(combo, args.preset):
            candidates.append(combo)

    candidates.sort(key=lambda c: (
        c["mainloop"],
        c["tile_m"],
        c["tile_n"],
        c["tile_k"],
        c["cluster"][0],
        c["cluster"][1],
        c["cluster"][2],
        c["epilogue"],
    ))

    if args.max_count is not None:
        candidates = candidates[: args.max_count]

    return candidates


def write_algo_dicts(root, candidates):
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
                    "Mainloop",
                    "Epilogue",
                ],
                "algorithms": json_items,
            },
            f,
            indent=2,
        )


def cpp_template_args(combo):
    tm = combo["tile_m"]
    tn = combo["tile_n"]
    tk = combo["tile_k"]

    cluster = cluster_cpp(combo["cluster"])
    mainloop = MAINLOOP_TYPES[combo["mainloop"]]
    epilogue = EPILOGUE_TYPES[combo["epilogue"]]

    return f"{tm}, {tn}, {tk}, {cluster}, {mainloop}, {epilogue}"


def write_signal_instances(root, candidates):
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
        f.write("  const char* mainloop;\n")
        f.write("  const char* epilogue;\n")
        f.write("};\n\n")

        f.write("static SignalSm90FuncPtr signal_sm90_func_table[] = {\n")
        for combo in candidates:
            args = cpp_template_args(combo)
            f.write(f"  &::cutlass_gemm_signal_sm90<{args}>,\n")
        f.write("};\n\n")

        f.write("static constexpr int signal_sm90_func_count = ")
        f.write(f"{len(candidates)};\n\n")

        f.write("static SignalSm90AlgoMeta signal_sm90_algo_meta[] = {\n")
        for combo in candidates:
            cm, cn, ck = combo["cluster"]
            f.write(
                "  {"
                f"{combo['tile_m']}, "
                f"{combo['tile_n']}, "
                f"{combo['tile_k']}, "
                f"{cm}, {cn}, {ck}, "
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
            f"mainloop={combo['mainloop']} "
            f"epilogue={combo['epilogue']}"
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Repository root. Default: parent of this script's parent, "
            "assuming script is in tool/."
        ),
    )

    ap.add_argument(
        "--preset",
        choices=["minimal", "safe", "extended"],
        default="safe",
        help=(
            "minimal: current 128x128 family only. "
            "safe: compact H100 search space. "
            "extended: larger experimental search space."
        ),
    )

    ap.add_argument(
        "--include-pingpong",
        action="store_true",
        help=(
            "Also generate KernelTmaWarpSpecializedPingpong entries. "
            "Do not enable until GemmSignalSm90 supports pingpong "
            "KernelArguments construction."
        ),
    )

    ap.add_argument(
        "--include-cooperative",
        action="store_true",
        help=(
            "Also generate KernelTmaWarpSpecializedCooperative entries. "
            "Do not enable until GemmSignalSm90 supports cooperative "
            "KernelArguments construction if needed."
        ),
    )

    ap.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Optional cap on number of generated algorithms.",
    )

    args = ap.parse_args()

    if args.root is None:
        root = Path(__file__).resolve().parents[1]
    else:
        root = Path(args.root).resolve()

    candidates = build_candidates(args)

    if len(candidates) == 0:
        raise RuntimeError("No SM90 candidates generated. Check filters/options.")

    write_algo_dicts(root, candidates)
    write_signal_instances(root, candidates)
    print_summary(candidates)

    print("")
    print("Wrote:")
    print(f"  {root / 'configs' / 'AlgoDictSm90.pt'}")
    print(f"  {root / 'configs' / 'AlgoDictSm90.json'}")
    print(f"  {root / 'src' / 'inc' / 'signal_instances_sm90.inc'}")
    print(f"  {root / 'src' / 'tiling' / 'signal_tiling_sm90.cuh'}")

    if args.include_pingpong or args.include_cooperative:
        print("")
        print("[WARN] You enabled experimental schedules.")
        print("       The generated table may not compile until")
        print("       GemmSignalSm90::initialize() has schedule-specific")
        print("       KernelArguments construction.")


if __name__ == "__main__":
    main()
