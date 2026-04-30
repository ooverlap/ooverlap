#!/usr/bin/env python3
"""
Search communication segment sizes for ooverlap SM90 GEMM+AllReduce overlap.

This is the ooverlap version of FlashOverlap's tuning/search step.  It uses the
same idea: try wave/group sizes, benchmark the overlapped GEMM+communication
path, and save the best configuration.  The implementation is intentionally
conservative for the current ooverlap state:

  * all_reduce is implemented through OverlapImpl.gemm_allreduce_overlap.
  * packed output is the default because overlap needs contiguous segment buffers.
  * cooperative SM90 algos are skipped by default because they have been observed
    to hang in the overlap signal path. Use --include-cooperative only after that
    path is fixed.

Examples:
  CUDA_VISIBLE_DEVICES=0,1 python tune/search.py \
    --m 16384 --n 4096 --k 2048 --comm-op all_reduce \
    --profile-config configs/m16384n4096k2048_nvidia_h100_nvl_normal_custom_sm90.json \
    --top-algos 4 --candidate-list 0,512,1024,2048,4096

  CUDA_VISIBLE_DEVICES=0,1 python tune/search.py \
    --m 16384 --n 4096 --k 2048 --predictive-search true \
    --bandwidth-json configs/bandwidth_nvidia_h100_nvl_all_reduce_2gpu.json
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import socket
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ------------------------- common utilities -------------------------


def repo_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name in ("tune", "tool", "test"):
        return here.parents[1]
    return here.parent


def load_ooverlap_ext():
    root = repo_root()
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ("1", "true", "yes", "y", "on"):
        return True
    if x in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {x!r}")


def parse_int_list(s: Optional[str]) -> List[int]:
    if not s:
        return []
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def gpu_slug() -> str:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "gpu"
    return name.lower().replace(" ", "_").replace("/", "_")


def sync_all(device_index: int):
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier(device_ids=[int(device_index)])
    torch.cuda.synchronize()


def reduce_max_float(x: float, device: torch.device) -> float:
    t = torch.tensor([float(x)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def time_cuda_max(fn, warmup: int, iters: int, device: torch.device, device_index: int):
    sync_all(device_index)

    for _ in range(warmup):
        fn()

    sync_all(device_index)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    local_ms = start.elapsed_time(end) / float(iters)
    max_ms = reduce_max_float(local_ms, device)

    sync_all(device_index)
    return max_ms, local_ms


# ------------------------- algo/config parsing -------------------------


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def default_algo_dict_path() -> Path:
    return repo_root() / "configs" / "AlgoDictSm90.json"


def load_algo_dict(path: Optional[str]) -> Tuple[Dict[int, dict], Path]:
    p = Path(path).expanduser().resolve() if path else default_algo_dict_path()
    if not p.exists():
        raise FileNotFoundError(f"Could not find algo dict: {p}")

    data = load_json(p)
    by_id = {}
    for item in data.get("algorithms", []):
        by_id[int(item["algo"])] = item
    if not by_id:
        raise RuntimeError(f"No algorithms found in {p}")
    return by_id, p


def find_profile_config(M: int, N: int, K: int, layout: str) -> Optional[Path]:
    cfg = repo_root() / "configs"
    patterns = [
        f"m{M}n{N}k{K}_*_{layout}_custom_sm90.json",
        f"m{M}n{N}k{K}_*_{layout}_sm90.json",
        f"m{M}n{N}k{K}_*_custom_sm90.json",
        f"m{M}n{N}k{K}_*_sm90.json",
    ]
    for pat in patterns:
        hits = sorted(cfg.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        hits = [p for p in hits if not p.name.endswith("_missing.json") and not p.name.endswith("_failed.json")]
        if hits:
            return hits[0]
    return None


def entry_ms(entry: dict) -> Optional[float]:
    for key in (
        "median_ms",
        "measured_ms",
        "normal_gemm_ms",
        "packed_signal_ms",
        "dur",
        "runtime_ms",
        "cutlass_runtime",
    ):
        if key in entry and entry[key] is not None:
            try:
                return float(entry[key])
            except Exception:
                pass
    return None


def profile_entries_from_json(data: dict) -> List[dict]:
    rows = []

    # Preferred formats from profile_gemm_signal_sm90_custom.py / gen_config_sm90.py.
    for field in ("top", "all_profiled"):
        for e in data.get(field, []) or []:
            if "algo" in e:
                rows.append(dict(e))

    # FlashOverlap-like compact config: Algo + dur arrays.
    if not rows and "Algo" in data:
        algos = data.get("Algo", [])
        durs = data.get("dur", [None] * len(algos))
        bms = data.get("BM", [None] * len(algos))
        bns = data.get("BN", [None] * len(algos))
        bks = data.get("BK", [None] * len(algos))
        for i, a in enumerate(algos):
            rows.append(
                {
                    "algo": int(a),
                    "median_ms": None if i >= len(durs) else durs[i],
                    "tile_m": None if i >= len(bms) else bms[i],
                    "tile_n": None if i >= len(bns) else bns[i],
                    "tile_k": None if i >= len(bks) else bks[i],
                }
            )

    # Deduplicate by algo, keeping the best/lower measured time if present.
    best = {}
    for e in rows:
        a = int(e["algo"])
        ms = entry_ms(e)
        if a not in best:
            best[a] = e
        else:
            old_ms = entry_ms(best[a])
            if old_ms is None or (ms is not None and ms < old_ms):
                best[a] = e

    out = list(best.values())
    out.sort(key=lambda e: float("inf") if entry_ms(e) is None else entry_ms(e))
    return out


def build_algo_candidates(
    args,
    algo_meta: Dict[int, dict],
) -> List[dict]:
    explicit_algos = parse_int_list(args.algo)

    profile_path = Path(args.profile_config).expanduser().resolve() if args.profile_config else None
    if profile_path is None and not explicit_algos:
        profile_path = find_profile_config(args.m, args.n, args.k, args.layout)

    entries = []
    if explicit_algos:
        entries = [{"algo": a, "median_ms": None} for a in explicit_algos]
    elif profile_path is not None and profile_path.exists():
        entries = profile_entries_from_json(load_json(profile_path))
    else:
        raise FileNotFoundError(
            "No --algo and no profile config found. Pass --algo 8 or --profile-config <json>."
        )

    candidates = []
    skipped = []
    seen = set()
    for e in entries:
        a = int(e["algo"])
        if a in seen:
            continue
        seen.add(a)

        meta = algo_meta.get(a)
        if meta is None:
            skipped.append((a, "missing in AlgoDictSm90"))
            continue

        mainloop = str(meta.get("mainloop", "")).lower()
        if (not args.include_cooperative) and "cooperative" in mainloop:
            skipped.append((a, "cooperative skipped"))
            continue

        tile_m = int(meta["tile_m"])
        tile_n = int(meta["tile_n"])
        if args.m % tile_m != 0 or args.n % tile_n != 0:
            skipped.append((a, f"shape not divisible by tile {tile_m}x{tile_n}"))
            continue

        item = dict(e)
        item.update(meta)
        item["algo"] = a
        item["profile_ms"] = entry_ms(e)
        candidates.append(item)

        if args.top_algos > 0 and len(candidates) >= args.top_algos:
            break

    if not candidates:
        raise RuntimeError(f"No usable algos after filtering. skipped={skipped[:20]}")

    return candidates


# ------------------------- tiling/search candidates -------------------------


def make_segments(num_tiles: int, group_tiles: int) -> List[int]:
    if group_tiles <= 0 or group_tiles >= num_tiles:
        return [num_tiles]
    segs = []
    rem = num_tiles
    while rem > 0:
        x = min(group_tiles, rem)
        segs.append(x)
        rem -= x
    return segs


def make_column_major_reorder(M, N, tile_m, tile_n, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    ra = torch.empty((num_tiles,), dtype=torch.int32, device=device)
    packed_pos = 0
    for tile_col in range(tile_cols):
        for tile_row in range(tile_rows):
            logical_tile = tile_row * tile_cols + tile_col
            ra[logical_tile] = packed_pos
            packed_pos += 1
    return ra


def make_identity_reorder(M, N, tile_m, tile_n, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def powers_of_two_candidates(num_tiles: int, min_group: int, max_group: int) -> List[int]:
    max_group = num_tiles if max_group <= 0 else min(max_group, num_tiles)
    min_group = max(1, min_group)
    vals = [0]  # full segment baseline
    p = 1
    while p < min_group:
        p *= 2
    while p <= max_group:
        vals.append(p)
        p *= 2
    vals.append(num_tiles)
    return sorted(set(v for v in vals if v == 0 or 1 <= v <= num_tiles), key=lambda x: (x == 0, x))


def load_bandwidth_curve(path: Optional[str]) -> Optional[List[dict]]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"bandwidth json not found: {p}")
    data = load_json(p)
    items = data.get("items", [])
    items = [x for x in items if "size_bytes" in x and ("alg_GBps" in x or "bus_GBps" in x)]
    items.sort(key=lambda x: int(x["size_bytes"]))
    return items


def interp_bandwidth_gbps(curve: List[dict], size_bytes: int, key: str = "alg_GBps") -> float:
    if not curve:
        return 1.0
    xs = [int(x["size_bytes"]) for x in curve]
    ys = [float(x.get(key, x.get("alg_GBps", 1.0))) for x in curve]
    if size_bytes <= xs[0]:
        return max(1e-9, ys[0])
    if size_bytes >= xs[-1]:
        return max(1e-9, ys[-1])
    for i in range(1, len(xs)):
        if size_bytes <= xs[i]:
            # Log-size interpolation is smoother for bandwidth curves.
            x0, x1 = math.log(xs[i - 1]), math.log(xs[i])
            y0, y1 = ys[i - 1], ys[i]
            t = (math.log(size_bytes) - x0) / max(1e-12, x1 - x0)
            return max(1e-9, y0 + t * (y1 - y0))
    return max(1e-9, ys[-1])


def predictive_candidates_for_algo(
    algo_entry: dict,
    num_tiles: int,
    tile_m: int,
    tile_n: int,
    dtype_bytes: int,
    curve: Optional[List[dict]],
    min_group: int,
    max_group: int,
    radius: int,
) -> List[int]:
    all_powers = powers_of_two_candidates(num_tiles, min_group, max_group)
    powers = [x for x in all_powers if x > 0]
    if not curve or not powers:
        return all_powers

    gemm_ms = algo_entry.get("profile_ms")
    if gemm_ms is None or gemm_ms <= 0:
        return all_powers

    tile_bytes = int(tile_m) * int(tile_n) * dtype_bytes

    # Pick the first group whose compute time per segment can hide the predicted
    # communication time for that segment. Then benchmark neighbors around it.
    chosen = powers[-1]
    for g in powers:
        seg_bytes = g * tile_bytes
        bw = interp_bandwidth_gbps(curve, seg_bytes, key="alg_GBps")
        comm_ms = seg_bytes / (bw * 1.0e9) * 1.0e3
        compute_ms = float(gemm_ms) * (float(g) / float(num_tiles))
        if compute_ms >= comm_ms:
            chosen = g
            break

    idx = min(range(len(powers)), key=lambda i: abs(powers[i] - chosen))
    lo = max(0, idx - radius)
    hi = min(len(powers), idx + radius + 1)
    vals = [0] + powers[lo:hi] + [num_tiles]
    return sorted(set(vals), key=lambda x: (x == 0, x))


def build_group_candidates(args, algo_entry: dict, num_tiles: int, tile_m: int, tile_n: int, curve) -> List[int]:
    explicit = parse_int_list(args.candidate_list)
    if explicit:
        vals = []
        for x in explicit:
            if x <= 0:
                vals.append(0)
            elif x <= num_tiles:
                vals.append(x)
        return sorted(set(vals), key=lambda x: (x == 0, x))

    if args.predictive_search:
        return predictive_candidates_for_algo(
            algo_entry=algo_entry,
            num_tiles=num_tiles,
            tile_m=tile_m,
            tile_n=tile_n,
            dtype_bytes=2,
            curve=curve,
            min_group=args.min_group_tiles,
            max_group=args.max_group_tiles,
            radius=args.predictive_radius,
        )

    return powers_of_two_candidates(num_tiles, args.min_group_tiles, args.max_group_tiles)


# ------------------------- benchmark worker -------------------------


def benchmark_algo_groups(
    rank: int,
    world: int,
    device: torch.device,
    ext,
    ov,
    args,
    algo_entry: dict,
    group_candidates: List[int],
) -> List[dict]:
    algo = int(algo_entry["algo"])
    tile_m = int(algo_entry["tile_m"])
    tile_n = int(algo_entry["tile_n"])
    tile_k = int(algo_entry["tile_k"])

    M, N, K = int(args.m), int(args.n), int(args.k)
    reldn = int(args.reldn)

    assert M % tile_m == 0
    assert N % tile_n == 0
    assert reldn > 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    packed_tile_cols = reldn
    packed_tile_rows = (num_tiles + packed_tile_cols - 1) // packed_tile_cols
    packed_M = packed_tile_rows * tile_m
    packed_N = packed_tile_cols * tile_n

    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B_ref = torch.randn((K, N), device=device, dtype=torch.float16)
    B_packed = B_ref.t().contiguous()
    C_packed = torch.empty((packed_M, packed_N), device=device, dtype=torch.float16)

    if args.reorder == "column_major":
        RA = make_column_major_reorder(M, N, tile_m, tile_n, device=device)
    elif args.reorder == "identity":
        RA = make_identity_reorder(M, N, tile_m, tile_n, device=device)
    else:
        raise ValueError(f"unknown reorder={args.reorder}")

    monitor = False
    results = []

    for g in group_candidates:
        cseg = make_segments(num_tiles, int(g))
        cseg_cpu = torch.tensor(cseg, dtype=torch.int32)
        cseg_gpu = cseg_cpu.to(device=device)
        MM = torch.empty((len(cseg) + num_tiles,), device=device, dtype=torch.int32)

        def run():
            MM.zero_()
            ov.gemm_allreduce_overlap(
                A,
                B_packed,
                C_packed,
                MM,
                RA,
                int(reldn),
                cseg_cpu,
                cseg_gpu,
                int(algo),
                monitor,
            )

        max_ms, local_ms = time_cuda_max(run, args.warmup, args.iters, device, rank)

        item = {
            "algo": algo,
            "group_tiles": int(g),
            "num_segments": len(cseg),
            "latency_ms": float(max_ms),
            "rank0_local_ms": float(local_ms) if rank == 0 else None,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "cluster": algo_entry.get("cluster"),
            "stages": algo_entry.get("stages"),
            "mainloop": algo_entry.get("mainloop"),
            "epilogue": algo_entry.get("epilogue"),
            "profile_ms": algo_entry.get("profile_ms"),
            "tile_rows": tile_rows,
            "tile_cols": tile_cols,
            "num_tiles": num_tiles,
            "reldn": reldn,
            "packed_shape": [packed_M, packed_N],
        }
        results.append(item)

        if rank == 0:
            print(
                f"  algo={algo:<4d} group={g:<8d} "
                f"segments={len(cseg):<5d} latency={max_ms:.6f} ms",
                flush=True,
            )

    # Drop references before next algo to reduce peak memory pressure.
    del A, B_ref, B_packed, C_packed, RA
    torch.cuda.empty_cache()
    sync_all(rank)

    return results


def worker(rank, world, dist_url, nccl_id, args_dict, algo_entries, curve):
    args = argparse.Namespace(**args_dict)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        rank=rank,
        world_size=world,
    )

    try:
        if args.comm_op != "all_reduce":
            raise RuntimeError("Current ooverlap search.py only supports --comm-op all_reduce.")

        torch.manual_seed(1234 + rank)

        ext = load_ooverlap_ext()
        ov = ext.OverlapImpl()
        ov.nccl_init(rank, world, nccl_id)
        ov.cutlass_init()
        ov.overlap_init()

        all_results = []

        for i, algo_entry in enumerate(algo_entries):
            algo = int(algo_entry["algo"])
            tile_m = int(algo_entry["tile_m"])
            tile_n = int(algo_entry["tile_n"])
            tile_rows = args.m // tile_m
            tile_cols = args.n // tile_n
            num_tiles = tile_rows * tile_cols
            groups = build_group_candidates(args, algo_entry, num_tiles, tile_m, tile_n, curve)

            if rank == 0:
                print("----------------------------------------", flush=True)
                print(
                    f"[{i + 1}/{len(algo_entries)}] algo={algo} "
                    f"tile={tile_m}x{tile_n}x{algo_entry.get('tile_k')} "
                    f"cluster={algo_entry.get('cluster')} stages={algo_entry.get('stages')} "
                    f"mainloop={algo_entry.get('mainloop')} profile_ms={algo_entry.get('profile_ms')}",
                    flush=True,
                )
                print(f"groups={groups}", flush=True)

            results = benchmark_algo_groups(rank, world, device, ext, ov, args, algo_entry, groups)
            if rank == 0:
                all_results.extend(results)

        if rank == 0:
            all_results.sort(key=lambda x: x["latency_ms"])
            best = all_results[0] if all_results else None

            out_json = Path(args.output).expanduser().resolve()
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_csv = out_json.with_suffix(".csv")

            payload = {
                "description": "ooverlap SM90 overlap segment-size search",
                "M": args.m,
                "N": args.n,
                "K": args.k,
                "gpu": torch.cuda.get_device_name(0),
                "world_size": world,
                "comm_op": args.comm_op,
                "layout": args.layout,
                "reorder": args.reorder,
                "reldn": args.reldn,
                "warmup": args.warmup,
                "iters": args.iters,
                "predictive_search": args.predictive_search,
                "candidate_list": args.candidate_list,
                "best": best,
                "top": all_results[: args.top_save],
                "results": all_results,
            }
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)

            fieldnames = [
                "algo",
                "group_tiles",
                "num_segments",
                "latency_ms",
                "tile_m",
                "tile_n",
                "tile_k",
                "cluster",
                "stages",
                "mainloop",
                "profile_ms",
                "num_tiles",
                "reldn",
            ]
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in all_results:
                    row = {k: item.get(k) for k in fieldnames}
                    writer.writerow(row)

            print("========================================")
            print("DONE")
            print(f"profiled combos: {len(all_results)}")
            print(f"wrote json:      {out_json}")
            print(f"wrote csv:       {out_csv}")
            print("")
            if best:
                print("Best config:")
                print(
                    f"  algo={best['algo']} group_tiles={best['group_tiles']} "
                    f"segments={best['num_segments']} latency_ms={best['latency_ms']:.6f} "
                    f"tile={best['tile_m']}x{best['tile_n']}x{best['tile_k']} "
                    f"mainloop={best.get('mainloop')} stages={best.get('stages')}"
                )
            print("========================================")

    finally:
        try:
            sync_all(rank)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


# ------------------------- CLI -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--comm-op", choices=["all_reduce", "reduce_scatter"], default="all_reduce")

    ap.add_argument("--layout", choices=["packed", "normal"], default="packed")
    ap.add_argument("--reorder", choices=["column_major", "identity"], default="column_major")
    ap.add_argument("--reldn", type=int, default=1)

    ap.add_argument("--algo", type=str, default=None, help="Comma-separated algos. Overrides profile config.")
    ap.add_argument("--algo-dict", type=str, default=None)
    ap.add_argument("--profile-config", type=str, default=None)
    ap.add_argument("--top-algos", type=int, default=4)
    ap.add_argument("--top-save", type=int, default=20)
    ap.add_argument("--include-cooperative", action="store_true")

    ap.add_argument("--candidate-list", type=str, default=None, help="Comma-separated group_tiles. 0 means full segment.")
    ap.add_argument("--min-group-tiles", type=int, default=64)
    ap.add_argument("--max-group-tiles", type=int, default=0, help="0 means num_tiles.")
    ap.add_argument("--predictive-search", type=parse_bool, default=False)
    ap.add_argument("--predictive-radius", type=int, default=2)
    ap.add_argument("--bandwidth-json", type=str, default=None)

    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--output", type=str, default=None)

    args = ap.parse_args()

    if args.layout != "packed":
        print("WARNING: overlap search uses packed output internally; --layout is kept for config discovery only.")

    assert torch.cuda.is_available(), "CUDA is required"
    gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
    assert gpus >= 1
    assert torch.cuda.device_count() >= gpus

    if args.comm_op != "all_reduce":
        raise RuntimeError("Current ooverlap search.py supports only --comm-op all_reduce.")

    algo_meta, algo_dict_path = load_algo_dict(args.algo_dict)
    algo_entries = build_algo_candidates(args, algo_meta)
    curve = load_bandwidth_curve(args.bandwidth_json)

    if args.output is None:
        suffix = "predictive" if args.predictive_search else "exhaustive"
        args.output = str(
            repo_root()
            / "configs"
            / f"m{args.m}n{args.n}k{args.k}_{gpu_slug()}_{args.comm_op}_{suffix}_sm90.json"
        )

    ext = load_ooverlap_ext()
    nccl_id = ext.generate_nccl_id()
    port = find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"

    print("========================================")
    print("search.py")
    print(f"shape:          M={args.m} N={args.n} K={args.k}")
    print(f"gpus:           {gpus}")
    print(f"comm_op:        {args.comm_op}")
    print(f"layout:         {args.layout}")
    print(f"reorder:        {args.reorder}")
    print(f"reldn:          {args.reldn}")
    print(f"algo_dict:      {algo_dict_path}")
    print(f"algos:          {[int(x['algo']) for x in algo_entries]}")
    print(f"skip coop:      {not args.include_cooperative}")
    print(f"predictive:     {args.predictive_search}")
    print(f"bandwidth_json: {args.bandwidth_json}")
    print(f"candidate_list: {args.candidate_list}")
    print(f"warmup/iters:   {args.warmup}/{args.iters}")
    print(f"output:         {Path(args.output).expanduser().resolve()}")
    print("========================================")

    mp.spawn(
        worker,
        args=(gpus, dist_url, nccl_id, vars(args), algo_entries, curve),
        nprocs=gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
