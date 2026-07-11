#!/usr/bin/env python3
"""
Verify SM90 signal epilogue tile-completion counters.

This runs the low-level pybind GEMM signal entrypoint:
    ooverlap_ext.gemm_signal_sm90(A, B, D, MM, RA, CommThr, ReLDN, algo, monitor)

It does not run communication, so MM segment counters are not consumed/reset by
kernel_wait_flag. That makes it useful for verifying the cooperative epilogue
signal path.

Example:
  CUDA_VISIBLE_DEVICES=0 python tool/verify_signal_tile_counts.py \
    --m 32768 --n 8192 --k 4096 --algo 0 --iters 5

For cooperative TMA epilogues, tile_done_sum is expected to be:
    tile_num * expected_arrivals_per_tile
while segment_ready_total and complete_tiles should both equal tile_num.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for q in [p.parent, *p.parents]:
        if (q / "build" / "lib" / "ooverlap_ext.so").exists():
            return q
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
    so = repo_root() / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    name = "ooverlap_ext"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension from {so}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def div_up(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def monitor_size(tile_num: int, seg_size: int, monitor: bool) -> int:
    if monitor:
        return seg_size + tile_num + 1 + tile_num
    return seg_size + tile_num


def expected_coop_tma_subtiles(tile_m: int, tile_n: int) -> int:
    # Matches CUTLASS cooperative EpilogueTileAuto for SM90 TMA epilogue:
    #   epi_m = min(128, tile_m)
    #   epi_n = gcd(min(32, tile_n), tile_n)
    epi_m = min(128, int(tile_m))
    epi_n = math.gcd(min(32, int(tile_n)), int(tile_n))
    return (int(tile_m) // epi_m) * (int(tile_n) // epi_n)


def make_segments(tile_num: int, num_segments: int) -> list[int]:
    if num_segments <= 0:
        raise ValueError("--segments must be > 0")
    num_segments = min(num_segments, tile_num)
    base = tile_num // num_segments
    rem = tile_num % num_segments
    return [base + (1 if i < rem else 0) for i in range(num_segments)]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--algo", type=int, required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--segments", type=int, default=1)
    ap.add_argument("--rldn", type=int, default=1)
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument(
        "--expected-arrivals",
        type=int,
        default=0,
        help="Override per-tile expected tile_done arrivals. Default uses cooperative TMA subtile estimate.",
    )
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.init()
    ext = load_ooverlap_ext()

    M, N, K = int(args.m), int(args.n), int(args.k)
    algo = int(args.algo)

    BM, BN = ext.gemm_signal_sm90_algo_info(algo)
    BM, BN = int(BM), int(BN)

    if M % BM != 0 or N % BN != 0:
        raise ValueError(
            f"M/N must be divisible by algo tile shape. "
            f"Got M={M}, N={N}, BM={BM}, BN={BN}."
        )

    tile_rows = M // BM
    tile_cols = N // BN
    tile_num = tile_rows * tile_cols

    if args.expected_arrivals > 0:
        expected_arrivals = int(args.expected_arrivals)
    else:
        expected_arrivals = expected_coop_tma_subtiles(BM, BN)

    rldn = int(args.rldn)
    packed_rows = div_up(tile_num, rldn) * BM
    packed_cols = rldn * BN

    cseg = make_segments(tile_num, int(args.segments))
    CommThr_cpu = torch.tensor(cseg, dtype=torch.int32)
    CommThr = CommThr_cpu.cuda()

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    D = torch.empty((packed_rows, packed_cols), dtype=torch.float16, device="cuda")

    # Identity logical_tile -> packed_tile map. For rldn=1 this stores into a tall packed output.
    RA = torch.arange(tile_num, dtype=torch.int32, device="cuda")

    MM = torch.zeros(
        (monitor_size(tile_num, len(cseg), bool(args.monitor)),),
        dtype=torch.int32,
        device="cuda",
    )

    def run_once():
        MM.zero_()
        ext.gemm_signal_sm90(A, B, D, MM, RA, CommThr, rldn, algo, bool(args.monitor))
        torch.cuda.synchronize()

        mm_cpu = MM.detach().cpu()
        seg_counts = mm_cpu[: len(cseg)]
        tile_done = mm_cpu[len(cseg) : len(cseg) + tile_num]

        segment_ready_total = int(seg_counts.sum().item())
        tile_done_sum = int(tile_done.sum().item())
        nonzero_tiles = int((tile_done > 0).sum().item())
        complete_tiles = int((tile_done >= expected_arrivals).sum().item())
        exact_tiles = int((tile_done == expected_arrivals).sum().item())
        over_tiles = int((tile_done > expected_arrivals).sum().item())

        return {
            "segment_ready_total": segment_ready_total,
            "tile_done_sum": tile_done_sum,
            "nonzero_tiles": nonzero_tiles,
            "complete_tiles": complete_tiles,
            "exact_tiles": exact_tiles,
            "over_tiles": over_tiles,
            "tile_done_min": int(tile_done.min().item()) if tile_num else 0,
            "tile_done_max": int(tile_done.max().item()) if tile_num else 0,
            "seg_min": int(seg_counts.min().item()) if len(cseg) else 0,
            "seg_max": int(seg_counts.max().item()) if len(cseg) else 0,
        }

    for _ in range(int(args.warmup)):
        run_once()

    print("========================================")
    print("SM90 signal tile-completion verification")
    print("========================================")
    print(f"M={M} N={N} K={K} algo={algo}")
    print(f"BM={BM} BN={BN} tile_rows={tile_rows} tile_cols={tile_cols}")
    print(f"tile_num={tile_num}")
    print(f"segments={len(cseg)} cseg_sum={sum(cseg)}")
    print(f"rldn={rldn} packed_D=({packed_rows}, {packed_cols})")
    print(f"expected_arrivals_per_tile={expected_arrivals}")
    print(f"expected_tile_done_sum={tile_num * expected_arrivals}")
    print("----------------------------------------")

    ok_all = True
    for i in range(int(args.iters)):
        s = run_once()
        ok = (
            s["segment_ready_total"] == tile_num
            and s["complete_tiles"] == tile_num
            and s["tile_done_sum"] == tile_num * expected_arrivals
            and s["over_tiles"] == 0
        )
        ok_all = ok_all and ok

        print(
            f"iter={i:03d} "
            f"segment_ready_total={s['segment_ready_total']}/{tile_num} "
            f"complete_tiles={s['complete_tiles']}/{tile_num} "
            f"exact_tiles={s['exact_tiles']}/{tile_num} "
            f"nonzero_tiles={s['nonzero_tiles']}/{tile_num} "
            f"tile_done_sum={s['tile_done_sum']}/{tile_num * expected_arrivals} "
            f"tile_done_minmax=[{s['tile_done_min']},{s['tile_done_max']}] "
            f"seg_minmax=[{s['seg_min']},{s['seg_max']}] "
            f"over_tiles={s['over_tiles']} "
            f"status={'PASS' if ok else 'FAIL'}"
        )

    print("----------------------------------------")
    print(f"overall={'PASS' if ok_all else 'FAIL'}")

    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
