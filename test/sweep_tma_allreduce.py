import argparse
import importlib.util
import json
import os
from pathlib import Path


def parse_int_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError(f"empty int list: {text!r}")
    return values


def parse_str_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(item)
    if not values:
        raise ValueError(f"empty string list: {text!r}")
    return values


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser("Sweep SM90 2-GPU TMA allreduce configs")
    parser.add_argument(
        "--numels",
        type=str,
        required=True,
        help="Comma-separated numel list, for example: 1048576,2097152,16777216",
    )
    parser.add_argument(
        "--kernels",
        type=str,
        default="tma_copy,seq_fast_gmem,overlap_fast_gmem",
        help="Comma-separated kernels: tma_copy,seq_fast_gmem,overlap_fast_gmem",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1024",
        help="Comma-separated CUDA block sizes, warp-aligned",
    )
    parser.add_argument(
        "--max-ctas",
        type=str,
        default="16",
        help="Comma-separated max total CTAs per rank",
    )
    parser.add_argument(
        "--window-chunks",
        type=str,
        default="16,32,64",
        help="Comma-separated chunks per signaling/work window",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--nccl-max-ctas",
        type=str,
        default=None,
        help="Optional NCCL_MAX_CTAS value to set before loading extension",
    )
    parser.add_argument(
        "--nccl-min-ctas",
        type=str,
        default=None,
        help="Optional NCCL_MIN_CTAS value to set before loading extension",
    )
    parser.add_argument(
        "--nccl-algo",
        type=str,
        default=None,
        help="Optional NCCL_ALGO value to set before loading extension",
    )
    parser.add_argument(
        "--nccl-proto",
        type=str,
        default=None,
        help="Optional NCCL_PROTO value to set before loading extension",
    )

    args = parser.parse_args()

    if args.nccl_max_ctas is not None:
        os.environ["NCCL_MAX_CTAS"] = str(args.nccl_max_ctas)
    if args.nccl_min_ctas is not None:
        os.environ["NCCL_MIN_CTAS"] = str(args.nccl_min_ctas)
    if args.nccl_algo is not None:
        os.environ["NCCL_ALGO"] = str(args.nccl_algo)
    if args.nccl_proto is not None:
        os.environ["NCCL_PROTO"] = str(args.nccl_proto)

    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

    numels = parse_int_list(args.numels)
    kernels = parse_str_list(args.kernels)
    threads = parse_int_list(args.threads)
    max_ctas = parse_int_list(args.max_ctas)
    window_chunks = parse_int_list(args.window_chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(f"[info] dev0={args.dev0} dev1={args.dev1}")
    print(f"[info] numels={numels}")
    print(f"[info] kernels={kernels}")
    print(f"[info] threads={threads}")
    print(f"[info] max_ctas={max_ctas}")
    print(f"[info] window_chunks={window_chunks}")
    print(f"[info] iters={args.iters} warmup={args.warmup}")
    print(f"[info] NCCL_MAX_CTAS={os.environ.get('NCCL_MAX_CTAS')}")
    print(f"[info] NCCL_MIN_CTAS={os.environ.get('NCCL_MIN_CTAS')}")
    print(f"[info] NCCL_ALGO={os.environ.get('NCCL_ALGO')}")
    print(f"[info] NCCL_PROTO={os.environ.get('NCCL_PROTO')}")

    ext = load_ooverlap_ext()
    print("[info] loaded extension:", ext)

    jsonl = ext.benchmark_tma_two_gpu_allreduce_sweep_sm90(
        [int(x) for x in numels],
        [str(x) for x in kernels],
        [int(x) for x in threads],
        [int(x) for x in max_ctas],
        [int(x) for x in window_chunks],
        int(args.iters),
        int(args.warmup),
        int(args.dev0),
        int(args.dev1),
    )

    if jsonl and not jsonl.endswith("\n"):
        jsonl += "\n"

    with out_path.open("a", encoding="utf-8") as f:
        f.write(jsonl)

    rows = [json.loads(line) for line in jsonl.splitlines() if line.strip()]
    candidate_rows = [r for r in rows if r.get("kernel") != "nccl"]

    print(f"[result] wrote {len(rows)} rows to {out_path}")

    if candidate_rows:
        best = max(candidate_rows, key=lambda r: float(r.get("speedup_vs_nccl", 0.0)))
        print("[best]")
        print(json.dumps(best, indent=2, sort_keys=True))

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)
    print("PASS ✅ TMA allreduce sweep")


if __name__ == "__main__":
    main()
