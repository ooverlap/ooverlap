#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import importlib.util

def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load extension spec for {so}")

    spec.loader.exec_module(mod)
    return mod



EXPERIMENT_NAMES = {
    0: "reduce_two_to_one",
    1: "copy_one_to_two",
}

METHOD_NAMES = {
    0: "reduce_sequential_all_ctas",
    1: "reduce_split_ctas_gpu_scope",
    2: "copy_sequential_all_ctas",
    3: "copy_fanout_one_load_two_stores",
}

def enrich(rows):
    out = []
    for r in rows:
        d = dict(r)
        d["experiment_name"] = EXPERIMENT_NAMES.get(int(d["experiment"]), "unknown")
        d["method_name"] = METHOD_NAMES.get(int(d["method"]), "unknown")
        out.append(d)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-bytes", type=int, default=512 * 1024)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    ooverlap_ext = load_ooverlap_ext()

    rows = ooverlap_ext.benchmark_tma_batch_experiment_sm90(
        args.min_bytes,
        args.max_bytes,
        args.iters,
        args.warmup,
        args.num_blocks,
        args.dev0,
        args.dev1,
    )

    rows = enrich(rows)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}")

    print(
        f"{'experiment':<20} {'method':<34} {'bytes':>12} "
        f"{'ctas':>5} {'ms':>10} {'GB/s':>10}"
    )

    for r in rows:
        print(
            f"{r['experiment_name']:<20} {r['method_name']:<34} "
            f"{int(r['bytes']):>12} {int(r['num_blocks']):>5} "
            f"{r['latency_ms']:>10.4f} {r['gbps']:>10.2f}"
        )

if __name__ == "__main__":
    main()
