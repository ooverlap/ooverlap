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
    2: "reduce_one_to_two_local_remote",
}

METHOD_NAMES = {
    0: "reduce_sequential_all_ctas",
    1: "reduce_split_ctas_gpu_scope",
    2: "copy_sequential_all_ctas",
    3: "copy_fanout_one_load_two_stores",
    4: "reduce_split_ctas_opposite_directions_gpu_scope",
    5: "reduce_fanout_sequential_all_ctas",
    6: "reduce_fanout_one_load_two_reduces_gpu_scope",
}


def enrich(rows):
    out = []
    for r in rows:
        d = dict(r)
        d["experiment"] = int(d["experiment"])
        d["method"] = int(d["method"])
        d["bytes"] = int(d["bytes"])
        d["payload_bytes"] = int(d["payload_bytes"])
        d["num_blocks"] = int(d["num_blocks"])
        d["split0_ctas"] = int(d["split0_ctas"])
        d["split1_ctas"] = int(d["split1_ctas"])
        d["experiment_name"] = EXPERIMENT_NAMES.get(d["experiment"], "unknown")
        d["method_name"] = METHOD_NAMES.get(d["method"], "unknown")
        out.append(d)
    return out


def make_power_of_two_sizes(min_bytes, max_bytes):
    sizes = []
    b = min_bytes
    while b <= max_bytes:
        sizes.append(b)
        if b > max_bytes // 2:
            break
        b *= 2
    return sizes


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def row_map(rows, num_blocks, bytes_, experiment, method):
    for r in rows:
        if (
            r["num_blocks"] == num_blocks
            and r["bytes"] == bytes_
            and r["experiment"] == experiment
            and r["method"] == method
        ):
            return r
    return None


def fmt_ms(row):
    return f"{row['latency_ms']:.4f}" if row is not None else "NA"


def fmt_gbps(row):
    return f"{row['gbps']:.1f}" if row is not None else "NA"


def fmt_speedup(base, other):
    if base is None or other is None or other["latency_ms"] <= 0.0:
        return "NA"
    return f"{base['latency_ms'] / other['latency_ms']:.3f}x"


def print_section(rows, num_blocks, sizes, title, experiment, baseline_method, compare_methods):
    print()
    print("=" * 110)
    print(f"CTA budget: {num_blocks} | {title}")
    print("=" * 110)

    header = f"{'MiB':>8} | {'baseline ms':>11} {'GB/s':>8}"
    for label, _ in compare_methods:
        header += f" | {label + ' ms':>13} {'spd':>8} {'GB/s':>8}"
    print(header)
    print("-" * len(header))

    for b in sizes:
        base = row_map(rows, num_blocks, b, experiment, baseline_method)
        line = f"{b / (1024 * 1024):8.1f} | {fmt_ms(base):>11} {fmt_gbps(base):>8}"

        for _, method in compare_methods:
            other = row_map(rows, num_blocks, b, experiment, method)
            line += (
                f" | {fmt_ms(other):>13}"
                f" {fmt_speedup(base, other):>8}"
                f" {fmt_gbps(other):>8}"
            )

        print(line)


def print_summary(rows, sizes, num_blocks_list):
    for nb in num_blocks_list:
        print_section(
            rows,
            nb,
            sizes,
            "reduce_two_to_one: peer src0 + peer src1 -> local dst",
            0,
            0,
            [
                ("split_fwd", 1),
                ("split_opp", 4),
            ],
        )

        print_section(
            rows,
            nb,
            sizes,
            "copy_one_to_two: local src -> peer dst0 + peer dst1",
            1,
            2,
            [
                ("fanout", 3),
            ],
        )

        print_section(
            rows,
            nb,
            sizes,
            "reduce_one_to_two: local src -> local dst0 + peer dst1",
            2,
            5,
            [
                ("red_fanout", 6),
            ],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-bytes", type=int, default=512 * 1024)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument(
        "--num-blocks-list",
        type=str,
        default=None,
        help="Comma-separated CTA budgets, e.g. 2,4,8. Overrides --num-blocks.",
    )
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    sizes = make_power_of_two_sizes(args.min_bytes, args.max_bytes)
    num_blocks_list = (
        parse_int_list(args.num_blocks_list)
        if args.num_blocks_list is not None
        else [args.num_blocks]
    )

    ooverlap_ext = load_ooverlap_ext()

    rows = ooverlap_ext.benchmark_tma_batch_experiment_sweep_sm90(
        sizes,
        num_blocks_list,
        args.iters,
        args.warmup,
        args.dev0,
        args.dev1,
    )

    rows = enrich(rows)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}")

    print_summary(rows, sizes, num_blocks_list)

    print()
    print("method ids:")
    for k in sorted(METHOD_NAMES):
        print(f"  {k}: {METHOD_NAMES[k]}")


if __name__ == "__main__":
    main()
