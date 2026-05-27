#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def best_by_ms(rows, metric_name):
    best = None

    for row in rows:
        if row.get("status") != "ok":
            continue

        metric = row.get("metrics", {}).get(metric_name)
        if not metric:
            continue

        ms = metric.get("ms")
        if ms is None:
            continue

        item = {
            "ms": float(ms),
            "tflops": metric.get("tflops"),
            "algo": metric.get("algo"),
            "config": row.get("config"),
        }

        if best is None or item["ms"] < best["ms"]:
            best = item

    return best


def matrix_key(row):
    m = row.get("matrix", {}).get("M")
    n = row.get("matrix", {}).get("N")
    k = row.get("matrix", {}).get("K")
    if m is None or n is None or k is None:
        return None
    return int(m), int(n), int(k)


def fmt(x, ndigits=4):
    if x is None:
        return "NA"
    return f"{x:.{ndigits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/top_noncoop_gemm_sm90.json",
        help="Input JSON from bench_top_noncoop_gemm_sm90.py",
    )
    parser.add_argument(
        "--out",
        default="results/top_noncoop_gemm_summary.txt",
        help="Output text/CSV file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text())
    runs = data.get("runs", [])

    grouped = {}
    for row in runs:
        key = matrix_key(row)
        if key is None:
            continue
        grouped.setdefault(key, []).append(row)

    lines = []
    lines.append(
        ",".join(
            [
                "M*N*K",
                "cublasbaseline_ms",
                "plain_ms",
                "reorder_ms",
                "plain_speedup_vs_cublas",
                "reorder_speedup_vs_plain",
                "plain_algo",
                "reorder_algo",
                "cublas_tflops",
                "plain_tflops",
                "reorder_tflops",
            ]
        )
    )

    for key in sorted(grouped):
        rows = grouped[key]
        m, n, k = key

        cublas = best_by_ms(rows, "baseline_gemm_col")
        plain = best_by_ms(rows, "plain_sm90")
        reorder = best_by_ms(rows, "signal_sm90")

        cublas_ms = cublas["ms"] if cublas else None
        plain_ms = plain["ms"] if plain else None
        reorder_ms = reorder["ms"] if reorder else None

        plain_speedup = None
        reorder_speedup = None

        if cublas_ms and plain_ms:
            plain_speedup = cublas_ms / plain_ms

        if plain_ms and reorder_ms:
            reorder_speedup = plain_ms / reorder_ms

        lines.append(
            ",".join(
                [
                    f"{m}*{n}*{k}",
                    fmt(cublas_ms),
                    fmt(plain_ms),
                    fmt(reorder_ms),
                    fmt(plain_speedup),
                    fmt(reorder_speedup),
                    str(plain.get("algo", "NA") if plain else "NA"),
                    str(reorder.get("algo", "NA") if reorder else "NA"),
                    fmt(cublas.get("tflops") if cublas else None, 2),
                    fmt(plain.get("tflops") if plain else None, 2),
                    fmt(reorder.get("tflops") if reorder else None, 2),
                ]
            )
        )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
