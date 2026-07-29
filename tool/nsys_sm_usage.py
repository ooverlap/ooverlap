#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report median active SM usage from an Nsight Systems SQLite export. "
            "The final value is the average of the per-GPU medians."
        )
    )
    parser.add_argument("sqlite", type=Path, help="Nsight Systems SQLite export")
    parser.add_argument(
        "--sms-per-device",
        type=int,
        required=True,
        help="physical SM count on each profiled GPU",
    )
    parser.add_argument(
        "--value-only",
        action="store_true",
        help="print only average_median_sms",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sms_per_device <= 0:
        raise SystemExit("--sms-per-device must be positive")
    if not args.sqlite.is_file():
        raise SystemExit(f"SQLite file not found: {args.sqlite}")

    query = """
    SELECT
        (g.typeId & 255) AS gpu_id,
        g.value
    FROM GPU_METRICS AS g
    JOIN TARGET_INFO_GPU_METRICS AS m
      ON g.typeId = m.typeId
     AND g.metricId = m.metricId
    WHERE m.metricName LIKE 'SMs Active%'
    ORDER BY gpu_id, g.timestamp
    """

    samples: dict[int, list[float]] = defaultdict(list)
    with sqlite3.connect(args.sqlite) as connection:
        for gpu_id, value in connection.execute(query):
            samples[int(gpu_id)].append(float(value))

    if not samples:
        raise SystemExit("No 'SMs Active' samples found")

    rows: list[tuple[int, int, float, float]] = []
    for gpu_id, values in sorted(samples.items()):
        active_values = [value for value in values if value > 0.0]
        if not active_values:
            raise SystemExit(f"GPU {gpu_id} has no active 'SMs Active' samples")

        median_percent = statistics.median(active_values)
        median_sms = median_percent * args.sms_per_device / 100.0
        rows.append((gpu_id, len(active_values), median_percent, median_sms))

    average_median_sms = statistics.fmean(row[3] for row in rows)

    if args.value_only:
        print(f"{average_median_sms:.6f}")
        return

    print("gpu\tactive_samples\tmedian_active_percent\tmedian_active_sms")
    for gpu_id, active_count, median_percent, median_sms in rows:
        print(f"{gpu_id}\t{active_count}\t{median_percent:.6f}\t{median_sms:.6f}")
    print(f"average_median_sms\t{average_median_sms:.6f}")


if __name__ == "__main__":
    main()
