from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("aggregate_vllm_backend_orders.py")
SPEC = importlib.util.spec_from_file_location("aggregate_vllm_backend_orders", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
aggregate_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aggregate_module
SPEC.loader.exec_module(aggregate_module)


FIELDS = [
    "model",
    "backend",
    "baseline_backend",
    "devices",
    "tensor_parallel_size",
    "workload",
    "input_len",
    "output_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "num_prompts",
    "num_warmups",
    "max_model_len",
    "gpu_memory_utilization",
    "repetitions_ok",
    "elapsed_mean_s",
    "elapsed_std_s",
    "requests_per_s_mean",
    "total_tokens_per_s_mean",
    "output_tokens_per_s_mean",
    "speedup_vs_base_mean",
    "change_vs_base_pct_mean",
]


class BackendOrderAggregationTest(unittest.TestCase):
    def write_order(self, root: Path, name: str, throughputs: dict[str, float]) -> None:
        order = root / name
        batch = order / "batch_scaling"
        batch.mkdir(parents=True)
        backends = tuple(throughputs)
        (order / "backend_order.txt").write_text(",".join(backends) + "\n")
        with (batch / "summary_aggregate.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for backend, throughput in throughputs.items():
                writer.writerow(
                    {
                        "model": "Qwen",
                        "backend": backend,
                        "baseline_backend": "pynccl",
                        "devices": "[0, 1, 2, 3]",
                        "tensor_parallel_size": 4,
                        "workload": "long-decode",
                        "input_len": 512,
                        "output_len": 1024,
                        "max_num_seqs": 32,
                        "max_num_batched_tokens": 4096,
                        "num_prompts": 64,
                        "num_warmups": 32,
                        "max_model_len": 8192,
                        "gpu_memory_utilization": 0.85,
                        "repetitions_ok": 1,
                        "elapsed_mean_s": 1000.0 / throughput,
                        "elapsed_std_s": 0.0,
                        "requests_per_s_mean": throughput / 1024.0,
                        "total_tokens_per_s_mean": throughput * 1.5,
                        "output_tokens_per_s_mean": throughput,
                        "speedup_vs_base_mean": "",
                        "change_vs_base_pct_mean": "",
                    }
                )

    def test_averages_available_orders_and_pairs_speedups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "backend_orders"
            self.write_order(
                root,
                "order-01__ooverlap__auto__pynccl",
                {"ooverlap": 15.0, "auto": 12.0, "pynccl": 10.0},
            )
            self.write_order(
                root,
                "order-02__ooverlap__pynccl__auto",
                {"ooverlap": 30.0, "pynccl": 20.0, "auto": 18.0},
            )
            # Models an interrupted order: auto completed, the other backends did not.
            self.write_order(
                root,
                "order-03__auto__ooverlap__pynccl",
                {"auto": 30.0},
            )

            sources = aggregate_module.discover_sources(root)
            rows = aggregate_module.load_order_rows(sources)
            aggregates = aggregate_module.aggregate_order_rows(
                rows,
                baseline_backend="pynccl",
                orders_found=3,
                orders_planned=6,
            )
            by_backend = {row["backend"]: row for row in aggregates}

            self.assertEqual(by_backend["auto"]["orders_ok"], 3)
            self.assertEqual(by_backend["pynccl"]["orders_ok"], 2)
            self.assertEqual(by_backend["ooverlap"]["orders_ok"], 2)
            self.assertAlmostEqual(by_backend["auto"]["output_tokens_per_s_mean"], 20.0)
            self.assertAlmostEqual(by_backend["pynccl"]["output_tokens_per_s_mean"], 15.0)
            self.assertAlmostEqual(by_backend["ooverlap"]["output_tokens_per_s_mean"], 22.5)

            # auto's unpaired third-order sample affects its throughput mean, but
            # not its speedup against pynccl.
            self.assertEqual(by_backend["auto"]["paired_orders_vs_base"], 2)
            self.assertAlmostEqual(by_backend["auto"]["speedup_vs_base_mean"], 1.05)
            self.assertEqual(by_backend["ooverlap"]["paired_orders_vs_base"], 2)
            self.assertAlmostEqual(by_backend["ooverlap"]["speedup_vs_base_mean"], 1.5)

            output = root / "aggregate" / "batch_scaling"
            order_count, row_count = aggregate_module.aggregate(
                root,
                output,
                baseline_backend="pynccl",
                target_backend="ooverlap",
                comparison_backends=("auto", "pynccl"),
            )
            self.assertEqual(order_count, 3)
            self.assertEqual(row_count, 3)
            summary = (output / "summary.txt").read_text()
            self.assertIn("orders=3/3", summary)
            self.assertIn("orders=2/3", summary)
            self.assertIn("Coverage warning", summary)


if __name__ == "__main__":
    unittest.main()
