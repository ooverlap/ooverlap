import csv
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("OUT", Path(os.environ["OOTMP"]) / "vllm_eager_paperlike"))
backends = os.environ.get(
    "BACKENDS",
    "pynccl ooverlap",
).split()

bszs = [
    int(value)
    for value in os.environ.get(
        "BSZS",
        "128 256 512",
    ).split()
]

cases = {
    "decode": {"input_len": 16, "output_len": 256},
    "mixed": {"input_len": 256, "output_len": 128},
    "long": {"input_len": 1024, "output_len": 128},
    "prefill": {"input_len": 1024, "output_len": 1},
}

def load(kind, backend, bsz):
    p = OUT / f"{kind}_backend_{backend}_bsz_{bsz}.json"
    with p.open() as f:
        return json.load(f)

rows = []

for kind, cfg in cases.items():
    for bsz in bszs:
        base = load(kind, "pynccl", bsz)
        base_elapsed = base["elapsed_time"]

        for backend in backends:
            r = load(kind, backend, bsz)
            num_requests = r["num_requests"]
            elapsed = r["elapsed_time"]
            output_len = cfg["output_len"]
            input_len = cfg["input_len"]

            rows.append({
                "kind": kind,
                "backend": backend,
                "bsz": bsz,
                "input_len": input_len,
                "output_len": output_len,
                "elapsed_s": elapsed,
                "speedup_vs_pynccl": base_elapsed / elapsed,
                "num_requests": num_requests,
                "requests_per_s": r["requests_per_second"],
                "total_tokens_per_s": r["tokens_per_second"],
                "output_tokens_per_s": (
                    num_requests * output_len
                ) / elapsed,
            })

csv_path = OUT / "summary_pynccl_vs_ooverlap.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {csv_path}")
print()

for kind in cases:
    print(f"================ {kind.upper()} ================")

    for bsz in bszs:
        print(f"\nmax_seqs={bsz}")

        for backend in backends:
            r = next(
                x
                for x in rows
                if x["kind"] == kind
                and x["backend"] == backend
                and x["bsz"] == bsz
            )

            print(
                f"{backend:8s} "
                f"elapsed={r['elapsed_s']:.4f}s "
                f"req/s={r['requests_per_s']:.2f} "
                f"out_tok/s={r['output_tokens_per_s']:.1f} "
                f"speedup={r['speedup_vs_pynccl']:.4f}x"
            )
