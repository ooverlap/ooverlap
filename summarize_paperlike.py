import csv
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("OUT", Path(os.environ["OOTMP"]) / "vllm_eager_paperlike"))
backends = ["pynccl", "ooverlap"]
bszs = [1, 4, 16, 64, 128, 256, 512]

cases = {
    "decode": {"input_len": 16, "output_len": 128},
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
                "requests_per_s": bsz / elapsed,
                "total_tokens_per_s": r.get("tokens_per_second", ""),
                "output_tokens_per_s": (bsz * output_len) / elapsed,
                "decode_step_ms": (elapsed / output_len * 1000.0) if kind == "decode" else "",
            })

csv_path = OUT / "summary_pynccl_vs_ooverlap.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {csv_path}")
print()

for kind in ["decode", "prefill"]:
    print(f"================ {kind.upper()} ================")
    for bsz in bszs:
        print(f"\nbsz={bsz}")
        for backend in backends:
            r = next(x for x in rows if x["kind"] == kind and x["backend"] == backend and x["bsz"] == bsz)
            if kind == "decode":
                print(
                    f"{backend:8s} elapsed={r['elapsed_s']:.4f}s "
                    f"decode_step={float(r['decode_step_ms']):.3f}ms "
                    f"out_tok/s={r['output_tokens_per_s']:.1f} "
                    f"speedup_vs_pynccl={r['speedup_vs_pynccl']:.4f}x"
                )
            else:
                print(
                    f"{backend:8s} elapsed={r['elapsed_s']:.4f}s "
                    f"req/s={r['requests_per_s']:.2f} "
                    f"speedup_vs_pynccl={r['speedup_vs_pynccl']:.4f}x"
                )
