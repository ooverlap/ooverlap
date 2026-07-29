#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {2|4} {AR|RS|AG} NUMEL [NUMEL ...]" >&2
  echo "Example: $0 2 AR 1048576 4194304 16777216 33554432" >&2
  exit 2
}

[[ $# -ge 3 ]] || usage
WORLD_SIZE="$1"
COLLECTIVE_ARG="${2^^}"
shift 2
NUMELS=("$@")

case "$COLLECTIVE_ARG" in
  AR)
    COLLECTIVE="allreduce"
    ;;
  RS)
    COLLECTIVE="reduce_scatter"
    ;;
  AG)
    COLLECTIVE="all_gather"
    ;;
  *)
    usage
    ;;
esac

# Match the unrestricted settings in run_external_p2p_bandwidth_latency.sh.
case "$WORLD_SIZE" in
  2)
    DEVICES="0,1"
    OOVERLAP_CTAS="8"
    MAX_CTAS_PER_REDUCE_TASK="8"
    ;;
  4)
    DEVICES="0,1,2,3"
    OOVERLAP_CTAS="9"
    MAX_CTAS_PER_REDUCE_TASK="3"
    ;;
  *)
    usage
    ;;
esac

for numel in "${NUMELS[@]}"; do
  [[ "$numel" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: NUMEL must be a positive integer; got: $numel" >&2
    exit 2
  }
  if [[ "$COLLECTIVE" != "allreduce" ]] && ((numel % WORLD_SIZE != 0)); then
    echo "error: $COLLECTIVE requires NUMEL divisible by world size $WORLD_SIZE; got: $numel" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DRIVER="$REPO_ROOT/test/test_external_p2p_collective.py"
ANALYZER="$REPO_ROOT/tool/nsys_sm_usage.py"
TUNING_POLICY="$REPO_ROOT/results/policies/tp4_policy.json"
PYTHON_BIN="${PYTHON_BIN:-python}"
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-100}"

OUT_DIR="$REPO_ROOT/results/evalution/external_p2p/sm_usage/tp${WORLD_SIZE}/unres/${COLLECTIVE}"
OUT_FILE="$OUT_DIR/sm_usage.txt"

[[ -f "$DRIVER" ]] || {
  echo "error: benchmark driver not found: $DRIVER" >&2
  exit 1
}
[[ -f "$ANALYZER" ]] || {
  echo "error: analyzer not found: $ANALYZER" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "error: Python executable not found: $PYTHON_BIN" >&2
  exit 1
}
command -v nsys >/dev/null 2>&1 || {
  echo "error: nsys not found" >&2
  exit 1
}

unset OOVERLAP_BENCH_ONLY
unset OOVERLAP_MAX_CTAS
unset OOVERLAP_MAX_CTAS_PER_REDUCE_TASK
unset NCCL_MIN_CTAS
unset NCCL_MAX_CTAS
unset OOVERLAP_TUNING_POLICY

export CUDA_VISIBLE_DEVICES="$DEVICES"

if [[ -f "$TUNING_POLICY" ]]; then
  export OOVERLAP_TUNING_POLICY="$TUNING_POLICY"
  TUNING_POLICY_LABEL="$TUNING_POLICY"
else
  echo "[evalution] warning: tuning policy not found: $TUNING_POLICY; continuing without it" >&2
  TUNING_POLICY_LABEL="not found (runtime fallback)"
fi

SMS_PER_DEVICE="$($PYTHON_BIN - "$WORLD_SIZE" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != expected:
    raise SystemExit(
        f"expected {expected} visible GPUs, found {torch.cuda.device_count()}"
    )
counts = [
    torch.cuda.get_device_properties(device).multi_processor_count
    for device in range(expected)
]
if len(set(counts)) != 1:
    raise SystemExit(f"profiled GPUs have different SM counts: {counts}")
print(counts[0])
PY
)"

mkdir -p "$OUT_DIR"
{
  echo "# collective=$COLLECTIVE"
  echo "# collective_argument=$COLLECTIVE_ARG"
  echo "# world_size=$WORLD_SIZE"
  echo "# devices=$DEVICES"
  echo "# ooverlap_max_ctas=$OOVERLAP_CTAS"
  echo "# max_ctas_per_reduce_task=$MAX_CTAS_PER_REDUCE_TASK"
  echo "# nccl_ctas=default"
  echo "# tuning_policy=$TUNING_POLICY_LABEL"
  echo "# sms_per_device=$SMS_PER_DEVICE"
  echo "# iters=$ITERS warmup=$WARMUP"
  printf "numel\tbytes\tooverlap_avg_gpu_median_active_sms\tnccl_avg_gpu_median_active_sms\tnccl_symmetric_avg_gpu_median_active_sms\n"
} > "$OUT_FILE"

cd "$REPO_ROOT"

for numel in "${NUMELS[@]}"; do
  declare -A RESULT=()

  for backend in ooverlap nccl nccl_symmetric; do
    REPORT="$OUT_DIR/${backend}_numel${numel}"
    echo "[evalution] collective=$COLLECTIVE backend=$backend numel=$numel report=$REPORT"

    if [[ "$backend" == "ooverlap" ]]; then
      OOVERLAP_BENCH_ONLY="$backend" \
      OOVERLAP_MAX_CTAS="$OOVERLAP_CTAS" \
      OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="$MAX_CTAS_PER_REDUCE_TASK" \
      nsys profile \
        --force-overwrite=true \
        --trace=cuda \
        --gpu-metrics-devices=cuda-visible \
        --gpu-metrics-frequency=20000 \
        --output="$REPORT" \
        "$PYTHON_BIN" "$DRIVER" \
          --mode bench \
          --collective "$COLLECTIVE" \
          --numel "$numel" \
          --iters "$ITERS" \
          --warmup "$WARMUP" \
          --devices "$DEVICES"
    else
      OOVERLAP_BENCH_ONLY="$backend" \
      nsys profile \
        --force-overwrite=true \
        --trace=cuda \
        --gpu-metrics-devices=cuda-visible \
        --gpu-metrics-frequency=20000 \
        --output="$REPORT" \
        "$PYTHON_BIN" "$DRIVER" \
          --mode bench \
          --collective "$COLLECTIVE" \
          --numel "$numel" \
          --iters "$ITERS" \
          --warmup "$WARMUP" \
          --devices "$DEVICES"
    fi

    nsys export \
      --type sqlite \
      --force-overwrite=true \
      --output="${REPORT}.sqlite" \
      "${REPORT}.nsys-rep"

    RESULT["$backend"]="$($PYTHON_BIN "$ANALYZER" \
      "${REPORT}.sqlite" \
      --sms-per-device "$SMS_PER_DEVICE" \
      --value-only)"
  done

  bytes=$((numel * 2))
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$numel" \
    "$bytes" \
    "${RESULT[ooverlap]}" \
    "${RESULT[nccl]}" \
    "${RESULT[nccl_symmetric]}" | tee -a "$OUT_FILE"
done

echo "[evalution] wrote: $OUT_FILE"
