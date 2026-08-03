#!/usr/bin/env bash
set -euo pipefail

# Fixed paper-evaluation wrapper for external-P2P collective latency/bandwidth.
#
# Usage:
#   ./evalution/run_external_p2p_bandwidth_latency.sh 2 unres
#   ./evalution/run_external_p2p_bandwidth_latency.sh 2 res
#   ./evalution/run_external_p2p_bandwidth_latency.sh 4 unres
#   ./evalution/run_external_p2p_bandwidth_latency.sh 4 res
#
# The first argument is the tensor-parallel/world size. The second selects the
# unrestricted or restricted CTA configuration.
# OOVERLAP_EXTERNAL_P2P_TP_CTA_MODE_V1

usage() {
  echo "Usage: $0 {2|4} {unres|res}" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
WORLD_SIZE="$1"
CTA_MODE="$2"

# Edit only the CTA values in these four branches for the paper scenarios.
# "default" leaves the OOverlap setting unset; -1 leaves NCCL_MAX_CTAS unset.
case "$WORLD_SIZE:$CTA_MODE" in
  2:unres)
    DEVICES="0,1"
    OOVERLAP_MAX_CTAS="16"
    NCCL_MAX_CTAS="-1"
    MAX_CTAS_PER_REDUCE_TASK="16"
    ;;
  2:res)
    DEVICES="0,1"
    OOVERLAP_MAX_CTAS="4"
    NCCL_MAX_CTAS="4"
    MAX_CTAS_PER_REDUCE_TASK="4"
    ;;
  4:unres)
    DEVICES="0,1,2,3"
    OOVERLAP_MAX_CTAS="18"
    NCCL_MAX_CTAS="-1"
    MAX_CTAS_PER_REDUCE_TASK="6"
    ;;
  4:res)
    DEVICES="0,1,2,3"
    OOVERLAP_MAX_CTAS="9"
    NCCL_MAX_CTAS="9"
    MAX_CTAS_PER_REDUCE_TASK="3"
    ;;
  *)
    echo "error: expected world size 2 or 4 and CTA mode unres or res; got: $WORLD_SIZE $CTA_MODE" >&2
    usage
    ;;
esac

if [[ "$NCCL_MAX_CTAS" == "-1" ]]; then
  NCCL_MAX_CTAS_LABEL="default"
elif [[ "$NCCL_MAX_CTAS" =~ ^[1-9][0-9]*$ ]]; then
  NCCL_MAX_CTAS_LABEL="$NCCL_MAX_CTAS"
else
  echo "error: NCCL_MAX_CTAS must be -1 or a positive integer; got: $NCCL_MAX_CTAS" >&2
  exit 2
fi

# Keep the fixed latency range at 1 KiB and above. The external-P2P sweep can
# stall in the very-small-message 8 B..512 B range on the current runtime,
# while this range is known to complete on the same installation.
LATENCY_BYTES="1K,2K,4K,8K,16K,32K,64K,128K,256K,512K"
BANDWIDTH_BYTES="1M,2M,4M,8M,16M,32M,64M,128M,256M,512M"
ITERS="150"
WARMUP="20"
RING_SIZE="16"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DRIVER="$REPO_ROOT/test/test_external_p2p_collective_sweep.py"
PLOTTER="$REPO_ROOT/test/plot_external_p2p_collective.py"
TUNING_POLICY="$REPO_ROOT/results/policies/tp4_policy.json"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUT_DIR="$REPO_ROOT/results/evalution/external_p2p/tp${WORLD_SIZE}/${CTA_MODE}"
OUT_PREFIX="$OUT_DIR/external_p2p_bandwidth_latency"
PLOT_PREFIX="$OUT_DIR/external_p2p_collective"

[[ -f "$DRIVER" ]] || {
  echo "error: benchmark driver not found: $DRIVER" >&2
  exit 1
}

[[ -f "$PLOTTER" ]] || {
  echo "error: plotter not found: $PLOTTER" >&2
  exit 1
}

# OOVERLAP_EXTERNAL_P2P_TUNING_POLICY_OPTIONAL_V1
if [[ -f "$TUNING_POLICY" ]]; then
  TUNING_POLICY_ARGS=(--tuning-policy "$TUNING_POLICY")
  TUNING_POLICY_LABEL="$TUNING_POLICY"
else
  echo "[evalution] warning: tuning policy not found: $TUNING_POLICY; continuing without it" >&2
  TUNING_POLICY_ARGS=()
  TUNING_POLICY_LABEL="not found (runtime fallback)"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "error: Python executable not found: $PYTHON_BIN" >&2
  exit 1
}

mkdir -p "$OUT_DIR"
rm -f \
  "${OUT_PREFIX}.txt" \
  "${OUT_PREFIX}.csv" \
  "${OUT_PREFIX}.jsonl" \
  "${OUT_PREFIX}_speedup.txt" \
  "${PLOT_PREFIX}.png" \
  "${PLOT_PREFIX}_latency.png" \
  "${PLOT_PREFIX}_bandwidth.png"

cd "$REPO_ROOT"

echo "[evalution] external-P2P latency and bandwidth"
echo "[evalution] world_size=$WORLD_SIZE devices=$DEVICES"
echo "[evalution] cta_mode=$CTA_MODE"
echo "[evalution] ooverlap_max_ctas=$OOVERLAP_MAX_CTAS"
echo "[evalution] nccl_max_ctas=$NCCL_MAX_CTAS_LABEL"
echo "[evalution] max_ctas_per_reduce_task=$MAX_CTAS_PER_REDUCE_TASK"
echo "[evalution] tuning_policy=$TUNING_POLICY_LABEL"
echo "[evalution] latency_bytes=$LATENCY_BYTES"
echo "[evalution] bandwidth_bytes=$BANDWIDTH_BYTES"
echo "[evalution] iters=$ITERS warmup=$WARMUP"
echo "[evalution] timing=batched-ring ring_size=$RING_SIZE"
echo "[evalution] output=${OUT_PREFIX}.txt"

# Use one worker invocation for the full fixed experiment. This intentionally
# matches the command shape verified to complete on the target machine.
OOVERLAP_BENCH_RING_SIZE="$RING_SIZE" \
"$PYTHON_BIN" "$DRIVER" \
  --mode bench \
  --use-ring-for-all-metrics \
  --collective all \
  --metric all \
  --ctas "$OOVERLAP_MAX_CTAS" \
  --nccl-ctas "$NCCL_MAX_CTAS" \
  --max-ctas-per-reduce-task "$MAX_CTAS_PER_REDUCE_TASK" \
  "${TUNING_POLICY_ARGS[@]}" \
  --latency-bytes "$LATENCY_BYTES" \
  --bandwidth-bytes "$BANDWIDTH_BYTES" \
  --bytes "$BANDWIDTH_BYTES" \
  --iters "$ITERS" \
  --warmup "$WARMUP" \
  --devices "$DEVICES" \
  --out-prefix "$OUT_PREFIX"

"$PYTHON_BIN" "$PLOTTER" \
  --text "${OUT_PREFIX}.txt" \
  --tp "$WORLD_SIZE" \
  --out-prefix "$PLOT_PREFIX"

# OOVERLAP_EXTERNAL_P2P_SPEEDUP_SUMMARY_V1
echo "[evalution] wrote text: ${OUT_PREFIX}.txt"
echo "[evalution] wrote csv:  ${OUT_PREFIX}.csv"
echo "[evalution] wrote jsonl:${OUT_PREFIX}.jsonl"
echo "[evalution] wrote speedup: ${OUT_PREFIX}_speedup.txt"
echo "[evalution] wrote plot: ${PLOT_PREFIX}.png"
