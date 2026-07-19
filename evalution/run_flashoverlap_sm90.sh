#!/usr/bin/env bash
set -euo pipefail

# Fixed paper-evaluation pipeline for the SM90 FlashOverlap experiment.
#
# Usage:
#   ./evalution/run_flashoverlap_sm90.sh 2
#   ./evalution/run_flashoverlap_sm90.sh 4
#
# Optional OOverlap-only overrides:
#   OOVERLAP_MAX_CTAS=8 \
#   OOVERLAP_MAX_CTAS_PER_REDUCE_TASK=4 \
#     ./evalution/run_flashoverlap_sm90.sh 4
#
# The only accepted argument is the tensor-parallel/world size. Devices,
# bandwidth settings, evaluation settings, and plotting settings are fixed here.

usage() {
  echo "Usage: $0 {2|4}" >&2
  exit 2
}

fail() {
  echo "error: $*" >&2
  exit 1
}

require_positive_integer() {
  local name="$1"
  local value="$2"

  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    fail "$name must be a positive integer; got: $value"
  fi
}

[[ $# -eq 1 ]] || usage
WORLD_SIZE="$1"

case "$WORLD_SIZE" in
  2)
    DEVICES="0,1"
    DEFAULT_OOVERLAP_MAX_CTAS="8"
    DEFAULT_OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="8"
    ;;
  4)
    DEVICES="0,1,2,3"
    DEFAULT_OOVERLAP_MAX_CTAS="12"
    DEFAULT_OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="4"
    ;;
  *)
    echo "error: world size must be exactly 2 or 4; got: $WORLD_SIZE" >&2
    usage
    ;;
esac

# These variables are passed only to OOverlap commands. They are deliberately
# removed from the NCCL bandwidth command's environment.
OOVERLAP_MAX_CTAS="${OOVERLAP_MAX_CTAS:-$DEFAULT_OOVERLAP_MAX_CTAS}"
OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="${OOVERLAP_MAX_CTAS_PER_REDUCE_TASK:-$DEFAULT_OOVERLAP_MAX_CTAS_PER_REDUCE_TASK}"

require_positive_integer "OOVERLAP_MAX_CTAS" "$OOVERLAP_MAX_CTAS"
require_positive_integer \
  "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK" \
  "$OOVERLAP_MAX_CTAS_PER_REDUCE_TASK"

# Fixed bandwidth-curve settings. The curve uses the mean latency because
# neither --use-median nor --use-p90 is passed to tool/bandwidth.py.
BANDWIDTH_WARMUP="20"
BANDWIDTH_ITERS="200"
BANDWIDTH_SLEEP_SECONDS="5.0"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

BANDWIDTH_SCRIPT="$REPO_ROOT/tool/bandwidth.py"
EVALUATE_SCRIPT="$REPO_ROOT/tool/evaluate_sm90.py"
PLOT_SCRIPT="$REPO_ROOT/test/plot_eval_sm90.py"
EXTENSION="$REPO_ROOT/build/lib/ooverlap_ext.so"
ALGO_DICT="$REPO_ROOT/configs/AlgoDictSm90.json"

CONFIG_DIR="$REPO_ROOT/configs"
EVAL_OUT_DIR="$REPO_ROOT/results/eval_sm90"
PLOT_OUT_DIR="$EVAL_OUT_DIR/operator_speedup"
PIPELINE_LOG_DIR="$EVAL_OUT_DIR/pipeline_logs"
EVAL_JSON="$EVAL_OUT_DIR/evaluation_result.json"

NCCL_BANDWIDTH_PT="$CONFIG_DIR/bandwidth_nccl_all_reduce_tp${WORLD_SIZE}.pt"
OOVERLAP_BANDWIDTH_PT="$CONFIG_DIR/bandwidth_ooverlap_all_reduce_tp${WORLD_SIZE}.pt"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"
command -v tee >/dev/null 2>&1 || fail "tee was not found"

for required in "$BANDWIDTH_SCRIPT" "$EVALUATE_SCRIPT" "$PLOT_SCRIPT" "$ALGO_DICT"; do
  [[ -f "$required" ]] || fail "required file not found: $required"
done

[[ -f "$EXTENSION" ]] || fail "extension not found: $EXTENSION; build the repository first"

mkdir -p "$CONFIG_DIR" "$EVAL_OUT_DIR" "$PLOT_OUT_DIR" "$PIPELINE_LOG_DIR"
cd "$REPO_ROOT"

# Keep the selected physical devices visible to the evaluator parent as well as
# its child processes. The fixed device lists are contiguous, so bandwidth.py
# can safely address the visible devices as 0..WORLD_SIZE-1.
export CUDA_VISIBLE_DEVICES="$DEVICES"
export PYTHONUNBUFFERED="1"
export MPLBACKEND="Agg"

# Verify the selected device count and SM capability before starting a long run.
"$PYTHON_BIN" - "$WORLD_SIZE" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the selected Python environment")

count = torch.cuda.device_count()
if count != expected:
    raise SystemExit(
        f"expected {expected} visible CUDA devices, but torch sees {count}; "
        f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

for index in range(count):
    major, minor = torch.cuda.get_device_capability(index)
    name = torch.cuda.get_device_name(index)
    print(f"[evalution] cuda:{index} name={name} capability={major}.{minor}")
    if major < 9:
        raise SystemExit(
            f"cuda:{index} has capability {major}.{minor}; eval_sm90 requires SM90 or newer"
        )
PY

# With --skip-existing-profile, a CUTLASS CSV is needed only when the matching
# GPU-specific packed config is absent or contains no Algo entries.
"$PYTHON_BIN" - "$EVALUATE_SCRIPT" <<'PY'
import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("ooverlap_evaluate_sm90_config", path)
if spec is None or spec.loader is None:
    raise SystemExit(f"could not import {path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

if not module.SHAPES:
    raise SystemExit("tool/evaluate_sm90.py has an empty SHAPES list")

missing = []
for shape in module.SHAPES:
    config_path = module.shape_config_path(shape)
    if module.config_has_algos(config_path):
        print(f"[evalution] profile config exists: {config_path}")
        continue

    csv_path = module.csv_path_for_shape(shape)
    if csv_path.is_file():
        print(f"[evalution] profile CSV ready: {csv_path}")
    else:
        missing.append((shape, csv_path, config_path))

if missing:
    lines = ["missing CUTLASS profile CSVs for shapes without reusable configs:"]
    for shape, csv_path, config_path in missing:
        lines.append(
            f"  M={shape.m} N={shape.n} K={shape.k}: csv={csv_path} config={config_path}"
        )
    raise SystemExit("\n".join(lines))
PY

run_bandwidth_if_missing() {
  local backend="$1"
  local output_pt="$2"
  local log_path="$3"

  if [[ -s "$output_pt" ]]; then
    echo "[evalution] bandwidth curve exists; skipping $backend: $output_pt"
    return
  fi

  echo "[evalution] generating $backend all-reduce bandwidth curve for TP=$WORLD_SIZE"

  if [[ "$backend" == "ooverlap" ]]; then
    OOVERLAP_MAX_CTAS="$OOVERLAP_MAX_CTAS" \
    OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="$OOVERLAP_MAX_CTAS_PER_REDUCE_TASK" \
      "$PYTHON_BIN" "$BANDWIDTH_SCRIPT" \
        --comm_backend "$backend" \
        --comm_op all_reduce \
        --devices "$DEVICES" \
        --warmup "$BANDWIDTH_WARMUP" \
        --iters "$BANDWIDTH_ITERS" \
        --sleep-seconds "$BANDWIDTH_SLEEP_SECONDS"
  else
    # Prevent caller-exported OOverlap controls from leaking into the NCCL run.
    (
      unset OOVERLAP_MAX_CTAS
      unset OOVERLAP_MAX_CTAS_PER_REDUCE_TASK

      "$PYTHON_BIN" "$BANDWIDTH_SCRIPT" \
        --comm_backend "$backend" \
        --comm_op all_reduce \
        --devices "$DEVICES" \
        --warmup "$BANDWIDTH_WARMUP" \
        --iters "$BANDWIDTH_ITERS" \
        --sleep-seconds "$BANDWIDTH_SLEEP_SECONDS"
    )
  fi 2>&1 | tee "$log_path"

  [[ -s "$output_pt" ]] || fail "bandwidth command completed but did not create: $output_pt"
}

echo "[evalution] SM90 FlashOverlap paper pipeline"
echo "[evalution] world_size=$WORLD_SIZE devices=$DEVICES"
echo "[evalution] ooverlap_max_ctas=$OOVERLAP_MAX_CTAS"
echo "[evalution] ooverlap_max_ctas_per_reduce_task=$OOVERLAP_MAX_CTAS_PER_REDUCE_TASK"
echo "[evalution] bandwidth_warmup=$BANDWIDTH_WARMUP"
echo "[evalution] bandwidth_iters=$BANDWIDTH_ITERS"
echo "[evalution] bandwidth_sleep_seconds=$BANDWIDTH_SLEEP_SECONDS"
echo "[evalution] evaluation_output=$EVAL_OUT_DIR"
echo "[evalution] plot_output=$PLOT_OUT_DIR"

run_bandwidth_if_missing \
  "nccl" \
  "$NCCL_BANDWIDTH_PT" \
  "$PIPELINE_LOG_DIR/tp${WORLD_SIZE}_bandwidth_nccl.log"

run_bandwidth_if_missing \
  "ooverlap" \
  "$OOVERLAP_BANDWIDTH_PT" \
  "$PIPELINE_LOG_DIR/tp${WORLD_SIZE}_bandwidth_ooverlap.log"

echo "[evalution] running SM90 search and overlap evaluation"
OOVERLAP_MAX_CTAS="$OOVERLAP_MAX_CTAS" \
OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="$OOVERLAP_MAX_CTAS_PER_REDUCE_TASK" \
  "$PYTHON_BIN" "$EVALUATE_SCRIPT" \
    --devices "$DEVICES" \
    --out-dir "$EVAL_OUT_DIR" \
    --skip-existing-profile \
    2>&1 | tee "$PIPELINE_LOG_DIR/tp${WORLD_SIZE}_evaluate_sm90.log"

[[ -s "$EVAL_JSON" ]] || fail "evaluation completed but did not create: $EVAL_JSON"

echo "[evalution] generating operator-level plots and summaries"
"$PYTHON_BIN" "$PLOT_SCRIPT" \
  --json "$EVAL_JSON" \
  --out-dir "$PLOT_OUT_DIR" \
  --name "operator_overlap_speedup_by_shape" \
  --baseline-source "nccl_cublas" \
  --test-mode "both" \
  --all-shapes \
  --tps "2,4" \
  --annotate \
  2>&1 | tee "$PIPELINE_LOG_DIR/tp${WORLD_SIZE}_plot_eval_sm90.log"

echo "[evalution] complete"
echo "[evalution] evaluation_json=$EVAL_JSON"
echo "[evalution] plots=$PLOT_OUT_DIR"
echo "[evalution] logs=$PIPELINE_LOG_DIR"
