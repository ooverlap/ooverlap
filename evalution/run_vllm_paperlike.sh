#!/usr/bin/env bash
set -euo pipefail

# Fixed paper-evaluation wrapper for vLLM throughput.
#
# Usage:
#   ./evalution/run_vllm_paperlike.sh 2
#   ./evalution/run_vllm_paperlike.sh 4
#
# The only positional argument is the tensor-parallel/world size. The model,
# workloads, batching matrix, backend order, repetitions, and runtime settings
# are fixed below so the paper experiment is reproducible.

usage() {
  echo "Usage: $0 {2|4}" >&2
  exit 2
}

fail() {
  echo "error: $*" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage
WORLD_SIZE="$1"

case "$WORLD_SIZE" in
  2)
    DEVICES="0,1"
    ;;
  4)
    DEVICES="0,1,2,3"
    ;;
  *)
    echo "error: world size must be exactly 2 or 4; got: $WORLD_SIZE" >&2
    usage
    ;;
esac

# -----------------------------------------------------------------------------
# Fixed paper matrix. Edit these constants in the repository only when the
# paper methodology changes; they are intentionally not command-line options.
# -----------------------------------------------------------------------------
MODEL="Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN="2048"
GPU_MEMORY_UTILIZATION="0.85"

BACKENDS="pynccl,ooverlap"
BASELINE_BACKEND="pynccl"
WORKLOADS="decode,mixed,long,prefill"
BATCH_SIZES="64,128,256"
MAX_BATCHED_TOKENS="4096,8192"
REPETITIONS="3"
PROMPT_MULTIPLIER="4"

DATASET_NAME="random"
RANDOM_RANGE_RATIO="0.0"
SEED="0"

FLASHINFER_SAMPLER="0"
OOVERLAP_DEBUG="0"
RR_DTYPE="bf16"
RR_SLOTS="32"
RR_CAPACITY_BYTES="33554432"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRIVER="$REPO_ROOT/test/benchmark_vllm_paperlike.py"
OUT_DIR="$REPO_ROOT/results/evalution/vllm/tp${WORLD_SIZE}"
PIPELINE_LOG="$OUT_DIR/run_vllm_paperlike_tp${WORLD_SIZE}.log"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || \
  fail "Python executable not found: $PYTHON_BIN"
command -v bash >/dev/null 2>&1 || fail "bash was not found"
command -v tee >/dev/null 2>&1 || fail "tee was not found"
[[ -f "$DRIVER" ]] || fail "vLLM benchmark driver not found: $DRIVER"

resolve_runtime_env_file() {
  local candidate=""

  if [[ -n "${OOVERLAP_VLLM_ENV_FILE:-}" ]]; then
    candidate="$OOVERLAP_VLLM_ENV_FILE"
    [[ -f "$candidate" ]] || \
      fail "OOVERLAP_VLLM_ENV_FILE does not exist: $candidate"
    printf '%s\n' "$candidate"
    return
  fi

  if [[ -n "${OOTMP:-}" && -f "$OOTMP/ooverlap_vllm_env.sh" ]]; then
    printf '%s\n' "$OOTMP/ooverlap_vllm_env.sh"
    return
  fi

  # setup_env.sh normally writes:
  #   /local/tmp.*/$USER/ooverlap_vllm_env.sh
  for candidate in /local/tmp.*/"${USER:-}"/ooverlap_vllm_env.sh; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  fail "could not find ooverlap_vllm_env.sh; run 'bash ./setup_env.sh' first or set OOVERLAP_VLLM_ENV_FILE"
}

RUNTIME_ENV_FILE="$(resolve_runtime_env_file)"
mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

# Validate the generated runtime environment before starting the long matrix.
# This uses the same source semantics as the Python driver and checks only
# non-secret runtime facts: vLLM availability, extension path, and GPU count.
CHECK_CODE='import os, sys, torch
expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the vLLM runtime environment")
count = torch.cuda.device_count()
visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
if count != expected:
    raise SystemExit(f"expected {expected} visible GPUs but torch sees {count}; CUDA_VISIBLE_DEVICES={visible_devices}")
extension = os.environ.get("VLLM_OOVERLAP_TORCH_EXT", "")
if not extension or not os.path.isfile(extension):
    raise SystemExit(f"VLLM_OOVERLAP_TORCH_EXT is missing or invalid: {extension!r}")
print(f"[evalution] torch={torch.__version__} cuda={torch.version.cuda}")
for index in range(count):
    print(f"[evalution] cuda:{index} name={torch.cuda.get_device_name(index)} capability={torch.cuda.get_device_capability(index)}")
print(f"[evalution] ooverlap_extension={extension}")'

bash -lc '
  set -e
  source "$1" >/dev/null
  export CUDA_VISIBLE_DEVICES="$2"
  command -v vllm >/dev/null
  exec python -c "$3" "$4"
' "ooverlap-vllm-preflight" \
  "$RUNTIME_ENV_FILE" \
  "$DEVICES" \
  "$CHECK_CODE" \
  "$WORLD_SIZE"

echo "[evalution] vLLM paper throughput evaluation"
echo "[evalution] world_size=$WORLD_SIZE devices=$DEVICES"
echo "[evalution] runtime_env=$RUNTIME_ENV_FILE"
echo "[evalution] model=$MODEL"
echo "[evalution] backends=$BACKENDS baseline=$BASELINE_BACKEND"
echo "[evalution] workloads=$WORKLOADS"
echo "[evalution] batch_sizes=$BATCH_SIZES"
echo "[evalution] max_batched_tokens=$MAX_BATCHED_TOKENS"
echo "[evalution] repetitions=$REPETITIONS prompt_multiplier=$PROMPT_MULTIPLIER"
echo "[evalution] output=$OUT_DIR"

"$PYTHON_BIN" "$DRIVER" \
  --runtime-env-file "$RUNTIME_ENV_FILE" \
  --working-dir "$REPO_ROOT" \
  --out-dir "$OUT_DIR" \
  --models "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --backends "$BACKENDS" \
  --baseline-backend "$BASELINE_BACKEND" \
  --device-groups "$DEVICES" \
  --workloads "$WORKLOADS" \
  --batch-sizes "$BATCH_SIZES" \
  --max-batched-tokens "$MAX_BATCHED_TOKENS" \
  --repetitions "$REPETITIONS" \
  --prompt-multiplier "$PROMPT_MULTIPLIER" \
  --dataset-name "$DATASET_NAME" \
  --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --seed "$SEED" \
  --enforce-eager \
  --disable-detokenize \
  --flashinfer-sampler "$FLASHINFER_SAMPLER" \
  --ooverlap-debug "$OOVERLAP_DEBUG" \
  --rr-dtype "$RR_DTYPE" \
  --rr-slots "$RR_SLOTS" \
  --rr-capacity-bytes "$RR_CAPACITY_BYTES" \
  2>&1 | tee "$PIPELINE_LOG"

for required_output in \
  "$OUT_DIR/manifest.json" \
  "$OUT_DIR/runs.jsonl" \
  "$OUT_DIR/summary.csv" \
  "$OUT_DIR/summary_aggregate.csv" \
  "$OUT_DIR/summary.txt"; do
  [[ -s "$required_output" ]] || \
    fail "benchmark completed but required output is missing or empty: $required_output"
done

echo "[evalution] complete"
echo "[evalution] summary=$OUT_DIR/summary.txt"
echo "[evalution] aggregate_csv=$OUT_DIR/summary_aggregate.csv"
echo "[evalution] per_run_data=$OUT_DIR/runs.jsonl"
echo "[evalution] pipeline_log=$PIPELINE_LOG"
