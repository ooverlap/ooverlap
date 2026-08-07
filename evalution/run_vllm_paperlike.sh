#!/usr/bin/env bash
set -euo pipefail

# OOVERLAP_VLLM_BATCH_ONLY_BENCHMARK_V1
# OOVERLAP_VLLM_1024X128_ISOLATED_RUNS_V1
# One paper-evaluation benchmark for vLLM:
#   output throughput versus the maximum number of active sequences
#
# Workload:
#   input=1024, output=128, max_num_seqs=1,2,4,8,16,32,64
#
# Isolation:
#   all permutations of the configured three backends, process-group cleanup,
#   a short inter-run cooldown within each order, and a longer cooldown between
#   backend-order permutations
#
# Usage:
#   ./evalution/run_vllm_paperlike.sh 2 [run|plot]
#   ./evalution/run_vllm_paperlike.sh 4 [run|plot]
#
# Useful overrides:
#   VLLM_MODEL_DIR=/scratch/local/models/Qwen2.5-72B-Instruct
#   VLLM_EVAL_BACKENDS=pynccl,nccl_symm,ooverlap
#   VLLM_PLOT_INCLUDE_BACKENDS=pynccl,nccl_symm,ooverlap
#   VLLM_PLOT_EXCLUDE_BACKENDS=
#   VLLM_EVAL_REPETITIONS=1
#   VLLM_BATCH_PROMPT_MULTIPLIER=2
#   VLLM_EVAL_COOLDOWN_SECONDS=10
#   VLLM_EVAL_ORDER_COOLDOWN_SECONDS=120
#   VLLM_OOVERLAP_AG_DTYPE=bf16
#   VLLM_OOVERLAP_AG_SLOTS=4
#   VLLM_OOVERLAP_AG_CAPACITY_BYTES=67108864
#   OOVERLAP_TUNING_POLICY=/path/to/tma_collective_policy.json

usage() {
  echo "Usage: $0 {2|4} [run|plot]" >&2
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

require_nonnegative_number() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    fail "$name must be a non-negative number; got: $value"
  fi
}

[[ $# -ge 1 && $# -le 2 ]] || usage
WORLD_SIZE="$1"
MODE="${2:-run}"
case "$MODE" in
  run|plot) ;;
  *) usage ;;
esac

case "$WORLD_SIZE" in
  2)
    DEVICES="0,1"
    MODEL_REPO_ID="Qwen/Qwen2.5-7B-Instruct"
    MODEL_BASENAME="Qwen2.5-7B-Instruct"
    GPU_MEMORY_UTILIZATION="0.85"
    DEFAULT_RR_SLOTS="32"
    DEFAULT_RR_CAPACITY_BYTES="33554432"
    ;;
  4)
    DEVICES="0,1,2,3"
    MODEL_REPO_ID="Qwen/Qwen2.5-72B-Instruct"
    MODEL_BASENAME="Qwen2.5-72B-Instruct"
    GPU_MEMORY_UTILIZATION="0.85"
    DEFAULT_RR_SLOTS="165"
    DEFAULT_RR_CAPACITY_BYTES="67108864"
    ;;
  *) usage ;;
esac

# 4096 prompt tokens plus up to 512 generated tokens fit below this limit.
MAX_MODEL_LEN="${VLLM_EVAL_MAX_MODEL_LEN:-8192}"
RR_DTYPE="${VLLM_OOVERLAP_RR_DTYPE:-bf16}"
RR_SLOTS="${VLLM_OOVERLAP_RR_SLOTS:-$DEFAULT_RR_SLOTS}"
RR_CAPACITY_BYTES="${VLLM_OOVERLAP_RR_CAPACITY_BYTES:-$DEFAULT_RR_CAPACITY_BYTES}"
AG_DTYPE="${VLLM_OOVERLAP_AG_DTYPE:-$RR_DTYPE}"
AG_SLOTS="${VLLM_OOVERLAP_AG_SLOTS:-4}"
AG_CAPACITY_BYTES="${VLLM_OOVERLAP_AG_CAPACITY_BYTES:-67108864}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
NCCL_NET_PLUGIN="${OOVERLAP_NCCL_NET_PLUGIN:-none}"
NCCL_NET="${OOVERLAP_NCCL_NET:-Socket}"
VLLM_USE_NCCL_SYMM_MEM="${VLLM_USE_NCCL_SYMM_MEM:-1}"
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
NCCL_WIN_ENABLE="${NCCL_WIN_ENABLE:-1}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"

# The three configured backends are expanded into all six execution orders.
# The configured order determines deterministic order numbering only.
BACKENDS="${VLLM_EVAL_BACKENDS:-ooverlap,auto,pynccl}"
BASELINE_BACKEND="${VLLM_EVAL_BASELINE_BACKEND:-pynccl}"
REPETITIONS="${VLLM_EVAL_REPETITIONS:-1}"
DATASET_NAME="random"
RANDOM_RANGE_RATIO="0.0"
SEED="0"
FLASHINFER_SAMPLER="0"
OOVERLAP_DEBUG="0"
MAX_BATCHED_TOKENS="4096"
BATCH_PROMPT_MULTIPLIER="${VLLM_BATCH_PROMPT_MULTIPLIER:-2}"
COOLDOWN_SECONDS="${VLLM_EVAL_COOLDOWN_SECONDS:-25}"
ORDER_COOLDOWN_SECONDS="${VLLM_EVAL_ORDER_COOLDOWN_SECONDS:-90}"

for pair in \
  "VLLM_OOVERLAP_RR_SLOTS:$RR_SLOTS" \
  "VLLM_OOVERLAP_RR_CAPACITY_BYTES:$RR_CAPACITY_BYTES" \
  "VLLM_OOVERLAP_AG_SLOTS:$AG_SLOTS" \
  "VLLM_OOVERLAP_AG_CAPACITY_BYTES:$AG_CAPACITY_BYTES" \
  "VLLM_EVAL_REPETITIONS:$REPETITIONS" \
  "VLLM_BATCH_PROMPT_MULTIPLIER:$BATCH_PROMPT_MULTIPLIER"; do
  require_positive_integer "${pair%%:*}" "${pair#*:}"
done

require_nonnegative_number "VLLM_EVAL_COOLDOWN_SECONDS" "$COOLDOWN_SECONDS"
require_nonnegative_number \
  "VLLM_EVAL_ORDER_COOLDOWN_SECONDS" "$ORDER_COOLDOWN_SECONDS"

IFS=',' read -r -a BACKEND_LIST <<< "$BACKENDS"
if [[ "${#BACKEND_LIST[@]}" -ne 3 ]]; then
  fail "VLLM_EVAL_BACKENDS must contain exactly three comma-separated backends; got: $BACKENDS"
fi

for backend_index in "${!BACKEND_LIST[@]}"; do
  backend_name="${BACKEND_LIST[$backend_index]}"
  backend_name="${backend_name//[[:space:]]/}"
  if [[ -z "$backend_name" ]]; then
    fail "VLLM_EVAL_BACKENDS contains an empty backend: $BACKENDS"
  fi
  if ! [[ "$backend_name" =~ ^[A-Za-z0-9_-]+$ ]]; then
    fail "backend names may contain only letters, digits, '_' and '-'; got: $backend_name"
  fi
  BACKEND_LIST[$backend_index]="$backend_name"
done

if [[ "${BACKEND_LIST[0]}" == "${BACKEND_LIST[1]}" ||
      "${BACKEND_LIST[0]}" == "${BACKEND_LIST[2]}" ||
      "${BACKEND_LIST[1]}" == "${BACKEND_LIST[2]}" ]]; then
  fail "VLLM_EVAL_BACKENDS must contain three distinct backends; got: $BACKENDS"
fi

BACKEND_ORDERS=(
  "${BACKEND_LIST[0]},${BACKEND_LIST[1]},${BACKEND_LIST[2]}"
  #"${BACKEND_LIST[0]},${BACKEND_LIST[2]},${BACKEND_LIST[1]}"
  #"${BACKEND_LIST[1]},${BACKEND_LIST[0]},${BACKEND_LIST[2]}"
  #"${BACKEND_LIST[1]},${BACKEND_LIST[2]},${BACKEND_LIST[0]}"
  #"${BACKEND_LIST[2]},${BACKEND_LIST[0]},${BACKEND_LIST[1]}"
  #"${BACKEND_LIST[2]},${BACKEND_LIST[1]},${BACKEND_LIST[0]}"
)

MAX_SUPPORTED_RR_SLOTS="257"
if (( RR_SLOTS > MAX_SUPPORTED_RR_SLOTS )); then
  fail "VLLM_OOVERLAP_RR_SLOTS must be <= $MAX_SUPPORTED_RR_SLOTS; got: $RR_SLOTS"
fi

case "$RR_DTYPE" in
  bf16|bfloat16|fp16|float16|fp32|float32) ;;
  *) fail "VLLM_OOVERLAP_RR_DTYPE must be bf16, fp16, or fp32; got: $RR_DTYPE" ;;
esac

case "$AG_DTYPE" in
  bf16|bfloat16|fp16|float16|fp32|float32) ;;
  *) fail "VLLM_OOVERLAP_AG_DTYPE must be bf16, fp16, or fp32; got: $AG_DTYPE" ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRIVER="$REPO_ROOT/test/benchmark_vllm_paperlike.py"
PLOTTER="$REPO_ROOT/test/plot_vllm_sweeps.py"
ORDER_AGGREGATOR="$REPO_ROOT/test/aggregate_vllm_backend_orders.py"
OUT_ROOT=""
WORKLOAD=""
TUNING_POLICY="${OOVERLAP_TUNING_POLICY:-$REPO_ROOT/results/policies/tp4_policy.json}"

if [[ "$MODE" != "plot" ]]; then
  [[ -f "$TUNING_POLICY" ]] || fail "ooverlap tuning policy not found: $TUNING_POLICY"
fi

if [[ -f "$TUNING_POLICY" ]]; then
  TUNING_POLICY="$(cd -- "$(dirname -- "$TUNING_POLICY")" && pwd)/$(basename -- "$TUNING_POLICY")"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"
command -v bash >/dev/null 2>&1 || fail "bash was not found"
command -v tee >/dev/null 2>&1 || fail "tee was not found"
[[ -f "$DRIVER" ]] || fail "vLLM benchmark driver not found: $DRIVER"
[[ -f "$PLOTTER" ]] || fail "vLLM sweep plotter not found: $PLOTTER"
[[ -f "$ORDER_AGGREGATOR" ]] || fail "vLLM backend-order aggregator not found: $ORDER_AGGREGATOR"

resolve_runtime_env_file() {
  local candidate=""
  if [[ -n "${OOVERLAP_VLLM_ENV_FILE:-}" ]]; then
    candidate="$OOVERLAP_VLLM_ENV_FILE"
    [[ -f "$candidate" ]] || fail "OOVERLAP_VLLM_ENV_FILE does not exist: $candidate"
    printf '%s\n' "$candidate"
    return
  fi
  if [[ -n "${OOTMP:-}" && -f "$OOTMP/ooverlap_vllm_env.sh" ]]; then
    printf '%s\n' "$OOTMP/ooverlap_vllm_env.sh"
    return
  fi
  for candidate in /local/tmp.*/"${USER:-}"/ooverlap_vllm_env.sh; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  fail "could not find ooverlap_vllm_env.sh; run setup_env.sh or set OOVERLAP_VLLM_ENV_FILE"
}

RUNTIME_ENV_FILE="$(resolve_runtime_env_file)"
RUNTIME_ENV_DIR="$(cd -- "$(dirname -- "$RUNTIME_ENV_FILE")" && pwd)"
DEFAULT_MODEL_DIR="$RUNTIME_ENV_DIR/models/$MODEL_BASENAME"
MODEL_DIR="${VLLM_MODEL_DIR:-$DEFAULT_MODEL_DIR}"
[[ -d "$MODEL_DIR" ]] || fail "model directory does not exist: $MODEL_DIR"
[[ -f "$MODEL_DIR/config.json" ]] || fail "model directory is missing config.json: $MODEL_DIR"
MODEL="$(cd -- "$MODEL_DIR" && pwd)"
cd "$REPO_ROOT"

# Validate all three configured backends before the benchmark run.
preflight() {
  local check_code
  check_code='import os, sys, torch
import vllm.envs as envs
from vllm.distributed.device_communicators.pynccl_allocator import get_nccl_mem_pool
expected = int(sys.argv[1])
if not torch.cuda.is_available(): raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != expected: raise SystemExit(f"expected {expected} GPUs, found {torch.cuda.device_count()}")
extension = os.environ.get("VLLM_OOVERLAP_TORCH_EXT", "")
if not extension or not os.path.isfile(extension): raise SystemExit(f"invalid VLLM_OOVERLAP_TORCH_EXT: {extension!r}")
for name in ("VLLM_USE_NCCL_SYMM_MEM", "NCCL_CUMEM_ENABLE", "NCCL_WIN_ENABLE"):
    if os.environ.get(name) != "1": raise SystemExit(f"{name} must be 1")
if not envs.VLLM_USE_NCCL_SYMM_MEM: raise SystemExit("vLLM rejected symmetric memory")
if tuple(int(v) for v in torch.cuda.nccl.version()) < (2, 27, 3): raise SystemExit("NCCL >= 2.27.3 is required")
torch.cuda.set_device(0)
if get_nccl_mem_pool() is None: raise SystemExit("NCCL symmetric allocator unavailable")
print("[evalution] preflight ready")'
  bash -lc '
    set -e
    source "$1" >/dev/null
    export CUDA_VISIBLE_DEVICES="$2"
    export VLLM_USE_NCCL_SYMM_MEM="$5"
    export NCCL_CUMEM_ENABLE="$6"
    export NCCL_WIN_ENABLE="$7"
    export NCCL_NVLS_ENABLE="$8"
    export NCCL_MNNVL_ENABLE="$9"
    exec python -c "$3" "$4"
  ' "ooverlap-vllm-preflight" "$RUNTIME_ENV_FILE" "$DEVICES" "$check_code" "$WORLD_SIZE" \
    "$VLLM_USE_NCCL_SYMM_MEM" "$NCCL_CUMEM_ENABLE" "$NCCL_WIN_ENABLE" \
    "$NCCL_NVLS_ENABLE" "$NCCL_MNNVL_ENABLE"
}

common_driver_args=()
common_driver_args+=(--runtime-env-file "$RUNTIME_ENV_FILE")
common_driver_args+=(--working-dir "$REPO_ROOT")
common_driver_args+=(--models "$MODEL")
common_driver_args+=(--max-model-len "$MAX_MODEL_LEN")
common_driver_args+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
common_driver_args+=(--baseline-backend "$BASELINE_BACKEND")
common_driver_args+=(--device-groups "$DEVICES")
common_driver_args+=(--max-batched-tokens "$MAX_BATCHED_TOKENS")
common_driver_args+=(--repetitions "$REPETITIONS")
common_driver_args+=(--cooldown-seconds "$COOLDOWN_SECONDS")
common_driver_args+=(--dataset-name "$DATASET_NAME")
common_driver_args+=(--random-range-ratio "$RANDOM_RANGE_RATIO")
common_driver_args+=(--seed "$SEED")
common_driver_args+=(--enforce-eager --disable-detokenize)
common_driver_args+=(--flashinfer-sampler "$FLASHINFER_SAMPLER")
common_driver_args+=(--ooverlap-debug "$OOVERLAP_DEBUG")
common_driver_args+=(--rr-dtype "$RR_DTYPE")
common_driver_args+=(--rr-slots "$RR_SLOTS")
common_driver_args+=(--rr-capacity-bytes "$RR_CAPACITY_BYTES")
common_driver_args+=(--env "VLLM_OOVERLAP_AG_DTYPE=$AG_DTYPE")
common_driver_args+=(--env "VLLM_OOVERLAP_AG_SLOTS=$AG_SLOTS")
common_driver_args+=(--env "VLLM_OOVERLAP_AG_CAPACITY_BYTES=$AG_CAPACITY_BYTES")
common_driver_args+=(--env "OOVERLAP_TUNING_POLICY=$TUNING_POLICY")
common_driver_args+=(--env "HF_HUB_OFFLINE=$HF_HUB_OFFLINE")
common_driver_args+=(--env "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE")
common_driver_args+=(--env "NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN")
common_driver_args+=(--env "NCCL_NET=$NCCL_NET")
#common_driver_args+=(--env "VLLM_USE_NCCL_SYMM_MEM=$VLLM_USE_NCCL_SYMM_MEM")
common_driver_args+=(--fail-fast)

backend_order_slug() {
  local backend_order="$1"
  printf '%s\n' "${backend_order//,/__}"
}

backend_order_dir_name() {
  local order_index="$1"
  local backend_order="$2"
  printf 'order-%02d__%s\n' \
    "$order_index" "$(backend_order_slug "$backend_order")"
}

write_backend_order_manifest() {
  local manifest="$OUT_ROOT/backend_orders.tsv"
  local array_index
  local order_index
  local backend_order
  local directory_name

  mkdir -p "$OUT_ROOT/backend_orders"
  printf 'order_index\tbackend_order\trelative_directory\n' > "$manifest"

  for array_index in "${!BACKEND_ORDERS[@]}"; do
    order_index=$((array_index + 1))
    backend_order="${BACKEND_ORDERS[$array_index]}"
    directory_name="$(backend_order_dir_name "$order_index" "$backend_order")"
    printf '%d\t%s\tbackend_orders/%s\n' \
      "$order_index" "$backend_order" "$directory_name" >> "$manifest"
  done
}

run_batch() {
  local backend_order="$1"
  local order_root="$2"
  local out_dir="$order_root/batch_scaling"

  mkdir -p "$out_dir"
  printf '%s\n' "$backend_order" > "$order_root/backend_order.txt"

  "$PYTHON_BIN" "$DRIVER" "${common_driver_args[@]}" \
    --backends "$backend_order" \
    --out-dir "$out_dir" --workloads "" \
    --workload "$WORKLOAD" \
    --batch-sizes "1,2,4,8,16,32,64" \
    --prompt-multiplier "$BATCH_PROMPT_MULTIPLIER" \
    2>&1 | tee "$out_dir/pipeline.log"
}

# Both paper workloads are dispatched below by run_workload().

# OOVERLAP_VLLM_PLOT_REBUILDS_SUMMARIES_V1
run_summary() {
  local order_root="$1"
  local out_dir="$order_root/batch_scaling"

  "$PYTHON_BIN" "$DRIVER" \
    --out-dir "$out_dir" \
    --summarize-only \
    --baseline-backend "$BASELINE_BACKEND"
}

run_plot() {
  local order_root="$1"
  local args=(--root "$order_root" --out-dir "$order_root/plots")

  if [[ -n "${VLLM_PLOT_INCLUDE_BACKENDS:-}" ]]; then
    args+=(--include-backends "$VLLM_PLOT_INCLUDE_BACKENDS")
  fi
  if [[ -n "${VLLM_PLOT_EXCLUDE_BACKENDS:-}" ]]; then
    args+=(--exclude-backends "$VLLM_PLOT_EXCLUDE_BACKENDS")
  fi
  "$PYTHON_BIN" "$PLOTTER" "${args[@]}"
}

run_backend_order_aggregate() {
  "$PYTHON_BIN" "$ORDER_AGGREGATOR" \
    --root "$OUT_ROOT/backend_orders" \
    --baseline-backend "$BASELINE_BACKEND" \
    --target-backend ooverlap \
    --comparison-backends "$BACKENDS"
}

run_backend_orders() {
  local order_count="${#BACKEND_ORDERS[@]}"
  local array_index
  local order_index
  local backend_order
  local directory_name
  local order_root

  for array_index in "${!BACKEND_ORDERS[@]}"; do
    order_index=$((array_index + 1))
    backend_order="${BACKEND_ORDERS[$array_index]}"
    directory_name="$(backend_order_dir_name "$order_index" "$backend_order")"
    order_root="$OUT_ROOT/backend_orders/$directory_name"

    echo "[evalution] backend_order_index=$order_index/$order_count"
    echo "[evalution] backend_order=$backend_order"
    echo "[evalution] backend_order_root=$order_root"

    if [[ "$MODE" == "run" ]]; then
      run_batch "$backend_order" "$order_root"
      # Keep a usable combined report after every completed order. This also
      # makes partial multi-hour evaluations immediately inspectable.
      run_backend_order_aggregate
      run_plot "$order_root"

      if (( order_index < order_count )); then
        echo "[evalution] cooling down ${ORDER_COOLDOWN_SECONDS}s before the next backend order"
        sleep "$ORDER_COOLDOWN_SECONDS"
      fi
    else
      if [[ ! -d "$order_root/batch_scaling" ]]; then
        fail "backend-order result directory does not exist: $order_root/batch_scaling"
      fi
      run_summary "$order_root"
      run_plot "$order_root"
    fi
  done

  if [[ "$MODE" == "plot" ]]; then
    run_backend_order_aggregate
  fi
}

run_workload() {
  local result_name="$1"
  WORKLOAD="$2"
  OUT_ROOT="$REPO_ROOT/results/evalution/vllm/${result_name}/tp${WORLD_SIZE}"
  mkdir -p "$OUT_ROOT"

  echo "[evalution] workload=$WORKLOAD"
  echo "[evalution] root=$OUT_ROOT"
  echo "[evalution] backend_orders=${#BACKEND_ORDERS[@]}"
  echo "[evalution] inter_order_cooldown_seconds=$ORDER_COOLDOWN_SECONDS"

  write_backend_order_manifest
  run_backend_orders
}

case "$MODE" in
  run)
    preflight
    #run_workload "qwen_1024_128" "realistic-conversation:1024:128"
    run_workload "qwen_512_1024" "long-decode:512:1024"
    ;;
  plot)
    #run_workload "qwen_1024_128" "realistic-conversation:1024:128"
    run_workload "qwen_512_1024" "long-decode:512:1024"
    ;;
esac

echo "[evalution] complete"
