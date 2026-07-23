#!/usr/bin/env bash
set -euo pipefail

# Fixed paper-evaluation wrapper for vLLM throughput.
#
# Usage:
#   ./evalution/run_vllm_paperlike.sh 2
#   ./evalution/run_vllm_paperlike.sh 4
#
# Default platform/model matrix:
#   TP=2: 2x H100, Qwen2.5-7B-Instruct
#   TP=4: 4x GH200, Qwen2.5-72B-Instruct
#
# Optional overrides:
#   VLLM_MODEL_DIR=/scratch/local/models/Qwen2.5-72B-Instruct \
#   VLLM_OOVERLAP_RR_SLOTS=200 \
#   VLLM_OOVERLAP_RR_CAPACITY_BYTES=67108864 \
#   OOVERLAP_MAX_CTAS=12 \
#   OOVERLAP_MAX_CTAS_PER_REDUCE_TASK=4 \
#     ./evalution/run_vllm_paperlike.sh 4
#
# The model, round-robin, CTA, NCCL, and offline variables are forwarded to
# every fresh vLLM child after the generated runtime environment is sourced.

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
    MODEL_REPO_ID="Qwen/Qwen2.5-7B-Instruct"
    MODEL_BASENAME="Qwen2.5-7B-Instruct"
    MAX_MODEL_LEN="2048"
    GPU_MEMORY_UTILIZATION="0.85"

    DEFAULT_OOVERLAP_MAX_CTAS="8"
    DEFAULT_OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="8"
    DEFAULT_RR_SLOTS="32"
    DEFAULT_RR_CAPACITY_BYTES="33554432"  # 32 MiB
    ;;
  4)
    DEVICES="0,1,2,3"
    MODEL_REPO_ID="Qwen/Qwen2.5-72B-Instruct"
    MODEL_BASENAME="Qwen2.5-72B-Instruct"
    MAX_MODEL_LEN="2048"
    GPU_MEMORY_UTILIZATION="0.85"

    DEFAULT_OOVERLAP_MAX_CTAS="12"
    DEFAULT_OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="4"
    DEFAULT_RR_SLOTS="165"
    DEFAULT_RR_CAPACITY_BYTES="67108864"  # 64 MiB
    ;;
  *)
    echo "error: world size must be exactly 2 or 4; got: $WORLD_SIZE" >&2
    usage
    ;;
esac

# OOVERLAP_VLLM_LOCAL_MODEL_AND_RR_OVERRIDES_V1
OOVERLAP_MAX_CTAS="${OOVERLAP_MAX_CTAS:-$DEFAULT_OOVERLAP_MAX_CTAS}"
OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="${OOVERLAP_MAX_CTAS_PER_REDUCE_TASK:-$DEFAULT_OOVERLAP_MAX_CTAS_PER_REDUCE_TASK}"
RR_DTYPE="${VLLM_OOVERLAP_RR_DTYPE:-bf16}"
RR_SLOTS="${VLLM_OOVERLAP_RR_SLOTS:-$DEFAULT_RR_SLOTS}"
RR_CAPACITY_BYTES="${VLLM_OOVERLAP_RR_CAPACITY_BYTES:-$DEFAULT_RR_CAPACITY_BYTES}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# OOVERLAP_VLLM_NCCL_ENV_FORWARDING_V1
# These are single-node paper experiments. Do not inherit the cluster-provided
# AWS OFI settings by accident. Dedicated OOVERLAP_* overrides are available
# when another NCCL network configuration is intentionally required.
NCCL_NET_PLUGIN="${OOVERLAP_NCCL_NET_PLUGIN:-none}"
NCCL_NET="${OOVERLAP_NCCL_NET:-Socket}"

# OOVERLAP_VLLM_NCCL_SYMM_MEM_V1
# Enable the vLLM NCCL symmetric-memory allocator and the NCCL features it
# depends on. The benchmark backend name "nccl_symm" is converted by the
# Python controller to VLLM_FORCE_ALLREDUCE_BACKEND=nccl_symm for that child.
# Explicit --env forwarding below reapplies these values after the generated
# runtime environment has been sourced.
VLLM_USE_NCCL_SYMM_MEM="${VLLM_USE_NCCL_SYMM_MEM:-1}"
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
NCCL_WIN_ENABLE="${NCCL_WIN_ENABLE:-1}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"

require_positive_integer "OOVERLAP_MAX_CTAS" "$OOVERLAP_MAX_CTAS"
require_positive_integer \
  "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK" \
  "$OOVERLAP_MAX_CTAS_PER_REDUCE_TASK"
require_positive_integer "VLLM_OOVERLAP_RR_SLOTS" "$RR_SLOTS"
require_positive_integer \
  "VLLM_OOVERLAP_RR_CAPACITY_BYTES" \
  "$RR_CAPACITY_BYTES"

MAX_SUPPORTED_RR_SLOTS="257"
if (( RR_SLOTS > MAX_SUPPORTED_RR_SLOTS )); then
  fail "VLLM_OOVERLAP_RR_SLOTS must be <= $MAX_SUPPORTED_RR_SLOTS; got: $RR_SLOTS"
fi

case "$RR_DTYPE" in
  bf16|bfloat16|fp16|float16|fp32|float32)
    ;;
  *)
    fail "VLLM_OOVERLAP_RR_DTYPE must be bf16, fp16, or fp32; got: $RR_DTYPE"
    ;;
esac

# -----------------------------------------------------------------------------
# Fixed paper matrix. Edit these constants in the repository only when the
# paper methodology changes; they are intentionally not command-line options.
# -----------------------------------------------------------------------------
BACKENDS="pynccl,nccl_symm,ooverlap"
BASELINE_BACKEND="pynccl"
#WORKLOADS="decode,mixed,long,prefill"
#BATCH_SIZES="64,128,256"
#MAX_BATCHED_TOKENS="4096,8192"
#REPETITIONS="3"
#PROMPT_MULTIPLIER="4"

WORKLOADS="decode,prefill"
BATCH_SIZES="64,128"
MAX_BATCHED_TOKENS="4096"
REPETITIONS="1"
PROMPT_MULTIPLIER="1"

DATASET_NAME="random"
RANDOM_RANGE_RATIO="0.0"
SEED="0"

FLASHINFER_SAMPLER="0"
OOVERLAP_DEBUG="0"

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
RUNTIME_ENV_DIR="$(cd -- "$(dirname -- "$RUNTIME_ENV_FILE")" && pwd)"
DEFAULT_MODEL_DIR="$RUNTIME_ENV_DIR/models/$MODEL_BASENAME"
MODEL_DIR="${VLLM_MODEL_DIR:-$DEFAULT_MODEL_DIR}"

[[ -d "$MODEL_DIR" ]] || fail \
  "model directory does not exist: $MODEL_DIR; download $MODEL_REPO_ID there or set VLLM_MODEL_DIR"
[[ -f "$MODEL_DIR/config.json" ]] || fail \
  "model directory is missing config.json: $MODEL_DIR"

MODEL="$(cd -- "$MODEL_DIR" && pwd)"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

# Validate the generated runtime environment before starting the long matrix.
# This uses the same source semantics as the Python driver and checks only
# non-secret runtime facts: vLLM availability, extension path, and GPU count.
CHECK_CODE='import os, sys, torch
import vllm.envs as envs
from vllm.distributed.device_communicators.pynccl_allocator import get_nccl_mem_pool

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

required_symm_env = {
    "VLLM_USE_NCCL_SYMM_MEM": "1",
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_WIN_ENABLE": "1",
}
for name, expected_value in required_symm_env.items():
    actual = os.environ.get(name)
    if actual != expected_value:
        raise SystemExit(f"{name} must be {expected_value!r}; got {actual!r}")
if not envs.VLLM_USE_NCCL_SYMM_MEM:
    raise SystemExit("vLLM did not accept VLLM_USE_NCCL_SYMM_MEM=1")

nccl_version = tuple(int(value) for value in torch.cuda.nccl.version())
if nccl_version < (2, 27, 3):
    raise SystemExit(f"NCCL >= 2.27.3 is required; found {nccl_version}")
torch.cuda.set_device(0)
if get_nccl_mem_pool() is None:
    raise SystemExit("vLLM NCCL symmetric-memory allocator is unavailable")

print(f"[evalution] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[evalution] nccl={nccl_version}")
for index in range(count):
    print(f"[evalution] cuda:{index} name={torch.cuda.get_device_name(index)} capability={torch.cuda.get_device_capability(index)}")
print(f"[evalution] ooverlap_extension={extension}")
print("[evalution] nccl_symmetric_allocator=ready")'

bash -lc '
  set -e
  source "$1" >/dev/null
  export CUDA_VISIBLE_DEVICES="$2"
  export VLLM_USE_NCCL_SYMM_MEM="$5"
  export NCCL_CUMEM_ENABLE="$6"
  export NCCL_WIN_ENABLE="$7"
  export NCCL_NVLS_ENABLE="$8"
  export NCCL_MNNVL_ENABLE="$9"
  command -v vllm >/dev/null
  exec python -c "$3" "$4"
' "ooverlap-vllm-preflight" \
  "$RUNTIME_ENV_FILE" \
  "$DEVICES" \
  "$CHECK_CODE" \
  "$WORLD_SIZE" \
  "$VLLM_USE_NCCL_SYMM_MEM" \
  "$NCCL_CUMEM_ENABLE" \
  "$NCCL_WIN_ENABLE" \
  "$NCCL_NVLS_ENABLE" \
  "$NCCL_MNNVL_ENABLE"

echo "[evalution] vLLM paper throughput evaluation"
echo "[evalution] world_size=$WORLD_SIZE devices=$DEVICES"
echo "[evalution] runtime_env=$RUNTIME_ENV_FILE"
echo "[evalution] model_repo_id=$MODEL_REPO_ID"
echo "[evalution] model_dir=$MODEL"
echo "[evalution] max_model_len=$MAX_MODEL_LEN"
echo "[evalution] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "[evalution] ooverlap_max_ctas=$OOVERLAP_MAX_CTAS"
echo "[evalution] ooverlap_max_ctas_per_reduce_task=$OOVERLAP_MAX_CTAS_PER_REDUCE_TASK"
echo "[evalution] rr_dtype=$RR_DTYPE"
echo "[evalution] rr_slots=$RR_SLOTS"
echo "[evalution] rr_capacity_bytes=$RR_CAPACITY_BYTES"
echo "[evalution] hf_hub_offline=$HF_HUB_OFFLINE"
echo "[evalution] transformers_offline=$TRANSFORMERS_OFFLINE"
echo "[evalution] nccl_net_plugin=$NCCL_NET_PLUGIN"
echo "[evalution] nccl_net=$NCCL_NET"
echo "[evalution] vllm_use_nccl_symm_mem=$VLLM_USE_NCCL_SYMM_MEM"
echo "[evalution] nccl_cumem_enable=$NCCL_CUMEM_ENABLE"
echo "[evalution] nccl_win_enable=$NCCL_WIN_ENABLE"
echo "[evalution] nccl_nvls_enable=$NCCL_NVLS_ENABLE"
echo "[evalution] nccl_mnnvl_enable=$NCCL_MNNVL_ENABLE"
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
  --env "OOVERLAP_MAX_CTAS=$OOVERLAP_MAX_CTAS" \
  --env "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK=$OOVERLAP_MAX_CTAS_PER_REDUCE_TASK" \
  --env "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
  --env "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE" \
  --env "NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN" \
  --env "NCCL_NET=$NCCL_NET" \
  --env "VLLM_USE_NCCL_SYMM_MEM=$VLLM_USE_NCCL_SYMM_MEM" \
  --env "NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE" \
  --env "NCCL_WIN_ENABLE=$NCCL_WIN_ENABLE" \
  --env "NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE" \
  --env "NCCL_MNNVL_ENABLE=$NCCL_MNNVL_ENABLE" \
  --fail-fast \
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
