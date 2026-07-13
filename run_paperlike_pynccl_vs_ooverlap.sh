#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-$OOTMP/vllm_eager_paperlike}"
mkdir -p "$OUT/logs"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BACKENDS="${BACKENDS:-pynccl ooverlap}"
BSZS="${BSZS:-128 256 512}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_OOVERLAP_DEBUG="${VLLM_OOVERLAP_DEBUG:-0}"
unset VLLM_OOVERLAP_BROKER_KEY

COMMON_ARGS=(
  --backend vllm
  --model "$MODEL"
  --dataset-name random
  --random-range-ratio 0.0
  --tensor-parallel-size 2
  --max-model-len 2048
  --gpu-memory-utilization 0.85
  --enforce-eager
  --disable-detokenize
  --seed 0
)

run_one() {
  local kind="$1"
  local backend="$2"
  local bsz="$3"
  local in_len="$4"
  local out_len="$5"
  local max_batched_tokens="$6"

  local json="$OUT/${kind}_backend_${backend}_bsz_${bsz}.json"
  local log="$OUT/logs/${kind}_backend_${backend}_bsz_${bsz}.log"

  echo
  echo "================================================================"
  echo "kind=$kind backend=$backend bsz=$bsz input_len=$in_len output_len=$out_len"
  echo "json=$json"
  echo "================================================================"

  VLLM_FORCE_ALLREDUCE_BACKEND="$backend" \
  CUDA_VISIBLE_DEVICES=0,1 \
  vllm bench throughput \
    "${COMMON_ARGS[@]}" \
    --random-input-len "$in_len" \
    --random-output-len "$out_len" \
    --num-prompts "$bsz" \
    --num-warmups "$bsz" \
    --max-num-seqs "$bsz" \
    --max-num-batched-tokens "$max_batched_tokens" \
    --output-json "$json" \
    2>&1 | tee "$log"
}

for backend in $BACKENDS; do
  for bsz in $BSZS; do
    # Decode-like: short prompt, many generated tokens.
    run_one decode "$backend" "$bsz" 16 128 1024

    # Prefill-like: long prompt, one generated token.
    run_one prefill "$backend" "$bsz" 1024 1 1024
  done
done
