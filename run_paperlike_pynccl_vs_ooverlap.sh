#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-$OOTMP/vllm_eager_paperlike}"
mkdir -p "$OUT/logs"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BACKENDS="${BACKENDS:-pynccl ooverlap}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_OOVERLAP_DEBUG="${VLLM_OOVERLAP_DEBUG:-0}"

BSZS="${BSZS:-128 256 512}"
PROMPT_MULTIPLIER="${PROMPT_MULTIPLIER:-4}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"

export VLLM_OOVERLAP_RR_DTYPE="${VLLM_OOVERLAP_RR_DTYPE:-bf16}"
export VLLM_OOVERLAP_RR_SLOTS="${VLLM_OOVERLAP_RR_SLOTS:-32}"
export VLLM_OOVERLAP_RR_CAPACITY_BYTES="${VLLM_OOVERLAP_RR_CAPACITY_BYTES:-33554432}"

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
  local max_seqs="$3"
  local in_len="$4"
  local out_len="$5"

  local num_prompts=$((max_seqs * PROMPT_MULTIPLIER))
  local num_warmups="$max_seqs"

  local json="$OUT/${kind}_backend_${backend}_bsz_${max_seqs}.json"
  local log="$OUT/logs/${kind}_backend_${backend}_bsz_${max_seqs}.log"

  echo
  echo "================================================================"
  echo "kind=$kind backend=$backend"
  echo "max_seqs=$max_seqs num_prompts=$num_prompts"
  echo "input_len=$in_len output_len=$out_len"
  echo "max_batched_tokens=$MAX_BATCHED_TOKENS"
  echo "rr_slots=$VLLM_OOVERLAP_RR_SLOTS"
  echo "rr_capacity=$VLLM_OOVERLAP_RR_CAPACITY_BYTES"
  echo "json=$json"
  echo "================================================================"

  VLLM_FORCE_ALLREDUCE_BACKEND="$backend" \
  CUDA_VISIBLE_DEVICES=0,1 \
  vllm bench throughput \
    "${COMMON_ARGS[@]}" \
    --random-input-len "$in_len" \
    --random-output-len "$out_len" \
    --num-prompts "$num_prompts" \
    --num-warmups "$num_warmups" \
    --max-num-seqs "$max_seqs" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --output-json "$json" \
    2>&1 | tee "$log"
}

for bsz in $BSZS; do
  # Pair the backends case-by-case to reduce machine/time drift.

  # Decode-heavy workload.
  for backend in $BACKENDS; do
    run_one decode "$backend" "$bsz" 16 256
  done

  # Typical chat-style workload.
  for backend in $BACKENDS; do
    run_one mixed "$backend" "$bsz" 256 128
  done

  # Longer-context chat workload.
  for backend in $BACKENDS; do
    run_one long "$backend" "$bsz" 1024 128
  done

  # Prefill-focused workload.
  for backend in $BACKENDS; do
    run_one prefill "$backend" "$bsz" 1024 1
  done
done
