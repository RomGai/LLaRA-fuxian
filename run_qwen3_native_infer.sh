#!/usr/bin/env bash
set -euo pipefail

# Example:
# MODEL_NAME=Qwen/Qwen3-8B DATASET_PREFIX=Baby_Products MAX_USERS=100 bash run_qwen3_native_infer.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-new_data}"
DATASET_PREFIX="${DATASET_PREFIX:-Baby_Products}"
SEQ_MAX_LEN="${SEQ_MAX_LEN:-10}"
NUM_NEGATIVES="${NUM_NEGATIVES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_USERS="${MAX_USERS:-0}"
SEED="${SEED:-1234}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

EXTRA_ARGS=()
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_thinking)
fi

python qwen3_native_infer.py \
  --model_name "${MODEL_NAME}" \
  --data_dir "${DATA_DIR}" \
  --dataset_prefix "${DATASET_PREFIX}" \
  --seq_max_len "${SEQ_MAX_LEN}" \
  --num_negatives "${NUM_NEGATIVES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_users "${MAX_USERS}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
