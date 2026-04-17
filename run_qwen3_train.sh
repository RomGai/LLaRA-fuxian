#!/usr/bin/env bash
set -euo pipefail

# Example:
# MODEL_NAME=Qwen/Qwen3-8B DATASET_PREFIX=Baby_Products OUTPUT_DIR=./checkpoints_qwen3_baby bash run_qwen3_train.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-new_data}"
DATASET_PREFIX="${DATASET_PREFIX:-Baby_Products}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints_qwen3_${DATASET_PREFIX}}"
EPOCHS="${EPOCHS:-1}"
TRAIN_BS="${TRAIN_BS:-1}"
EVAL_BS="${EVAL_BS:-1}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-1234}"

python qwen3_train.py \
  --model_name "${MODEL_NAME}" \
  --data_dir "${DATA_DIR}" \
  --dataset_prefix "${DATASET_PREFIX}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LR}" \
  --max_length "${MAX_LENGTH}" \
  --max_samples "${MAX_SAMPLES}" \
  --seed "${SEED}"
