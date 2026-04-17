#!/usr/bin/env bash
set -euo pipefail

# Example:
# HF_MODEL_ID=meta-llama/Llama-2-7b-hf DATASET_PREFIX=Baby_Products REC_MODEL_PATH=./rec_model/baby_sasrec.pt \
# CKPT_PATH=./checkpoints/Baby_Products/last.ckpt bash test_new_data_rank.sh

HF_MODEL_ID="${HF_MODEL_ID:-meta-llama/Llama-2-7b-hf}"
DATA_DIR="${DATA_DIR:-new_data}"
DATASET_PREFIX="${DATASET_PREFIX:-Baby_Products}"
REC_MODEL_PATH="${REC_MODEL_PATH:-./rec_model/baby_products.pt}"
PROMPT_PATH="${PROMPT_PATH:-./prompt/game.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${DATASET_PREFIX}_test_rank}"
CKPT_PATH="${CKPT_PATH:-}"

python main.py \
  --mode test \
  --dataset amazon_new_data \
  --data_dir "${DATA_DIR}" \
  --dataset_prefix "${DATASET_PREFIX}" \
  --seq_max_len 10 \
  --cans_num 1001 \
  --batch_size 4 \
  --llm_path "${HF_MODEL_ID}" \
  --rec_model_path "${REC_MODEL_PATH}" \
  --prompt_path "${PROMPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --rank_eval \
  --verbose_step_print \
  ${CKPT_PATH:+--ckpt_path "${CKPT_PATH}"}
