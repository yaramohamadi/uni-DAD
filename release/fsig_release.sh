#!/usr/bin/env bash
# Release-weights run for FSIG (paper Table 1, "Uni-DAD" row, with target teacher).
# Usage: bash release/fsig_release.sh <babies|metfaces> [extra env overrides via environment]
# Paper settings (Appx 8.1): 20K iters, NFE 3, lr 2e-6, bs 1, bf16, update ratio 5,
# lambda_G = 0.01, lambda_D = 0.03, a = 0.25 (close) / 0.75 (distant).
set -Eeuo pipefail

DS="${1:?usage: fsig_release.sh <babies|metfaces>}"
case "$DS" in
  babies)   A=0.25 ;;
  metfaces) A=0.75 ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
SRC_W=$(python3 -c "print(1-$A)")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$HOME/uni-DAD-runs}"
RUN_NAME="${RUN_NAME:-release_${DS}_a${A}}"

export EXPERIMENT_NAME="$RUN_NAME"
export WANDB_NAME="$RUN_NAME"
export OUTPUT_PATH="$RUNS_ROOT/$RUN_NAME"
export CHECKPOINT_INIT="${CHECKPOINT_INIT:-$HOME/uni-DAD-ckpt/ffhq.pt}"
export DATASET_NAME="$DS"
export DATASET_SIZE=10
export CATEGORY="$DS"

# one GPU shared by train and the streaming FID evaluator
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAIN_GPUS="${TRAIN_GPUS:-0}"
export TEST_GPUS="${TEST_GPUS:-0}"
export NPROC_PER_NODE=1 NNODES=1

export SEED="${SEED:-10}"
export BATCH_SIZE=1 GRAD_ACCUM_STEPS=1
export TRAIN_ITERS="${TRAIN_ITERS:-20000}"
export NUM_DENOISING_STEP=3
export GEN_LR=2e-6
export DFAKE_GEN_UPDATE_RATIO=5
export GEN_CLS_LOSS_WEIGHT=0.01   # lambda_G (paper); repo default was 15e-3
export CLS_LOSS_WEIGHT=0.03       # lambda_D (paper); repo default was 5e-2

export DMD_SOURCE_WEIGHT="$SRC_W"
export DMD_TARGET_WEIGHT="$A"
export USE_SOURCE_TEACHER=1.0
export USE_TARGET_TEACHER=1.0
export TRAIN_TARGET_TEACHER=1.0

export GAN_ADV_LOSS=bce
export GAN_MULTIHEAD=--gan_multihead
export TOTAL_EVAL_SAMPLES=5000

# W&B offline by default: no key on the box, and the team quota is small.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$OUTPUT_PATH"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "git: ${GIT_COMMIT:-unknown}"
  echo "UNIDAD_GRAD_CHECKPOINT=${UNIDAD_GRAD_CHECKPOINT:-0}"
  env | grep -E '^(DATASET|DMD_|USE_|TRAIN_|GEN_|CLS_|NUM_|SEED|BATCH|GAN_|TOTAL_EVAL|CHECKPOINT_INIT|OUTPUT_PATH|WANDB_MODE)' | sort
} > "$OUTPUT_PATH/release_config.txt"

exec bash "$REPO/1_FSIG/experiments/train_test_stream.sh"
