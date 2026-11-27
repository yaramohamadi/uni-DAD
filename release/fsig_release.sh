#!/usr/bin/env bash
# Release-weights run for FSIG (paper Table 1, "Uni-DAD" row, with target teacher).
#
# Usage:
#   bash release/fsig_release.sh <babies|metfaces> train   # phase 1: train, keep generator snapshots
#   bash release/fsig_release.sh <babies|metfaces> eval    # phase 2: FID/LPIPS on every snapshot
#
# Why two phases: on a single 48 GB GPU, training (37.7 GB with gradient checkpointing)
# plus the streaming evaluator (14.7 GB) do not fit at the same time. So training saves a
# generator-only snapshot at every checkpoint (UNIDAD_GEN_SNAPSHOT_DIR), and the repo's own
# evaluator scores them afterwards and keeps the best FID (same selection as the paper).
#
# Paper settings (Appx 8.1): 20K iters, NFE 3, lr 2e-6, bs 1, bf16, update ratio 5,
# lambda_G = 0.01, lambda_D = 0.03, a = 0.25 (close) / 0.75 (distant), FID on 5K samples.
set -Eeuo pipefail

DS="${1:?usage: fsig_release.sh <babies|metfaces> <train|eval>}"
MODE="${2:?usage: fsig_release.sh <babies|metfaces> <train|eval>}"
case "$DS" in
  babies)   A=0.25 ;;
  metfaces) A=0.75 ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
SRC_W=$(python3 -c "print(1-$A)")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO/1_FSIG/checkpoint_path}"
RUN_NAME="${RUN_NAME:-release_${DS}_a${A}}"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
SNAP_DIR="$RUN_DIR/gen_snapshots"

export DATASET_NAME="$DS" DATASET_SIZE=10 CATEGORY="$DS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TRAIN_GPUS=0 TEST_GPUS=0
export NPROC_PER_NODE=1 NNODES=1
export NUM_DENOISING_STEP=3
export TOTAL_EVAL_SAMPLES="${TOTAL_EVAL_SAMPLES:-5000}"
# W&B offline by default (no API key needed); set WANDB_MODE=online to sync.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$RUN_DIR"

if [[ "$MODE" == train ]]; then
  export EXPERIMENT_NAME="$RUN_NAME" WANDB_NAME="$RUN_NAME" OUTPUT_PATH="$RUN_DIR"
  export CHECKPOINT_INIT="${CHECKPOINT_INIT:-$REPO/1_FSIG/checkpoints/ffhq.pt}"
  export SEED="${SEED:-10}"
  export BATCH_SIZE=1 GRAD_ACCUM_STEPS=1
  export TRAIN_ITERS="${TRAIN_ITERS:-20000}"
  export LOG_ITERS="${LOG_ITERS:-1000}"      # checkpoint + snapshot cadence
  export GEN_LR=2e-6
  export DFAKE_GEN_UPDATE_RATIO=5
  export GEN_CLS_LOSS_WEIGHT=0.01   # lambda_G (paper); repo default was 15e-3
  export CLS_LOSS_WEIGHT=0.03       # lambda_D (paper); repo default was 5e-2
  export DMD_SOURCE_WEIGHT="$SRC_W" DMD_TARGET_WEIGHT="$A"
  export USE_SOURCE_TEACHER=1.0 USE_TARGET_TEACHER=1.0 TRAIN_TARGET_TEACHER=1.0
  export GAN_ADV_LOSS=bce GAN_MULTIHEAD=--gan_multihead
  export UNIDAD_GRAD_CHECKPOINT="${UNIDAD_GRAD_CHECKPOINT:-1}"
  export UNIDAD_GEN_SNAPSHOT_DIR="$SNAP_DIR"

  {
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "git: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    env | grep -E '^(UNIDAD_|DATASET|DMD_|USE_|TRAIN_|GEN_|CLS_|NUM_|SEED|BATCH|GAN_|LOG_ITERS|CHECKPOINT_INIT|OUTPUT_PATH|WANDB_MODE)' | sort
  } > "$RUN_DIR/release_config.txt"

  exec bash "$REPO/1_FSIG/experiments/train.sh"

elif [[ "$MODE" == eval ]]; then
  export EXPERIMENT_NAME="${RUN_NAME}_eval" WANDB_NAME="${RUN_NAME}_eval" OUTPUT_PATH="$SNAP_DIR"
  n_snap=$(find "$SNAP_DIR" -maxdepth 1 -type d -name 'checkpoint_model_*' | wc -l)
  echo "[eval] $n_snap snapshots in $SNAP_DIR"
  LOG="$RUN_DIR/eval.log"
  ( setsid bash "$REPO/1_FSIG/experiments/test.sh" > "$LOG" 2>&1 ) &
  EVAL_PID=$!
  trap 'kill -TERM -"$(ps -o pgid= $EVAL_PID | tr -d " ")" 2>/dev/null || true' EXIT
  # the evaluator loops forever; stop it once every snapshot has a result line
  while kill -0 "$EVAL_PID" 2>/dev/null; do
    done_n=$(grep -c -E '^checkpoint .* FID [0-9.]+ ' "$LOG" || true)
    [[ "$done_n" -ge "$n_snap" ]] && break
    sleep 30
  done
  grep -E '^checkpoint .* FID ' "$LOG" > "$RUN_DIR/eval_results.txt" || true
  echo "[eval] results: $RUN_DIR/eval_results.txt"
  cat "$RUN_DIR/eval_results.txt"
else
  echo "unknown mode $MODE"; exit 1
fi
