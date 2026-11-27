#!/usr/bin/env bash
# v2 release run for FSIG: paper Appx 8.1 recipe + stabilisers + in-training eval.
#
# Usage: bash release/fsig_release_v2.sh <babies|metfaces>
#
# Changes vs the paper recipe (all opt-in in the code, set here):
#   EMA of generator weights   UNIDAD_EMA_DECAY (per generator update; G updates every 5 iters)
#   LR decay                   cosine from UNIDAD_LR_DECAY_START * iters down to UNIDAD_LR_MIN_RATIO
#   batch size                 BATCH_SIZE / GRAD_ACCUM_STEPS
# In-training eval: every UNIDAD_SNAPSHOT_EVERY iters the trainer saves the EMA generator
# (and the raw one every UNIDAD_RAW_SNAPSHOT_EVERY), moves optimizer state to CPU, runs the
# repo evaluator (5K samples, NFE 3, FID + Intra-LPIPS) on the same GPU, then continues.
set -Eeuo pipefail

DS="${1:?usage: fsig_release_v2.sh <babies|metfaces>}"
case "$DS" in
  babies)   A=0.25 ;;
  metfaces) A=0.75 ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
SRC_W=$(python3 -c "print(1-$A)")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO/1_FSIG/checkpoint_path}"
SNAP_ROOT="${SNAP_ROOT:-$RUNS_ROOT}"   # generator snapshots (~2.2 GB each); put on a big disk if needed
RUN_NAME="${RUN_NAME:-v2_${DS}_a${A}}"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"

export DATASET_NAME="$DS" DATASET_SIZE=10 CATEGORY="$DS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TRAIN_GPUS=0 TEST_GPUS=0
export NPROC_PER_NODE=1 NNODES=1
export EXPERIMENT_NAME="$RUN_NAME" WANDB_NAME="$RUN_NAME" OUTPUT_PATH="$RUN_DIR"
export CHECKPOINT_INIT="${CHECKPOINT_INIT:-$REPO/1_FSIG/checkpoints/ffhq.pt}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ---- paper recipe ----
export SEED="${SEED:-10}"
export TRAIN_ITERS="${TRAIN_ITERS:-20000}"
export NUM_DENOISING_STEP=3
export GEN_LR=2e-6
export DFAKE_GEN_UPDATE_RATIO=5
export GEN_CLS_LOSS_WEIGHT=0.01 CLS_LOSS_WEIGHT=0.03
export DMD_SOURCE_WEIGHT="$SRC_W" DMD_TARGET_WEIGHT="$A"
export USE_SOURCE_TEACHER=1.0 USE_TARGET_TEACHER=1.0 TRAIN_TARGET_TEACHER=1.0
export GAN_ADV_LOSS=bce GAN_MULTIHEAD=--gan_multihead
export UNIDAD_GRAD_CHECKPOINT="${UNIDAD_GRAD_CHECKPOINT:-1}"

# ---- stabilisers (v2) ----
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export UNIDAD_EMA_DECAY="${UNIDAD_EMA_DECAY:-0.998}"
export UNIDAD_LR_DECAY_START="${UNIDAD_LR_DECAY_START:-0.5}"
export UNIDAD_LR_MIN_RATIO="${UNIDAD_LR_MIN_RATIO:-0.1}"

# ---- snapshots + eval during training ----
# The trainer stops every UNIDAD_STOP_EVERY iters (full checkpoint), the evaluator then gets the
# whole GPU, and the trainer resumes (UNIDAD_FULL_RESUME). In-process pausing (INLINE_EVAL=1)
# made training ~8x slower afterwards on this WSL box (GPU memory filled to 48.3 GB after the
# models were moved back), so it is off.
export UNIDAD_STOP_EVERY="${UNIDAD_STOP_EVERY:-500}"
export LOG_ITERS="${LOG_ITERS:-500}"             # full 21 GB checkpoint (only the newest is kept), for resume
export MAX_CHECKPOINT=1
export UNIDAD_SNAPSHOT_DIR="$SNAP_ROOT/$RUN_NAME"
export UNIDAD_SNAPSHOT_EVERY="${UNIDAD_SNAPSHOT_EVERY:-500}"
export UNIDAD_RAW_SNAPSHOT_EVERY="${UNIDAD_RAW_SNAPSHOT_EVERY:-2000}"
export UNIDAD_INLINE_EVAL="${UNIDAD_INLINE_EVAL:-0}"
export UNIDAD_EVAL_OFFLOAD="${UNIDAD_EVAL_OFFLOAD:-1}"
export UNIDAD_FAST_LPIPS="${UNIDAD_FAST_LPIPS:-1}"
export TOTAL_EVAL_SAMPLES="${TOTAL_EVAL_SAMPLES:-5000}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"   # 50 overflows 48 GB and spills to system RAM (9x slower)

mkdir -p "$RUN_DIR" "$UNIDAD_SNAPSHOT_DIR"

latest_ckpt() {
  ls -d "$RUN_DIR"/checkpoint_model_* 2>/dev/null | while read -r d; do [[ -f "$d/.READY" ]] && echo "$d"; done | sort | tail -1 || true
}

eval_pending() {   # score every snapshot not yet in stats.json (ema and raw)
  for kind in ema raw; do
    local f="$UNIDAD_SNAPSHOT_DIR/$kind"
    [[ -d "$f" ]] || continue
    ls -d "$f"/checkpoint_model_* >/dev/null 2>&1 || continue
    echo "[v2] $(date -Is) eval $kind"
    OUTPUT_PATH="$f" EXPERIMENT_NAME="${RUN_NAME}_${kind}" WANDB_NAME="${RUN_NAME}_${kind}_eval" \
      UNIDAD_EVAL_RESUME=1 UNIDAD_EVAL_EXIT_WHEN_IDLE=1 \
      bash "$REPO/1_FSIG/experiments/test.sh" >> "$f/eval.log" 2>&1 || echo "[v2] eval $kind rc=$?"
    grep -a -E '^checkpoint .* FID' "$f/eval.log" | tail -1
  done
}

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "git: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  env | grep -E '^(UNIDAD_|DATASET|DMD_|USE_|TRAIN_|GEN_|CLS_|NUM_|SEED|BATCH|GRAD_ACCUM|GAN_|LOG_ITERS|CHECKPOINT_PATH|CHECKPOINT_INIT|OUTPUT_PATH|WANDB_MODE|TOTAL_EVAL|EVAL_BATCH)' | sort
} >> "$RUN_DIR/release_config.txt"   # append: one block per (re)start

# train -> stop -> eval -> resume, until the trainer writes TRAINING_DONE
eval_pending                       # catch up first (e.g. after a reboot during eval)
while [[ ! -f "$RUN_DIR/TRAINING_DONE" ]]; do
  LATEST="$(latest_ckpt)"
  if [[ -n "$LATEST" ]]; then
    export CHECKPOINT_PATH="$LATEST" UNIDAD_FULL_RESUME=1
    echo "[v2] $(date -Is) resume from $LATEST"
  else
    unset CHECKPOINT_PATH UNIDAD_FULL_RESUME
    echo "[v2] $(date -Is) fresh start"
  fi
  bash "$REPO/1_FSIG/experiments/train.sh" || { echo "[v2] train failed rc=$?"; exit 1; }
  eval_pending
done
eval_pending
echo "[v2] $(date -Is) done"
