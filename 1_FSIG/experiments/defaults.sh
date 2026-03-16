#!/usr/bin/env bash
set -euo pipefail

# repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# make guided-diffusion importable
export PYTHONPATH="$ROOT_DIR/third_party/dhariwal:${PYTHONPATH:-}"

# -----------------------
# Project / dataset paths
# -----------------------
export PROJECT_PATH="${PROJECT_PATH:-FFHQ_src}"
export DATASET_NAME="${DATASET_NAME:-babies}"
export DATASET_SIZE="${DATASET_SIZE:-10}"

export CHECKPOINT_INIT="${CHECKPOINT_INIT:-$PROJECT_PATH/checkpoints/ffhq.pt}"
export REAL_IMAGE_PATH="${REAL_IMAGE_PATH:-$PROJECT_PATH/datasets/targets/${DATASET_SIZE}_${DATASET_NAME}_lmdb}"
export FID_NPZ_ROOT="${FID_NPZ_ROOT:-$PROJECT_PATH/datasets/fid_npz}"
export FEWSHOT_DATASET="${FEWSHOT_DATASET:-$PROJECT_PATH/datasets/targets/${DATASET_SIZE}_${DATASET_NAME}/0}"
export CATEGORY="${CATEGORY:-$DATASET_NAME}"

# -----------------------
# Logging
# -----------------------
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-UniDAD}"
# It is recommended to NOT store WANDB_API_KEY in code. Use `wandb login` or env vars externally.

# -----------------------
# Hardware / distributed
# -----------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRAIN_GPUS="${TRAIN_GPUS:-0}"
export TEST_GPUS="${TEST_GPUS:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# -----------------------
# Core training config
# -----------------------
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
export NUM_DENOISING_STEP="${NUM_DENOISING_STEP:-3}"
export TRAIN_ITERS="${TRAIN_ITERS:-40000}"
export SEED="${SEED:-10}"
export RESOLUTION="${RESOLUTION:-256}"

export GEN_LR="${GEN_LR:-2e-6}"
export DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-1}"
export DMD_SOURCE_WEIGHT="${DMD_SOURCE_WEIGHT:-1.0}"
export DMD_TARGET_WEIGHT="${DMD_TARGET_WEIGHT:-1.0}"
export DFAKE_GEN_UPDATE_RATIO="${DFAKE_GEN_UPDATE_RATIO:-5}"

export CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-5e-2}"
export GEN_CLS_LOSS_WEIGHT="${GEN_CLS_LOSS_WEIGHT:-15e-3}"
export DIFFUSION_GAN_MAX_TIMESTEP="${DIFFUSION_GAN_MAX_TIMESTEP:-1000}"

export DENOISING_SIGMA_END="${DENOISING_SIGMA_END:-0.5}"
export CONDITIONING_SIGMA="${CONDITIONING_SIGMA:-80.0}"

export LOG_ITERS="${LOG_ITERS:-500}"
export WANDB_ITERS="${WANDB_ITERS:-500}"
export MAX_CHECKPOINT="${MAX_CHECKPOINT:-500}"

export TOTAL_EVAL_SAMPLES="${TOTAL_EVAL_SAMPLES:-5000}"
export LPIPS_CLUSTER_SIZE="${LPIPS_CLUSTER_SIZE:-100}"

# -----------------------
# Fixed unconditional setup
# -----------------------
export LABEL_DIM=0
export LABEL_DROPOUT_P=0.0
export HAS_NULL=""

# -----------------------
# Teacher / GAN settings
# -----------------------
export USE_SOURCE_TEACHER="${USE_SOURCE_TEACHER:-1.0}"
export USE_TARGET_TEACHER="${USE_TARGET_TEACHER:-0.0}"
export TRAIN_TARGET_TEACHER="${TRAIN_TARGET_TEACHER:-1.0}"
export TARGET_TEACHER_CHECKPOINT_PATH="${TARGET_TEACHER_CHECKPOINT_PATH:-}"

export GAN_CLASSIFIER="${GAN_CLASSIFIER:---gan_classifier}"
export GAN_MULTIHEAD="${GAN_MULTIHEAD:---gan_multihead}"
export GAN_HEAD_TYPE="${GAN_HEAD_TYPE:-global}"
export GAN_HEAD_LAYERS="${GAN_HEAD_LAYERS:-all}"
export GAN_ADV_LOSS="${GAN_ADV_LOSS:-bce}"

export TRAIN_FAKE_ON_REAL="${TRAIN_FAKE_ON_REAL:-}"
export REVERSE_DMD="${REVERSE_DMD:-}"
export TT_MATCH_GUIDANCE="${TT_MATCH_GUIDANCE:-}"

# -----------------------
# Misc
# -----------------------
export USE_BF16="${USE_BF16:---use_bf16}"
export NO_LPIPS="${NO_LPIPS:-}"
export MAKE_DDIM_GRID="${MAKE_DDIM_GRID:-}"
export DDIM_GRID_ONLY="${DDIM_GRID_ONLY:-}"
export EVAL_BEST_ONCE="${EVAL_BEST_ONCE:-}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

# -----------------------
# Naming
# -----------------------
export EXTRA_TAG="${EXTRA_TAG:-}"
_default_name="${DATASET_NAME}_lr${GEN_LR}_bs${BATCH_SIZE}_dn${NUM_DENOISING_STEP}"
if [[ -n "$EXTRA_TAG" ]]; then
  _default_name="${_default_name}_${EXTRA_TAG}"
fi

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$_default_name}"
export OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_PATH/checkpoint_path/$EXPERIMENT_NAME}"
export WANDB_NAME="${WANDB_NAME:-$EXPERIMENT_NAME}"

print_config() {
  echo "========== RUN CONFIG =========="
  echo "DATASET_NAME      $DATASET_NAME"
  echo "DATASET_SIZE      $DATASET_SIZE"
  echo "GEN_LR            $GEN_LR"
  echo "BATCH_SIZE        $BATCH_SIZE"
  echo "TRAIN_ITERS       $TRAIN_ITERS"
  echo "NUM_DENOISING     $NUM_DENOISING_STEP"
  echo "DMD_SRC_W         $DMD_SOURCE_WEIGHT"
  echo "DMD_TGT_W         $DMD_TARGET_WEIGHT"
  echo "USE_SRC_TEACHER   $USE_SOURCE_TEACHER"
  echo "USE_TGT_TEACHER   $USE_TARGET_TEACHER"
  echo "TRAIN_TGT_TEACHER $TRAIN_TARGET_TEACHER"
  echo "OUTPUT_PATH       $OUTPUT_PATH"
  echo "WANDB_PROJECT     $WANDB_PROJECT"
  echo "WANDB_NAME        $WANDB_NAME"
  echo "==============================="
}