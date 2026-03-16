#!/bin/bash
set -e

# -----------------------
# Environment
# -----------------------
# Assumes you have a conda env named "unidad" already created.
# (Create environment using FFHQ_src/scripts/create_env.sh)
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate unidad
else
  echo "[WARN] conda not found in PATH; please activate your env manually."
fi

# Dhariwal guided-diffusion in third_party directory
export PYTHONPATH="$PWD/third_party/dhariwal:$PYTHONPATH"

# -----------------------
# Project / dataset paths
# -----------------------
export PROJECT_PATH="FFHQ_src"
export DATASET_NAME="${DATASET_NAME:-babies}"
export DATASET_SIZE="${DATASET_SIZE:-10}"  # 10 / 5 / 1
export CHECKPOINT_INIT="${CHECKPOINT_INIT:-"$PROJECT_PATH/checkpoints/ffhq.pt"}"
export REAL_IMAGE_PATH="$PROJECT_PATH/datasets/targets/${DATASET_SIZE}_${DATASET_NAME}_lmdb"
export FID_NPZ_ROOT="$PROJECT_PATH/datasets/fid_npz"
export FEWSHOT_DATASET="$PROJECT_PATH/datasets/targets/${DATASET_SIZE}_${DATASET_NAME}/0"

export WANDB_ENTITY="ENTER YOUR WANDB INFO HERE"
export WANDB_PROJECT="ENTER YOUR WANDB INFO HERE"
export WANDB_API_KEY="ENTER YOUR WANDB INFO HERE"

# -----------------------
# Hardware / distributed
# -----------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRAIN_GPUS="${TRAIN_GPUS:-0}"
export TEST_GPUS="${TEST_GPUS:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# -----------------------
# Core training config
# -----------------------
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE=10
export NUM_DENOISING_STEP="${NUM_DENOISING_STEP:-3}"
export TRAIN_ITERS="${TRAIN_ITERS:-40000}"
export SEED=10
export RESOLUTION=256

export GEN_LR="${GEN_LR:-2e-6}"  # e.g. 2e-6 / 5e-8
export DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-1}"
export DMD_SOURCE_WEIGHT="${DMD_SOURCE_WEIGHT:-1.0}"
export DMD_TARGET_WEIGHT="${DMD_TARGET_WEIGHT:-1.0}"
export DFAKE_GEN_UPDATE_RATIO=5

export CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-5e-2}"
export GEN_CLS_LOSS_WEIGHT="${GEN_CLS_LOSS_WEIGHT:-15e-3}"
export DIFFUSION_GAN_MAX_TIMESTEP=1000

export DENOISING_SIGMA_END=0.5
export CONDITIONING_SIGMA=80.0

export LOG_ITERS=500
export WANDB_ITERS=500
export MAX_CHECKPOINT=500

export TOTAL_EVAL_SAMPLES=5000
export LPIPS_CLUSTER_SIZE=100
export NO_LPIPS=${NO_LPIPS-}  # --no_lpips

# -----------------------
# Label / conditioning (unconditional)
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
export TARGET_TEACHER_CHECKPOINT_PATH="${TARGET_TEACHER_CHECKPOINT_PATH-}" # --target_teacher_ckpt_path ...

export GAN_CLASSIFIER="${GAN_CLASSIFIER-"--gan_classifier"}" # enable GAN by default
export GAN_MULTIHEAD="${GAN_MULTIHEAD-"--gan_multihead"}"   # multi-head by default
export GAN_HEAD_TYPE="global"
export GAN_HEAD_LAYERS="${GAN_HEAD_LAYERS:-all}"
export GAN_ADV_LOSS="${GAN_ADV_LOSS:-bce}"

export TRAIN_FAKE_ON_REAL="${TRAIN_FAKE_ON_REAL-}" # --train_fake_on_real
export REVERSE_DMD="${REVERSE_DMD-}"
export TT_MATCH_GUIDANCE="${TT_MATCH_GUIDANCE-}"   # --tt_match_guidance

# -----------------------
# Misc / logging
# -----------------------
export USE_BF16="--use_bf16"  # --use_bf16
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error

export DEN_FLAG="--denoising"
export BEST_FLAG=""
export CHECKPOINT_PATH="${CHECKPOINT_PATH-}"

export MAKE_DDIM_GRID=${MAKE_DDIM_GRID:-} # --make_ddim_grid
export DDIM_GRID_ONLY=${DDIM_GRID_ONLY-} # --ddim_grid_only
export EVAL_BEST_ONCE=${EVAL_BEST_ONCE-} # --eval_best_once

# -----------------------
# Experiment naming
# -----------------------
export EXTRA_TAG="${EXTRA_TAG-}"
export EXPERIMENT_NAME="${DATASET_NAME}_lr${GEN_LR}_bs${BATCH_SIZE}_dn${NUM_DENOISING_STEP}_${TRAIN_FAKE_ON_REAL}_DMD${DMD_LOSS_WEIGHT}_GClsw${GEN_CLS_LOSS_WEIGHT}_${EXTRA_TAG}"
export OUTPUT_PATH="$PROJECT_PATH/checkpoint_path/$EXPERIMENT_NAME"
export WANDB_NAME="$EXPERIMENT_NAME"

# -----------------------
# Print all important configs
# -----------------------
echo "========== RUN CONFIG =========="
echo "PROJECT_PATH           ${PROJECT_PATH}"
echo "DATASET_NAME           ${DATASET_NAME}"
echo "DATASET_SIZE           ${DATASET_SIZE}"
echo "REAL_IMAGE_PATH        ${REAL_IMAGE_PATH}"
echo "FEWSHOT_DATASET        ${FEWSHOT_DATASET}"
echo "FID_NPZ_ROOT           ${FID_NPZ_ROOT}"
echo
echo "TRAIN_ITERS            ${TRAIN_ITERS}"
echo "BATCH_SIZE             ${BATCH_SIZE}"
echo "GRAD_ACCUM_STEPS       ${GRAD_ACCUM_STEPS}"
echo "NUM_DENOISING_STEP     ${NUM_DENOISING_STEP}"
echo "DFAKE_GEN_UPDATE_RATIO ${DFAKE_GEN_UPDATE_RATIO}"
echo "RESOLUTION             ${RESOLUTION}"
echo "SEED                   ${SEED}"
echo
echo "GEN_LR                 ${GEN_LR}"
echo "DMD_LOSS_WEIGHT        ${DMD_LOSS_WEIGHT}"
echo "DMD_SOURCE_WEIGHT      ${DMD_SOURCE_WEIGHT}"
echo "DMD_TARGET_WEIGHT      ${DMD_TARGET_WEIGHT}"
echo "CLS_LOSS_WEIGHT        ${CLS_LOSS_WEIGHT}"
echo "GEN_CLS_LOSS_WEIGHT    ${GEN_CLS_LOSS_WEIGHT}"
echo "DIFF_GAN_MAX_TIMESTEP  ${DIFFUSION_GAN_MAX_TIMESTEP}"
echo
echo "USE_SOURCE_TEACHER     ${USE_SOURCE_TEACHER}"
echo "USE_TARGET_TEACHER     ${USE_TARGET_TEACHER}"
echo "TRAIN_TARGET_TEACHER   ${TRAIN_TARGET_TEACHER}"
echo "TARGET_TCHR_CKPT_PATH  ${TARGET_TEACHER_CHECKPOINT_PATH}"
echo
echo "GAN_CLASSIFIER         ${GAN_CLASSIFIER}"
echo "GAN_MULTIHEAD          ${GAN_MULTIHEAD}"
echo "GAN_HEAD_TYPE          ${GAN_HEAD_TYPE}"
echo "GAN_HEAD_LAYERS        ${GAN_HEAD_LAYERS}"
echo "GAN_ADV_LOSS           ${GAN_ADV_LOSS}"
echo "TRAIN_FAKE_ON_REAL     ${TRAIN_FAKE_ON_REAL}"
echo "REVERSE_DMD            ${REVERSE_DMD}"
echo "TT_MATCH_GUIDANCE      ${TT_MATCH_GUIDANCE}"
echo
echo "LABEL_DIM              ${LABEL_DIM}"
echo "LABEL_DROPOUT_P        ${LABEL_DROPOUT_P}"
echo "HAS_NULL               ${HAS_NULL}"
echo
echo "TOTAL_EVAL_SAMPLES     ${TOTAL_EVAL_SAMPLES}"
echo "CONDITIONING_SIGMA     ${CONDITIONING_SIGMA}"
echo "LPIPS_CLUSTER_SIZE     ${LPIPS_CLUSTER_SIZE}"
echo "NO_LPIPS               ${NO_LPIPS}"
echo
echo "CUDA_VISIBLE_DEVICES   ${CUDA_VISIBLE_DEVICES}"
echo "TRAIN_GPUS             ${TRAIN_GPUS}"
echo "TEST_GPUS              ${TEST_GPUS}"
echo "NPROC_PER_NODE         ${NPROC_PER_NODE}"
echo "NNODES                 ${NNODES}"
echo
echo "WANDB_ENTITY           ${WANDB_ENTITY}"
echo "WANDB_PROJECT          ${WANDB_PROJECT}"
echo "WANDB_NAME             ${WANDB_NAME}"
echo
echo "OUTPUT_PATH            ${OUTPUT_PATH}"
echo "EXPERIMENT_NAME        ${EXPERIMENT_NAME}"
echo "CHECKPOINT_INIT        ${CHECKPOINT_INIT}"
echo "CHECKPOINT_PATH        ${CHECKPOINT_PATH}"
echo "==============================="
echo "[RUN] $EXPERIMENT_NAME"
echo

# -----------------------
# Launch training
# -----------------------
GEN_LR="$GEN_LR" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_DENOISING_STEP="$NUM_DENOISING_STEP" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
OUTPUT_PATH="$OUTPUT_PATH" \
WANDB_NAME="$WANDB_NAME" \
CHECKPOINT_INIT="$CHECKPOINT_INIT" \
REAL_IMAGE_PATH="$REAL_IMAGE_PATH" \
WANDB_ENTITY="$WANDB_ENTITY" \
WANDB_PROJECT="$WANDB_PROJECT" \
TRAIN_ITERS="$TRAIN_ITERS" \
SEED="$SEED" \
RESOLUTION="$RESOLUTION" \
LABEL_DIM="$LABEL_DIM" \
DATASET_NAME="$DATASET_NAME" \
DFAKE_GEN_UPDATE_RATIO="$DFAKE_GEN_UPDATE_RATIO" \
CLS_LOSS_WEIGHT="$CLS_LOSS_WEIGHT" \
GEN_CLS_LOSS_WEIGHT="$GEN_CLS_LOSS_WEIGHT" \
DMD_LOSS_WEIGHT="$DMD_LOSS_WEIGHT" \
DIFFUSION_GAN_MAX_TIMESTEP="$DIFFUSION_GAN_MAX_TIMESTEP" \
LOG_ITERS="$LOG_ITERS" \
WANDB_ITERS="$WANDB_ITERS" \
MAX_CHECKPOINT="$MAX_CHECKPOINT" \
FID_NPZ_ROOT="$FID_NPZ_ROOT" \
CATEGORY="$DATASET_NAME" \
FEWSHOT_DATASET="$FEWSHOT_DATASET" \
EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
TOTAL_EVAL_SAMPLES="$TOTAL_EVAL_SAMPLES" \
CONDITIONING_SIGMA="$CONDITIONING_SIGMA" \
LPIPS_CLUSTER_SIZE="$LPIPS_CLUSTER_SIZE" \
NO_LPIPS="$NO_LPIPS" \
LABEL_DROPOUT_P="$LABEL_DROPOUT_P" \
HAS_NULL="$HAS_NULL" \
bash "$PROJECT_PATH/experiments/run_train_test_stream.sh"
