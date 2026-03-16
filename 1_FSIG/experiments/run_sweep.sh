#!/bin/bash
# Sweep driver for Uni-DAD ablations
#
# Currently: runs a SINGLE experiment:
#   - Dataset: babies
#   - Source teacher ON  (USE_SOURCE_TEACHER=1)
#   - Target teacher OFF (USE_TARGET_TEACHER=0, TRAIN_TARGET_TEACHER=0)
#   - DMDsrc weight = 1.0, DMDtrg weight = 0.0
#   - GAN: multi-head classifier (GAN_MULTIHEAD)
#
# Other possible variants you might want to explore later (for the paper):
#   (A) GAN configurations:
#       - no GAN       : gan="none"
#       - single-head  : gan="single"
#       - multi-head   : gan="multi"
#   (B) Teacher configurations:
#       - DMDsrc only         : USE_SOURCE_TEACHER=1, USE_TARGET_TEACHER=0
#       - DMDtrg only         : USE_SOURCE_TEACHER=0, USE_TARGET_TEACHER=1
#       - DMDsrc + DMDtrg     : USE_SOURCE_TEACHER=1, USE_TARGET_TEACHER=1
#   (C) Example combined rows:
#       - DMDsrc only + no GAN
#       - DMDtrg only + multi-head GAN
#       - DMDsrc + DMDtrg + single-head GAN
#
# Right now we only call ONE concrete row below.

set -e

CHILD="FFHQ_src/experiments/run_config.sh"
LOGDIR="FFHQ_src/logs"
mkdir -p "$LOGDIR"

export WANDB_PROJECT="${WANDB_PROJECT:-UniDAD_babies_ablation}"

# --------- Base hparams ---------
export GEN_LR="2e-6"
export GEN_CLS_LOSS_WEIGHT="1e-2"
export CLS_LOSS_WEIGHT="3e-3"
export DMD_LOSS_WEIGHT="1"
export DATASET_SIZE="10"
export NUM_DENOISING_STEP="3"
export GRAD_ACCUM_STEPS=1
export BATCH_SIZE=1
export TRAIN_GPUS=0
export TEST_GPUS=1
export CUDA_VISIBLE_DEVICES="0,1"
export NPROC_PER_NODE=1
export NNODES=1
export TRAIN_FAKE_ON_REAL=""
export TT_MATCH_GUIDANCE=""

# Simple helper to format floats into tags
fmtw () { echo "$1" | sed 's/\./p/g'; }

submit_run () {
  local tag="$1"
  echo "[LOCAL] launching run with tag: $tag"
  bash "$CHILD"
}

run_row () {
  # args: ds use_src use_tgt sw tw gan rowname
  local ds="$1" use_src="$2" use_tgt="$3" sw="$4" tw="$5" gan="$6" row="$7"

  # Map GAN mode -> flags
  case "$gan" in
    none)   export GAN_CLASSIFIER="";                 export GAN_MULTIHEAD="";;
    single) export GAN_CLASSIFIER="--gan_classifier"; export GAN_MULTIHEAD="";;
    multi)  export GAN_CLASSIFIER="--gan_classifier"; export GAN_MULTIHEAD="--gan_multihead";;
    *) echo "Unknown GAN mode: $gan"; exit 1;;
  esac

  export DATASET_NAME="$ds"
  export USE_SOURCE_TEACHER="$use_src"
  export USE_TARGET_TEACHER="$use_tgt"
  export TRAIN_TARGET_TEACHER=0   # explicitly do NOT train a target teacher here
  export DMD_SOURCE_WEIGHT="$sw"
  export DMD_TARGET_WEIGHT="$tw"

  local s_tag t_tag
  s_tag="$(fmtw "$sw")"; t_tag="$(fmtw "$tw")"
  local tag="BABIES_${row}_src${use_src}_tgt${use_tgt}_SW${s_tag}_TW${t_tag}_gan${gan}"

  echo "[LOCAL] ds=$ds | row=$row | src=$use_src tgt=$use_tgt SW=$sw TW=$tw | GAN=$gan | tag=$tag"

  export EXTRA_TAG="_${tag}"
  submit_run "$tag"
}

# ---------------- SINGLE RUN: babies, DMDsrc only, multi-head GAN, no target teacher ----------------
run_row "babies" 1 0 1.0 0.0 "multi" "dmdsrc_only_gan_multi"