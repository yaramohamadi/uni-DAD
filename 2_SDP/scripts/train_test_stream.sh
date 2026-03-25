#!/usr/bin/env bash
set -euo pipefail
set -x

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ----- unique per-run rendezvous config -----
MASTER_ADDR=127.0.0.1
MASTER_PORT=$(( 10000 + RANDOM % 40000 ))   # unique random port per run
RDZV_ID="run_$$_$(date +%s%N)"              # unique id per run

# ----- GPU selection (change per run if needed) -----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # set 0 / 1 in each shell


# Paths
# Adapt to your repo structure and data locations. The script assumes a shared filesystem across nodes.
export RUN_ROOT=/projets/$USER/uni-dad/sdp_training
CHECKPOINT_PATH="$RUN_ROOT/checkpoints/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35"
WANDB_ENTITY="your wandb entity here"
WANDB_PROJECT="Uni-DAD_SDP_training"
LMDB_PATH="$RUN_ROOT/data/instances.lmdb"
PROMPT_PATH="$RUN_ROOT/data/prompts_and_classes.txt"
GEN_OUT_ROOT="$RUN_ROOT/output_dmd2"
mkdir -p "$CHECKPOINT_PATH"

# Optional: make the wandb run name unique too HIGHLR_2e-6_LOSSGEN0.75e-1_LOSSGUID1.5e-2
WANDB_NAME="UniDAD_SDV15_baseline_8node_guidance${G_S}_lr${GEN_LR}_seed${SEED}_dfake${GAN_D}_diffusion1000_gan${GAN_G}_resume_fid8.35"

# ------------------- HPARAMS -------------------
GEN_LR="5e-6"
GUID_LR="5e-6"
BATCH_SIZE="1"
TRAIN_ITERS="5000"
SEED="10"
G_S="7.5"
INSTANCE_ID="3"
GAN_G="0.001"
GAN_D="0.01"

# ----- Launch WITHOUT --standalone; use explicit rendezvous -----
exec torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=${RDZV_ID} \
  main/pipeline/train_sd.py \
  --model_id "runwayml/stable-diffusion-v1-5" \
  --real_image_path "$LMDB_PATH" \
  --train_lmdb_path "$LMDB_PATH" \
  --rare_token prt \
  --train_iters "${TRAIN_ITERS}" \
  --generator_lr "${GEN_LR}" \
  --guidance_lr "${GUID_LR}" \
  --batch_size "${BATCH_SIZE}" \
  --initialie_generator --log_iters 1000 \
  --resolution 512 \
  --latent_resolution 64 \
  --seed 10 \
  --use_fp16 \
  --real_guidance_scale "${G_S}" \
  --fake_guidance_scale 1.0 \
  --max_grad_norm 10.0 \
  --log_loss \
  --gen_include_labels "${INSTANCE_ID}" \
  --dfake_gen_update_ratio 10 \
  --diffusion_gan \
  --diffusion_gan_max_timestep 1000 \
  --ckpt_only_path "$CHECKPOINT_PATH/checkpoint_model_041000" \
  --output_path "$CHECKPOINT_PATH/subject_ft_sd15" \
  --wandb_iters 50 \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$WANDB_NAME" \
  --grid_size 4 \
  --cache_dir "$RUN_ROOT/.cache/dmd2" \
  --skip_accelerate_state \
  --gen_pause_steps 1,200,600,800,1000,1200,1600,2000,2200,2600,3000,3500,4000,4500,5000,6000,7000,8000,9000,10000 \
  --gen_prompts_file $PROMPT_PATH \
  --gen_outputs_root "$GEN_OUT_ROOT" \
  --gen_training_name "$WANDB_NAME" \
  --gen_samples_per_prompt 4 \
  --gen_seeds 0,1,2,3 \
  --gen_delete_ckpt \
  --eval_enable \
  --multihead_gan \
  --gan_bce_weight_d "$GAN_D"\
  --gan_bce_weight_g "$GAN_G" \
  --gan_bce_random_t \
  --no-mhgan-freeze-encoder-dstep \
  --eval_finalize_on_last_pause \
