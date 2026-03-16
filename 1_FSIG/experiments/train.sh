#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/defaults.sh"
print_config

cmd=(
  torchrun
  --nproc_per_node "$NPROC_PER_NODE"
  --nnodes "$NNODES"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  main/dhariwal/train_dhariwal.py
  --generator_lr "$GEN_LR"
  --guidance_lr "$GEN_LR"
  --train_iters "$TRAIN_ITERS"
  --output_path "$OUTPUT_PATH"
  --batch_size "$BATCH_SIZE"
  --initialie_generator
  --log_iters "$LOG_ITERS"
  --resolution "$RESOLUTION"
  --label_dim "$LABEL_DIM"
  --dataset_name "$DATASET_NAME"
  --seed "$SEED"
  --model_id "$CHECKPOINT_INIT"
  --wandb_iters "$WANDB_ITERS"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_project "$WANDB_PROJECT"
  --wandb_name "$WANDB_NAME"
  --real_image_path "$REAL_IMAGE_PATH"
  --dfake_gen_update_ratio "$DFAKE_GEN_UPDATE_RATIO"
  --cls_loss_weight "$CLS_LOSS_WEIGHT"
  --gen_cls_loss_weight "$GEN_CLS_LOSS_WEIGHT"
  --dmd_loss_weight "$DMD_LOSS_WEIGHT"
  --diffusion_gan
  --diffusion_gan_max_timestep "$DIFFUSION_GAN_MAX_TIMESTEP"
  --delete_ckpts
  --max_checkpoint "$MAX_CHECKPOINT"
  --denoising
  --num_denoising_step "$NUM_DENOISING_STEP"
  --denoising_sigma_end "$DENOISING_SIGMA_END"
  --label_dropout_p "$LABEL_DROPOUT_P"
  --gan_head_type "$GAN_HEAD_TYPE"
  --gan_head_layers "$GAN_HEAD_LAYERS"
  --gan_adv_loss "$GAN_ADV_LOSS"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --use_source_teacher "$USE_SOURCE_TEACHER"
  --use_target_teacher "$USE_TARGET_TEACHER"
  --train_target_teacher "$TRAIN_TARGET_TEACHER"
  --dmd_source_weight "$DMD_SOURCE_WEIGHT"
  --dmd_target_weight "$DMD_TARGET_WEIGHT"
)

[[ -n "$GAN_CLASSIFIER" ]] && cmd+=("$GAN_CLASSIFIER")
[[ -n "$GAN_MULTIHEAD" ]] && cmd+=("$GAN_MULTIHEAD")
[[ -n "$USE_BF16" ]] && cmd+=("$USE_BF16")
[[ -n "$REVERSE_DMD" ]] && cmd+=("$REVERSE_DMD")
[[ -n "$TRAIN_FAKE_ON_REAL" ]] && cmd+=("$TRAIN_FAKE_ON_REAL")
[[ -n "$TT_MATCH_GUIDANCE" ]] && cmd+=("$TT_MATCH_GUIDANCE")
[[ -n "$TARGET_TEACHER_CHECKPOINT_PATH" ]] && cmd+=(--target_teacher_ckpt_path "$TARGET_TEACHER_CHECKPOINT_PATH")
[[ -n "$CHECKPOINT_PATH" ]] && cmd+=(--checkpoint_path "$CHECKPOINT_PATH")

echo "[train] Starting training..."
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "${cmd[@]}"