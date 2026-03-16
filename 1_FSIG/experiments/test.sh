#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/defaults.sh"
print_config

cmd=(
  python -u main/dhariwal/test_dhariwal.py
  --folder "$OUTPUT_PATH"
  --wandb_name "${WANDB_NAME}_eval_uniform"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_project "$WANDB_PROJECT"
  --fid_npz_root "$FID_NPZ_ROOT"
  --category "$CATEGORY"
  --resolution "$RESOLUTION"
  --label_dim "$LABEL_DIM"
  --label_mode uniform
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --total_eval_samples "$TOTAL_EVAL_SAMPLES"
  --conditioning_sigma "$CONDITIONING_SIGMA"
  --lpips_cluster_size "$LPIPS_CLUSTER_SIZE"
  --fewshotdataset "$FEWSHOT_DATASET"
  --denoising
  --num_denoising_step "$NUM_DENOISING_STEP"
)

[[ -n "$HAS_NULL" ]] && cmd+=("$HAS_NULL")
[[ -n "$NO_LPIPS" ]] && cmd+=("$NO_LPIPS")
[[ -n "$USE_BF16" ]] && cmd+=("$USE_BF16")
[[ -n "$MAKE_DDIM_GRID" ]] && cmd+=("$MAKE_DDIM_GRID")
[[ -n "$DDIM_GRID_ONLY" ]] && cmd+=("$DDIM_GRID_ONLY")
[[ -n "$EVAL_BEST_ONCE" ]] && cmd+=("$EVAL_BEST_ONCE")

echo "[test] Starting evaluation..."
CUDA_VISIBLE_DEVICES="$TEST_GPUS" "${cmd[@]}"