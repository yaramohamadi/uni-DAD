#!/usr/bin/env bash
set -euo pipefail

MANIFEST_CSV="${1:-}"
SAVE_PREFIX="${2:-clip_eval_results}"

if [[ -z "$MANIFEST_CSV" ]]; then
  echo "Usage: ./run_eval.sh path/to/manifest.csv [save_prefix]"
  exit 1
fi

python evaluation/eval_posttrain.py \
  --manifest_csv "$MANIFEST_CSV" \
  --save_prefix "$SAVE_PREFIX"

  """
Example usage:
  ./run_eval.sh evaluation/eval_manifest.csv outputs/final_eval
  """