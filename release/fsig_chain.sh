#!/usr/bin/env bash
# Run the FSIG release jobs one after another on a single GPU.
# Usage: nohup bash release/fsig_chain.sh babies metfaces > ~/uni-DAD-runs/chain.log 2>&1 &
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate unidad
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$HOME/uni-DAD-runs}"
cd "$REPO"
for ds in "$@"; do
  for mode in train eval; do
    log="$RUNS_ROOT/chain_${ds}_${mode}.log"
    echo "[chain] $(date -Is) start $ds $mode -> $log"
    bash release/fsig_release.sh "$ds" "$mode" > "$log" 2>&1
    rc=$?
    echo "[chain] $(date -Is) end $ds $mode rc=$rc"
    if [[ $rc -ne 0 ]]; then echo "[chain] stopping: $ds $mode failed"; exit $rc; fi
  done
done
echo "[chain] $(date -Is) all done"
