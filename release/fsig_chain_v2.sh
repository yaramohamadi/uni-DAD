#!/usr/bin/env bash
# Run v2 FSIG release jobs one after another. Usage: nohup bash release/fsig_chain_v2.sh babies metfaces &
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate unidad
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$HOME/uni-DAD-runs}"
cd "$REPO"
for ds in "$@"; do
  log="$RUNS_ROOT/chain_v2_${ds}.log"
  echo "[chain] $(date -Is) start $ds -> $log"
  bash release/fsig_release_v2.sh "$ds" > "$log" 2>&1
  rc=$?
  echo "[chain] $(date -Is) end $ds rc=$rc"
  [[ $rc -ne 0 ]] && { echo "[chain] stopping: $ds failed"; exit $rc; }
done
echo "[chain] $(date -Is) all done"
