#!/usr/bin/env bash
# Run v2 FSIG release jobs one after another. Usage: nohup bash release/fsig_chain_v2.sh babies metfaces &
set -uo pipefail
# activate the FSIG env if conda is available (CONDA_ENV=unidad by default)
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV:-unidad}"; fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO/1_FSIG/checkpoint_path}"
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
