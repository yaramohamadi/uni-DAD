#!/usr/bin/env bash
# Run the FSIG release jobs one after another on a single GPU.
# Usage: nohup bash release/fsig_chain.sh babies metfaces > chain.log 2>&1 &
set -uo pipefail
# activate the FSIG env if conda is available (CONDA_ENV=unidad by default)
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV:-unidad}"; fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO/1_FSIG/checkpoint_path}"
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
