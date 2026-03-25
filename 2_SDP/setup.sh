#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-uni_dad_sdp}"
ENV_YAML="${ENV_YAML:-$SCRIPT_DIR/uni_dad_sdp_environment.yml}"

# Optional cache location; user may override
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/.cache}"
mkdir -p "$CACHE_DIR"

export PIP_CACHE_DIR="$CACHE_DIR/pip"
export HF_HOME="$CACHE_DIR/huggingface"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export TMPDIR="$CACHE_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

echo "== Project installer =="
echo "[INFO] Project root: $SCRIPT_DIR"

if command -v conda >/dev/null 2>&1; then
  echo "[INFO] conda detected"

  if [[ ! -f "$ENV_YAML" ]]; then
    echo "[ERROR] Environment file not found: $ENV_YAML"
    exit 1
  fi

  eval "$(conda shell.bash hook)"

  TMP_ENV_YAML="$(mktemp "$TMPDIR/uni_dad_sdp_env_XXXXXX.yml")"
  trap 'rm -f "$TMP_ENV_YAML"' EXIT

  python - "$ENV_YAML" "$TMP_ENV_YAML" "$ENV_NAME" <<'PYENV'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
env_name = sys.argv[3]

lines = src.read_text(encoding='utf-8').splitlines()
out = []
name_written = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith('prefix:'):
        continue
    if stripped.startswith('name:'):
        out.append(f'name: {env_name}')
        name_written = True
        continue
    out.append(line)

if not name_written:
    out.insert(0, f'name: {env_name}')

dst.write_text('\n'.join(out) + '\n', encoding='utf-8')
PYENV

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] Updating existing conda env: $ENV_NAME"
    conda env update -n "$ENV_NAME" -f "$TMP_ENV_YAML" --prune
  else
    echo "[INFO] Creating conda env: $ENV_NAME"
    conda env create -f "$TMP_ENV_YAML"
  fi

  conda activate "$ENV_NAME"
  python -V

  # Install the local project after the environment dependencies are resolved.
  pip install --upgrade pip
  pip install -e .

  echo
  echo "Done."
  echo "Activate with: conda activate $ENV_NAME"
  echo "Environment spec used: $ENV_YAML"

else
  echo "[INFO] conda not found, using venv fallback"
  echo "[WARN] $ENV_YAML cannot be applied without conda; falling back to editable install only"

  PY_BIN="$(command -v python3 || command -v python)"
  "$PY_BIN" -m venv venv
  source venv/bin/activate

  pip install --upgrade pip

  pip install -e .

  echo
  echo "Done."
  echo "Activate with: source venv/bin/activate"
fi
