#!/usr/bin/env bash
set -euo pipefail

# ---------- environment ----------
ENV_NAME="unidad"

# ---------- fixed versions ----------
PYTHON_VERSION="3.10.13"
TORCH_VERSION="2.0.1"
TORCHVISION_VERSION="0.15.2"
TORCH_CUDA_TAG="cu118"

# ---------- paths ----------
FSIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${FSIG_ROOT}/.." && pwd)"
DHARIWAL_SUBMODULE_PATH="${REPO_ROOT}/third_party/dhariwal"

cd "$FSIG_ROOT"

install_torch() {
  python -m pip install \
    "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
    "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}" \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
}

install_repo() {
  python -m pip install -U pip setuptools wheel
  install_torch
  python -m pip install -r requirements.txt
  python -m pip install -e .

  git submodule update --init --recursive
  if [[ -d "$DHARIWAL_SUBMODULE_PATH" ]]; then
    python -m pip install -e "$DHARIWAL_SUBMODULE_PATH" --no-deps --no-build-isolation
  fi
}

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
fi

conda activate "$ENV_NAME"
install_repo

echo "Environment setup complete."
echo "Activate later with: conda activate $ENV_NAME"
