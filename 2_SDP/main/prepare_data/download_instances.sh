#!/usr/bin/env bash
set -euo pipefail

# This helper fills 2_SDP/data/instance_images in one of two ways:
# 1) default mode: download the archived Google DreamBooth dataset and copy all
#    subject folders into the repo's instance_images directory.
# 2) local mode: copy a folder that already exists on your machine.
#
# Local mode supports two layouts:
# - a dataset root containing one subfolder per subject
# - a single subject folder containing image files directly
#
# Examples:
#   bash main/prepare_data/download_dreambooth_instances.sh
#   bash main/prepare_data/download_dreambooth_instances.sh --local-source /path/to/subjects_root
#   bash main/prepare_data/download_dreambooth_instances.sh --local-source /path/to/my_subject --subject-name my_subject

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TARGET_ROOT="${TARGET_ROOT:-$PROJECT_ROOT/data/instance_images}"
PROMPTS_OUT="${PROMPTS_OUT:-$PROJECT_ROOT/data/prompts_and_classes.txt}"
REPO_URL="${REPO_URL:-https://github.com/google/dreambooth.git}"
ARCHIVE_URL="${ARCHIVE_URL:-https://codeload.github.com/google/dreambooth/tar.gz/refs/heads/main}"
COPY_PROMPTS_FILE="${COPY_PROMPTS_FILE:-1}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
MODE="${MODE:-dreambooth}"
LOCAL_SOURCE_DIR="${LOCAL_SOURCE_DIR:-}"
SUBJECT_NAME="${SUBJECT_NAME:-}"

TMP_DIR=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --dreambooth            Download the archived Google DreamBooth dataset (default).
  --local-source PATH     Copy subjects from a local folder on this machine.
  --subject-name NAME     Destination folder name when --local-source points to a
                          single subject folder that contains images directly.
  --target-root PATH      Output folder for subject folders.
  --prompts-out PATH      Where to copy prompts_and_classes.txt when available.
  --force-overwrite       Replace existing destination folders/files.
  --no-copy-prompts       Do not copy prompts_and_classes.txt.
  -h, --help              Show this help message.

Local source layout examples:
  1) One folder per subject:
     /path/to/my_subjects/
       dog/
       cat/
       vase/

  2) One folder containing images for a single subject:
     /path/to/my_dog/
       00.jpg
       01.jpg
       02.png

Examples:
  $(basename "$0")
  $(basename "$0") --local-source /path/to/my_subjects
  $(basename "$0") --local-source /path/to/my_dog --subject-name dog_custom
EOF
}

cleanup() {
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dreambooth)
      MODE="dreambooth"
      shift
      ;;
    --local-source)
      [[ $# -ge 2 ]] || { echo "[ERROR] --local-source requires a path"; exit 1; }
      MODE="local"
      LOCAL_SOURCE_DIR="$2"
      shift 2
      ;;
    --subject-name)
      [[ $# -ge 2 ]] || { echo "[ERROR] --subject-name requires a value"; exit 1; }
      SUBJECT_NAME="$2"
      shift 2
      ;;
    --target-root)
      [[ $# -ge 2 ]] || { echo "[ERROR] --target-root requires a path"; exit 1; }
      TARGET_ROOT="$2"
      shift 2
      ;;
    --prompts-out)
      [[ $# -ge 2 ]] || { echo "[ERROR] --prompts-out requires a path"; exit 1; }
      PROMPTS_OUT="$2"
      shift 2
      ;;
    --force-overwrite)
      FORCE_OVERWRITE=1
      shift
      ;;
    --no-copy-prompts)
      COPY_PROMPTS_FILE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      echo
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$TARGET_ROOT"

echo "== Subject folder importer =="
echo "[INFO] Mode: $MODE"
echo "[INFO] Target folder: $TARGET_ROOT"

copy_subject_dir() {
  local src_dir="$1"
  local dest_name="${2:-$(basename "$src_dir")}"
  local dest_dir="$TARGET_ROOT/$dest_name"

  if [[ -e "$dest_dir" ]]; then
    if [[ "$FORCE_OVERWRITE" == "1" ]]; then
      echo "[WARN] Replacing existing folder: $dest_dir"
      rm -rf "$dest_dir"
    else
      echo "[SKIP] $dest_dir already exists. Set --force-overwrite or FORCE_OVERWRITE=1 to replace it."
      return 1
    fi
  fi

  cp -a "$src_dir" "$dest_dir"
  echo "[COPY] $dest_name -> $dest_dir"
  return 0
}

copy_prompts_file_if_present() {
  local src_file="$1"

  if [[ "$COPY_PROMPTS_FILE" != "1" || ! -f "$src_file" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$PROMPTS_OUT")"
  if [[ -e "$PROMPTS_OUT" && "$FORCE_OVERWRITE" != "1" ]]; then
    echo "[SKIP] $PROMPTS_OUT already exists. Set --force-overwrite or FORCE_OVERWRITE=1 to replace it."
  else
    cp "$src_file" "$PROMPTS_OUT"
    echo "[COPY] $(basename "$src_file") -> $PROMPTS_OUT"
  fi
}

folder_has_top_level_images() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f \(     -iname '*.jpg' -o     -iname '*.jpeg' -o     -iname '*.png' -o     -iname '*.webp' -o     -iname '*.bmp'   \) -print -quit | grep -q .
}

copy_local_source() {
  local copied=0
  local skipped=0

  if [[ -z "$LOCAL_SOURCE_DIR" ]]; then
    echo "[ERROR] Local mode requires --local-source PATH"
    exit 1
  fi
  if [[ ! -d "$LOCAL_SOURCE_DIR" ]]; then
    echo "[ERROR] Local source directory not found: $LOCAL_SOURCE_DIR"
    exit 1
  fi

  echo "[INFO] Local source: $LOCAL_SOURCE_DIR"

  # If the provided folder already contains images, treat it as a single subject.
  # Otherwise, treat each immediate subdirectory as one subject folder.
  if folder_has_top_level_images "$LOCAL_SOURCE_DIR"; then
    local resolved_name="${SUBJECT_NAME:-$(basename "$LOCAL_SOURCE_DIR")}" 
    if copy_subject_dir "$LOCAL_SOURCE_DIR" "$resolved_name"; then
      copied=$((copied + 1))
    else
      skipped=$((skipped + 1))
    fi
  else
    local found_subdir=0
    while IFS= read -r -d '' subject_dir; do
      found_subdir=1
      if copy_subject_dir "$subject_dir"; then
        copied=$((copied + 1))
      else
        skipped=$((skipped + 1))
      fi
    done < <(find "$LOCAL_SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    if [[ "$found_subdir" == "0" ]]; then
      echo "[ERROR] $LOCAL_SOURCE_DIR contains neither top-level images nor subject subfolders."
      exit 1
    fi
  fi

  copy_prompts_file_if_present "$LOCAL_SOURCE_DIR/prompts_and_classes.txt"

  echo "[DONE] Copied $copied subject folders to $TARGET_ROOT"
  if [[ "$skipped" -gt 0 ]]; then
    echo "[DONE] Skipped $skipped existing subject folders"
  fi
}

download_archive() {
  local archive="$TMP_DIR/dreambooth-main.tar.gz"

  echo "[INFO] Source repo: $REPO_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$ARCHIVE_URL" -o "$archive"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive" "$ARCHIVE_URL"
  else
    echo "[ERROR] Neither curl nor wget is available. Install one of them and retry."
    exit 1
  fi

  tar -xzf "$archive" -C "$TMP_DIR"
}

copy_dreambooth_dataset() {
  local src_root
  local copied=0
  local skipped=0

  TMP_DIR="$(mktemp -d)"
  download_archive

  src_root="$(find "$TMP_DIR" -maxdepth 2 -type d -path '*/dataset' | head -n 1)"
  if [[ -z "$src_root" ]]; then
    echo "[ERROR] Could not find the dataset directory in the downloaded archive."
    exit 1
  fi

  while IFS= read -r -d '' subject_dir; do
    if copy_subject_dir "$subject_dir"; then
      copied=$((copied + 1))
    else
      skipped=$((skipped + 1))
    fi
  done < <(find "$src_root" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  copy_prompts_file_if_present "$src_root/prompts_and_classes.txt"

  echo "[DONE] Copied $copied subject folders to $TARGET_ROOT"
  if [[ "$skipped" -gt 0 ]]; then
    echo "[DONE] Skipped $skipped existing subject folders"
  fi
}

if [[ "$MODE" == "local" ]]; then
  copy_local_source
else
  copy_dreambooth_dataset
fi
