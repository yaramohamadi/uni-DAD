#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

start_streaming_test() {
  ( setsid bash experiments/test.sh ) &
  TEST_PID=$!
  TEST_PGID="$(ps -o pgid= "$TEST_PID" | tr -d ' ')" || true
  echo "[orchestrator] started streaming test (pid=$TEST_PID, pgid=${TEST_PGID:-?})"
}

stop_streaming_test() {
  if [[ -n "${TEST_PID:-}" ]] && kill -0 "$TEST_PID" 2>/dev/null; then
    echo "[orchestrator] stopping streaming test (pid=$TEST_PID, pgid=${TEST_PGID:-?})"
    if [[ -n "${TEST_PGID:-}" ]]; then
      kill -TERM "-$TEST_PGID" 2>/dev/null || true
    else
      kill -TERM "$TEST_PID" 2>/dev/null || true
    fi
    wait "$TEST_PID" 2>/dev/null || true
    TEST_PID=""
    TEST_PGID=""
  fi
}

trap 'stop_streaming_test' EXIT INT TERM ERR

start_streaming_test
bash experiments/train.sh
train_rc=$?
stop_streaming_test
exit "$train_rc"