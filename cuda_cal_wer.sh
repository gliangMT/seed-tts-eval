#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0}"
IFS=',' read -r -a devices <<< "$CUDA_DEVICE_LIST"
export NUM_GPUS="${NUM_GPUS:-${#devices[@]}}"

bash "$SCRIPT_DIR/cal_wer.sh" \
  "${SEED_TTS_META:-$WORKSPACE_ROOT/data/seedtts_testset/en/meta.lst}" \
  "${SEED_TTS_OUTPUT:-$WORKSPACE_ROOT/outputs/seedtts_eval/en}" \
  "${SEED_TTS_LANG:-en}"
