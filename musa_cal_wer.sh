#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

unset CUDA_VISIBLE_DEVICES
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

bash "$SCRIPT_DIR/cal_wer.sh" \
  "${SEED_TTS_META:-$WORKSPACE_ROOT/data/seedtts_testset/en/meta.lst}" \
  "${SEED_TTS_OUTPUT:-$WORKSPACE_ROOT/outputs/seedtts_eval/en}" \
  "${SEED_TTS_LANG:-en}"
