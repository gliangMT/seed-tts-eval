#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0}"
IFS=',' read -r -a devices <<< "$CUDA_DEVICE_LIST"
export EVAL_BACKEND=cuda
export NUM_GPUS="${NUM_GPUS:-${#devices[@]}}"
export WAVLM_LARGE_CKPT="${WAVLM_LARGE_CKPT:-$WORKSPACE_ROOT/data/models/wavlm_large.pt}"

bash "$SCRIPT_DIR/cal_sim.sh" \
  "${SEED_TTS_META:-$WORKSPACE_ROOT/data/seedtts_testset/en/meta.lst}" \
  "${SEED_TTS_OUTPUT:-$WORKSPACE_ROOT/outputs/seedtts_eval/en}" \
  "${WAVLM_FINETUNE_CKPT:-$WORKSPACE_ROOT/data/models/wavlm_large_finetune.pth}"
