#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
META=${1:-${SEED_TTS_META:-$WORKSPACE_ROOT/data/seedtts_testset/en/meta.lst}}
OUT=${2:-${SEED_TTS_OUTPUT:-$WORKSPACE_ROOT/outputs/seedtts_eval/en}}
DEVICE_LIST=${CUDA_DEVICE_LIST:-0}
export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-$WORKSPACE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B}"
export COSYVOICE_ROOT="${COSYVOICE_ROOT:-$WORKSPACE_ROOT/CosyVoice}"

if [ ! -f "$META" ]; then
  echo "Seed-TTS meta file not found: $META" >&2
  exit 1
fi
if [ ! -d "$COSYVOICE_MODEL_DIR" ]; then
  echo "CosyVoice model directory not found: $COSYVOICE_MODEL_DIR" >&2
  exit 1
fi

IFS=',' read -r -a devices <<< "$DEVICE_LIST"
if [ "${#devices[@]}" -eq 0 ]; then
  echo "CUDA_DEVICE_LIST must contain at least one device, for example 0 or 0,1." >&2
  exit 2
fi
for device in "${devices[@]}"; do
  if ! [[ "$device" =~ ^[0-9]+$ ]]; then
    echo "Invalid CUDA device in CUDA_DEVICE_LIST=$DEVICE_LIST: $device" >&2
    exit 2
  fi
done

num_shards=${#devices[@]}
mkdir -p "$OUT"

pids=()
for shard_index in "${!devices[@]}"; do
  CUDA_VISIBLE_DEVICES="${devices[$shard_index]}" python3 "$SCRIPT_DIR/infer_cosyvoice_seedtts.py" \
    "$META" \
    "$OUT" \
    --num-shards "$num_shards" \
    --overwrite \
    --shard-index "$shard_index" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
