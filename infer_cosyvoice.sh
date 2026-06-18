#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

musa_devices="${MUSA_VISIBLE_DEVICES:-}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-}"

if [ -n "$musa_devices" ] && [ -n "$cuda_devices" ]; then
  echo "Set only one of MUSA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES." >&2
  exit 2
elif [ -n "$musa_devices" ]; then
  backend=musa
  visible_devices_var=MUSA_VISIBLE_DEVICES
  device_list="$musa_devices"
elif [ -n "$cuda_devices" ]; then
  backend=cuda
  visible_devices_var=CUDA_VISIBLE_DEVICES
  device_list="$cuda_devices"
else
  echo "Set MUSA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES, for example MUSA_VISIBLE_DEVICES=0,1." >&2
  exit 2
fi

default_model_dir="$WORKSPACE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B-test"

META=${1:-${SEED_TTS_META:-$WORKSPACE_ROOT/data/seedtts_testset/en/meta.lst}}
OUT=${2:-${SEED_TTS_OUTPUT:-$WORKSPACE_ROOT/outputs/seedtts_eval/en}}
SEED="${COSYVOICE_SEED:-1986}"
export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-$default_model_dir}"
export COSYVOICE_ROOT="${COSYVOICE_ROOT:-$WORKSPACE_ROOT/CosyVoice}"

if [ ! -f "$META" ]; then
  echo "Seed-TTS meta file not found: $META" >&2
  exit 1
fi
if [ ! -d "$COSYVOICE_MODEL_DIR" ]; then
  echo "CosyVoice model directory not found: $COSYVOICE_MODEL_DIR" >&2
  exit 1
fi

IFS=',' read -r -a devices <<< "$device_list"

if [ "${#devices[@]}" -eq 0 ]; then
  echo "Device list must contain at least one device, for example 0 or 0,1." >&2
  exit 2
fi
for device in "${devices[@]}"; do
  if ! [[ "$device" =~ ^([0-9]+|GPU-[0-9A-Fa-f-]+)$ ]]; then
    echo "Invalid $backend device: $device" >&2
    exit 2
  fi
done

num_shards=${#devices[@]}
selected_devices="$(IFS=,; echo "${devices[*]}")"
mkdir -p "$OUT"

echo "Evaluation backend: $backend"
echo "Evaluation devices: $selected_devices ($num_shards workers)"
echo "Output directory: $OUT"

pids=()
for shard_index in "${!devices[@]}"; do
  env "$visible_devices_var=${devices[$shard_index]}" \
    python3 "$SCRIPT_DIR/infer_cosyvoice_seedtts.py" \
      "$META" \
      "$OUT" \
      --device-backend "$backend" \
      --num-shards "$num_shards" \
      --seed "$SEED" \
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
