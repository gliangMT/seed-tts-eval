#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

backend="${EVAL_BACKEND:-}"
if [ -z "$backend" ]; then
  if [ -n "${MUSA_DEVICE_LIST:-}" ]; then
    backend=musa
  elif [ -n "${CUDA_DEVICE_LIST:-}" ]; then
    backend=cuda
  else
    backend="${DEFAULT_EVAL_BACKEND:-musa}"
  fi
fi

case "$backend" in
  musa)
    visible_devices_var=MUSA_VISIBLE_DEVICES
    device_list="${MUSA_DEVICE_LIST:-}"
    default_num_devices="${NUM_GPUS:-${ARNOLD_WORKER_GPU:-8}}"
    default_model_dir="$WORKSPACE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B-test"
    ;;
  cuda)
    visible_devices_var=CUDA_VISIBLE_DEVICES
    device_list="${CUDA_DEVICE_LIST:-0}"
    default_num_devices=""
    default_model_dir="$WORKSPACE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B"
    ;;
  *)
    echo "EVAL_BACKEND must be musa or cuda: $backend" >&2
    exit 2
    ;;
esac

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

if [ -n "$device_list" ]; then
  IFS=',' read -r -a devices <<< "$device_list"
else
  if ! [[ "$default_num_devices" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS/ARNOLD_WORKER_GPU must be a positive integer: $default_num_devices" >&2
    exit 2
  fi
  devices=()
  for rank in $(seq 0 $((default_num_devices - 1))); do
    devices+=("$rank")
  done
fi

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
mkdir -p "$OUT"

pids=()
for shard_index in "${!devices[@]}"; do
  env "$visible_devices_var=${devices[$shard_index]}" \
    EVAL_BACKEND="$backend" \
    python3 "$SCRIPT_DIR/infer_cosyvoice_seedtts.py" \
      "$META" \
      "$OUT" \
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
