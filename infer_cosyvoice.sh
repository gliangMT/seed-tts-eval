#!/bin/bash

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
META=/home/cosyvoice-test/data/seedtts_testset/en/meta.lst
OUT=/home/cosyvoice-test/outputs/seedtts_eval/en
export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B}"
export DEFAULT_COSYVOICE_ROOT="${DEFAULT_COSYVOICE_ROOT:-/home/cosyvoice-test/CosyVoice}"
mkdir -p "$OUT"

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  MUSA_VISIBLE_DEVICES=$rank python3 "$SCRIPT_DIR/infer_cosyvoice_seedtts.py" \
    "$META" \
    "$OUT" \
    --num-shards 8 \
    --overwrite \
    --shard-index $rank &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
