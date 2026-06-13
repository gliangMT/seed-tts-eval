#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 CHECKPOINT OUTPUT_DIR [DEVICE] [LIMIT] [COMPONENT]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

checkpoint="$(realpath "$1")"
output_dir="$2"
device="${3:-0}"
limit="${4:-}"
component="${5:-${MODEL_COMPONENT:-llm}}"

base_model_dir="${BASE_MODEL_DIR:-${WORKSPACE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B}"
meta="${SEED_TTS_META:-${WORKSPACE_ROOT}/data/seedtts_testset/en/meta.lst}"
cosyvoice_root="${COSYVOICE_ROOT:-${WORKSPACE_ROOT}/CosyVoice}"

if [ ! -f "${checkpoint}" ]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 1
fi
if [ ! -d "${base_model_dir}" ]; then
  echo "Base model directory not found: ${base_model_dir}" >&2
  exit 1
fi
if [ ! -f "${meta}" ]; then
  echo "Seed-TTS meta file not found: ${meta}" >&2
  exit 1
fi
if [ "${component}" != "llm" ] && [ "${component}" != "flow" ]; then
  echo "COMPONENT must be llm or flow: ${component}" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "${device}"
if [ "${#devices[@]}" -eq 0 ]; then
  echo "DEVICE must contain at least one CUDA device, for example 0 or 0,1,2,3." >&2
  exit 2
fi
for eval_device in "${devices[@]}"; do
  if ! [[ "${eval_device}" =~ ^[0-9]+$ ]]; then
    echo "Invalid CUDA device in DEVICE=${device}: ${eval_device}" >&2
    exit 2
  fi
done
if [ -n "${limit}" ]; then
  if ! [[ "${limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be a positive integer: ${limit}" >&2
    exit 2
  fi
  if [ "${limit}" -lt "${#devices[@]}" ]; then
    devices=("${devices[@]:0:${limit}}")
    device="$(IFS=,; echo "${devices[*]}")"
  fi
fi
num_devices="${#devices[@]}"

mkdir -p "${output_dir}"
output_dir="$(realpath "${output_dir}")"
model_dir="${output_dir}/model"
wav_dir="${output_dir}/wavs"
raw_score="${output_dir}/wer.raw"
score_file="${output_dir}/wer.txt"
wav_list="${output_dir}/wav_res_ref_text"

mkdir -p "${model_dir}" "${wav_dir}"

# Reuse fixed model assets and replace only the component being trained.
while IFS= read -r asset; do
  name="$(basename "${asset}")"
  if [ "${name}" != "${component}.pt" ]; then
    ln -sfn "${asset}" "${model_dir}/${name}"
  fi
done < <(find "${base_model_dir}" -mindepth 1 -maxdepth 1 -print)
ln -sfn "${checkpoint}" "${model_dir}/${component}.pt"

infer_args=(
  "${meta}"
  "${wav_dir}"
  --model-dir "${model_dir}"
  --cosyvoice-root "${cosyvoice_root}"
)
if [ -n "${limit}" ]; then
  infer_args+=(--limit "${limit}")
fi
if [ "${EVAL_OVERWRITE:-0}" = "1" ]; then
  infer_args+=(--overwrite)
fi

echo "Evaluating checkpoint: ${checkpoint}"
echo "Evaluating component: ${component}"
echo "Evaluation devices: ${device} (${num_devices} workers)"

infer_pids=()
for shard_index in "${!devices[@]}"; do
  eval_device="${devices[$shard_index]}"
  CUDA_VISIBLE_DEVICES="${eval_device}" \
    python3 "${SCRIPT_DIR}/infer_cosyvoice_seedtts.py" \
      "${infer_args[@]}" \
      --num-shards "${num_devices}" \
      --shard-index "${shard_index}" &
  infer_pids+=("$!")
done

infer_status=0
for pid in "${infer_pids[@]}"; do
  if ! wait "${pid}"; then
    infer_status=1
  fi
done
if [ "${infer_status}" -ne 0 ]; then
  echo "CosyVoice inference worker failed; partial wav files kept in ${wav_dir}" >&2
  exit "${infer_status}"
fi

wer_args=("${meta}" "${wav_dir}" en)
if [ -n "${limit}" ]; then
  wer_args+=("${limit}")
fi
CUDA_DEVICE_LIST="${device}" \
  NUM_GPUS="${num_devices}" \
  bash "${SCRIPT_DIR}/cal_wer.sh" "${wer_args[@]}"
cp "${wav_dir}/wav_res_ref_text.wer" "${score_file}"
cp "${wav_dir}/wav_res_ref_text" "${wav_list}"

touch "${output_dir}/.complete"
echo "WER result: ${score_file}"
