#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 MODEL_DIR OUTPUT_ROOT [INTERVAL] [DEVICE_OR_DEVICE_LIST] [COMPONENT]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="$(realpath "$1")"
output_root="$2"
interval="${3:-5}"
device="${4:-0}"
component="${5:-${MODEL_COMPONENT:-llm}}"
poll_seconds="${POLL_SECONDS:-30}"
max_epoch="${MAX_EPOCH:-}"
limit="${EVAL_LIMIT:-}"
wer_window="${WER_WINDOW:-3}"
wer_delta="${WER_DELTA:-0.1}"
stop_file="${EARLY_STOP_FILE:-${model_dir}/STOP_TRAIN}"

if [ ! -d "${model_dir}" ]; then
  echo "Training model directory not found: ${model_dir}" >&2
  exit 1
fi
if ! [[ "${interval}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INTERVAL must be a positive integer: ${interval}" >&2
  exit 2
fi
if ! [[ "${wer_window}" =~ ^[2-9][0-9]*$ ]]; then
  echo "WER_WINDOW must be an integer >= 2: ${wer_window}" >&2
  exit 2
fi
if ! [[ "${wer_delta}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "WER_DELTA must be a non-negative number: ${wer_delta}" >&2
  exit 2
fi
if [ "${component}" != "llm" ] && [ "${component}" != "flow" ]; then
  echo "COMPONENT must be llm or flow: ${component}" >&2
  exit 2
fi

mkdir -p "${output_root}"
output_root="$(realpath "${output_root}")"
history_file="${output_root}/wer_history.tsv"

echo "Watching ${model_dir} for every ${interval} completed epochs."
echo "Evaluating component: ${component}"
echo "Evaluation devices: ${device}"
echo "Epoch files are zero-based, so epoch_4_whole.pt is the fifth completed epoch."
echo "Early stop rule: latest ${wer_window} WER values have max-min <= ${wer_delta} percentage points."
echo "Stop file: ${stop_file}"

record_wer_and_maybe_stop() {
  local completed_epoch="$1"
  local score_file="$2"
  local wer
  local recent_values
  local value_count
  local spread

  wer="$(sed -n 's/^WER: \([0-9.]*\)%$/\1/p' "${score_file}" | tail -1)"
  if [ -z "${wer}" ]; then
    echo "Could not parse WER from ${score_file}" >&2
    return 1
  fi

  if awk -F '\t' -v epoch="${completed_epoch}" '$1 == epoch { found=1 } END { exit !found }' \
      "${history_file}" 2>/dev/null; then
    return 0
  fi
  printf '%s\t%s\t%s\n' "${completed_epoch}" "${wer}" "${score_file}" >> "${history_file}"

  recent_values="$(tail -n "${wer_window}" "${history_file}" | cut -f2)"
  value_count="$(printf '%s\n' "${recent_values}" | sed '/^$/d' | wc -l)"
  if [ "${value_count}" -lt "${wer_window}" ]; then
    echo "WER history: ${value_count}/${wer_window} values collected."
    return 0
  fi

  spread="$(printf '%s\n' "${recent_values}" | awk '
    NR == 1 { min=$1; max=$1 }
    $1 < min { min=$1 }
    $1 > max { max=$1 }
    END { printf "%.6f", max-min }
  ')"
  echo "Latest ${wer_window} WER values: $(printf '%s ' ${recent_values})spread=${spread}"

  if awk -v spread="${spread}" -v delta="${wer_delta}" 'BEGIN { exit !(spread <= delta) }'; then
    tmp_stop="${stop_file}.tmp.$$"
    mkdir -p "$(dirname "${stop_file}")"
    {
      echo "Seed-TTS WER early stopping requested."
      echo "completed_epoch=${completed_epoch}"
      echo "window=${wer_window}"
      echo "delta=${wer_delta}"
      echo "spread=${spread}"
      echo "values=$(printf '%s ' ${recent_values})"
    } > "${tmp_stop}"
    mv "${tmp_stop}" "${stop_file}"
    echo "WER converged; requested training stop via ${stop_file}."
    return 2
  fi
}

while true; do
  found_pending=0

  while IFS= read -r checkpoint; do
    filename="$(basename "${checkpoint}")"
    epoch_index="${filename#epoch_}"
    epoch_index="${epoch_index%_whole.pt}"
    completed_epoch=$((epoch_index + 1))
    checkpoint_info="${checkpoint%.pt}.yaml"

    if (( completed_epoch % interval != 0 )); then
      continue
    fi
    if [ ! -f "${checkpoint_info}" ]; then
      continue
    fi

    epoch_output="${output_root}/epoch_${completed_epoch}"
    if [ -f "${epoch_output}/.complete" ]; then
      if record_wer_and_maybe_stop "${completed_epoch}" "${epoch_output}/wer.txt"; then
        :
      else
        status=$?
        if [ "${status}" -eq 2 ]; then
          exit 0
        fi
        exit "${status}"
      fi
      continue
    fi

    found_pending=1
    "${SCRIPT_DIR}/eval_cosyvoice_checkpoint.sh" \
      "${checkpoint}" "${epoch_output}" "${device}" "${limit}" "${component}"

    if record_wer_and_maybe_stop "${completed_epoch}" "${epoch_output}/wer.txt"; then
      :
    else
      status=$?
      if [ "${status}" -eq 2 ]; then
        exit 0
      fi
      exit "${status}"
    fi
  done < <(find "${model_dir}" -maxdepth 1 -type f -name 'epoch_*_whole.pt' | sort -V)

  if [ -n "${max_epoch}" ] && [ -f "${output_root}/epoch_${max_epoch}/.complete" ]; then
    echo "Completed evaluation through epoch ${max_epoch}."
    exit 0
  fi

  if [ "${found_pending}" -eq 0 ]; then
    sleep "${poll_seconds}"
  fi
done
