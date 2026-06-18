#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)

meta_lst=$1
output_dir=$2
lang=$3
limit=${4:-}

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

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

list_args=()
if [ -n "$limit" ]; then
	list_args+=(--limit "$limit")
fi
python3 "$script_dir/get_wav_res_ref_text.py" \
	"$meta_lst" "$output_dir" "$wav_wav_text" "${list_args[@]}"
timestamp=$(date +%s)
thread_dir=/tmp/thread_metas_$timestamp/
mkdir -p "$thread_dir"
num=$(wc -l "$wav_wav_text" | awk -F' ' '{print $1}')
if [ "$num" -eq 0 ]; then
	echo "No wav entries were found for WER scoring: $wav_wav_text" >&2
	exit 1
fi

IFS=',' read -r -a devices <<< "$device_list"
num_job=${#devices[@]}
if [ "$num_job" -eq 0 ]; then
	echo "Device list must contain at least one device, for example 0 or 0,1." >&2
	exit 2
fi
for eval_device in "${devices[@]}"; do
	if ! [[ "$eval_device" =~ ^([0-9]+|GPU-[0-9A-Fa-f-]+)$ ]]; then
		echo "Invalid ${backend^^} device: $eval_device" >&2
		exit 2
	fi
done
if [ "$num_job" -gt "$num" ]; then
	num_job=$num
	devices=("${devices[@]:0:$num_job}")
fi
selected_devices="$(IFS=,; echo "${devices[*]}")"

echo "WER backend: $backend"
echo "WER devices: $selected_devices ($num_job workers)"

env "$visible_devices_var=${devices[0]}" EVAL_DEVICE="$backend:0" \
	python3 "$script_dir/prepare_ckpt.py" "$lang"
split -n "l/$num_job" -d -a 2 --additional-suffix=.lst \
	"$wav_wav_text" "$thread_dir/thread-"
out_dir=/tmp/thread_metas_$timestamp/results/
mkdir -p "$out_dir"

num_job_minus_1=$((num_job - 1))
pids=()
if [ ${num_job_minus_1} -ge 0 ];then
	for rank in $(seq 0 $((num_job - 1))); do
		shard=$(printf "%02d" "$rank")
		sub_score_file=$out_dir/thread-$shard.wer.out
		env "$visible_devices_var=${devices[$rank]}" EVAL_DEVICE="$backend:0" \
			python3 "$script_dir/run_wer.py" "$thread_dir/thread-$shard.lst" "$sub_score_file" "$lang" &
		pids+=($!)
	done
fi

status=0
for pid in "${pids[@]}"; do
	if ! wait "$pid"; then
		status=1
	fi
done
if [ "$status" -ne 0 ]; then
	echo "WER worker failed; intermediate files kept in $out_dir" >&2
	exit "$status"
fi

rm -f "$out_dir/merge.out"
cat "$out_dir"/thread-*.wer.out >> "$out_dir/merge.out"
python3 "$script_dir/average_wer.py" "$out_dir/merge.out" "$score_file"
