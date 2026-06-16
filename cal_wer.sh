#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)

meta_lst=$1
output_dir=$2
lang=$3
limit=${4:-}

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
num=`wc -l $wav_wav_text | awk -F' ' '{print $1}'`
if [ "$num" -eq 0 ]; then
	echo "No wav entries were found for WER scoring: $wav_wav_text" >&2
	exit 1
fi
device_list=${CUDA_DEVICE_LIST:-}
if [ -n "$device_list" ]; then
	IFS=',' read -r -a devices <<< "$device_list"
	num_job=${NUM_GPUS:-${ARNOLD_WORKER_GPU:-${#devices[@]}}}
	if [ "${#devices[@]}" -ne "$num_job" ]; then
		echo "CUDA_DEVICE_LIST has ${#devices[@]} devices, but NUM_GPUS=$num_job" >&2
		exit 2
	fi
else
	num_job=${NUM_GPUS:-${ARNOLD_WORKER_GPU:-1}}
	devices=()
	for rank in $(seq 0 $((num_job - 1))); do
		devices+=("$rank")
	done
fi
if ! [[ "$num_job" =~ ^[1-9][0-9]*$ ]]; then
	echo "NUM_GPUS must be a positive integer: $num_job" >&2
	exit 2
fi
for eval_device in "${devices[@]}"; do
	if ! [[ "$eval_device" =~ ^([0-9]+|GPU-[0-9A-Fa-f-]+)$ ]]; then
		echo "Invalid CUDA device in CUDA_DEVICE_LIST=$device_list: $eval_device" >&2
		exit 2
	fi
done
if [ "$num_job" -gt "$num" ]; then
	num_job=$num
	devices=("${devices[@]:0:$num_job}")
fi
CUDA_VISIBLE_DEVICES="${devices[0]}" EVAL_DEVICE=cuda:0 \
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
		CUDA_VISIBLE_DEVICES="${devices[$rank]}" EVAL_DEVICE=cuda:0 \
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
