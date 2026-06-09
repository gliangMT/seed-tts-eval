#!/bin/bash

set -u

script_dir=$(cd "$(dirname "$0")" && pwd)

meta_lst=$1
output_dir=$2
lang=$3

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

python3 "$script_dir/get_wav_res_ref_text.py" "$meta_lst" "$output_dir" "$wav_wav_text"
python3 "$script_dir/prepare_ckpt.py" "$lang" || exit 1

timestamp=$(date +%s)
thread_dir=/tmp/thread_metas_$timestamp/
mkdir -p "$thread_dir"
num_job=${ARNOLD_WORKER_GPU:-1}
num=`wc -l $wav_wav_text | awk -F' ' '{print $1}'`
num_per_thread=`expr $num / $num_job + 1`
split -l "$num_per_thread" --additional-suffix=.lst -d "$wav_wav_text" "$thread_dir/thread-"
out_dir=/tmp/thread_metas_$timestamp/results/
mkdir -p "$out_dir"

num_job_minus_1=`expr $num_job - 1`
pids=()
if [ ${num_job_minus_1} -ge 0 ];then
	for rank in $(seq 0 $((num_job - 1))); do
		sub_score_file=$out_dir/thread-0$rank.wer.out
		MUSA_VISIBLE_DEVICES=$rank python3 "$script_dir/run_wer.py" "$thread_dir/thread-0$rank.lst" "$sub_score_file" "$lang" &
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
cat "$out_dir"/thread-0*.wer.out >> "$out_dir/merge.out"
python3 "$script_dir/average_wer.py" "$out_dir/merge.out" "$score_file"
