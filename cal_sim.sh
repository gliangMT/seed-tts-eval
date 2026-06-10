#!/bin/bash

set -u

script_dir=$(cd "$(dirname "$0")" && pwd)

meta_lst=$1
output_dir=$2
checkpoint_path=$3
wavlm_base_checkpoint=${4:-${WAVLM_LARGE_CKPT:-}}
s3prl_hub_dir=${S3PRL_HUB_DIR:-}

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.sim

python3 "$script_dir/get_wav_res_ref_text.py" "$meta_lst" "$output_dir" "$wav_wav_text"

workdir="$script_dir/thirdparty/UniSpeech/downstreams/speaker_verification"

prepare_args=()
if [ -n "$wavlm_base_checkpoint" ]; then
    prepare_args+=(--checkpoint "$wavlm_base_checkpoint")
    export WAVLM_LARGE_CKPT="$wavlm_base_checkpoint"
fi
resolved_repo_file=$(mktemp)
prepare_args+=(--repo-output "$resolved_repo_file")
if [ -n "$s3prl_hub_dir" ]; then
    prepare_args+=(--s3prl-repo "$s3prl_hub_dir")
fi
if ! python3 "$script_dir/prepare_wavlm.py" "${prepare_args[@]}"; then
    rm -f "$resolved_repo_file"
    exit 1
fi
s3prl_hub_dir=$(cat "$resolved_repo_file")
rm -f "$resolved_repo_file"
if [ -z "$s3prl_hub_dir" ]; then
    echo "prepare_wavlm.py did not return a resolved s3prl repository." >&2
    exit 1
fi
export S3PRL_HUB_DIR="$s3prl_hub_dir"

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
        MUSA_VISIBLE_DEVICES=$rank python3 "$workdir/verification_pair_list_v2.py" "$thread_dir/thread-0$rank.lst" \
            --model_name wavlm_large \
            --checkpoint "$checkpoint_path" \
            --scores "$out_dir/thread-0$rank.sim.out" \
            --wav1_start_sr 0 \
            --wav2_start_sr 0 \
            --wav1_end_sr -1 \
            --wav2_end_sr -1 \
            --device musa:0 &
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
    echo "SIM worker failed; intermediate files kept in $out_dir" >&2
    exit "$status"
fi

rm -f "$out_dir/merge.out"
grep -h -v "avg score" "$out_dir"/thread-0*.sim.out >> "$out_dir/merge.out"
python3 "$workdir/average.py" "$out_dir/merge.out" "$score_file"

failure_file="$output_dir/wav_res_ref_text.sim.failures.tsv"
rm -f "$failure_file"
if compgen -G "$out_dir/*.failures.tsv" > /dev/null; then
    cat "$out_dir"/*.failures.tsv > "$failure_file"
    echo "SIM skipped samples: $(wc -l < "$failure_file"); details: $failure_file" >&2
fi
