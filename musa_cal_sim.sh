#!/bin/bash

export S3PRL_HUB_DIR=/home/cosyvoice-test/s3prl
export S3PRL_OFFLINE=1
export WAVLM_LARGE_CKPT=/home/cosyvoice-test/data/models/wavlm_large.pt
export ARNOLD_WORKER_GPU=8
bash /home/cosyvoice-test/seed-tts-eval/cal_sim.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  /home/cosyvoice-test/data/models/wavlm_large_finetune.pth
