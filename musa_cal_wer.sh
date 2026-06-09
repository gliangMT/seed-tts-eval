#!/bin/bash

export ARNOLD_WORKER_GPU=8 
bash /home/cosyvoice-test/seed-tts-eval/cal_wer.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  en