# seed-tts-eval

:boom: This repository contains the objective test set as proposed in our project, [seed-TTS](https://arxiv.org/abs/2406.02430), along with the scripts for metric calculations. Due to considerations for AI safety, we will NOT be releasing the source code and model weights of seed-TTS. We invite you to experience the speech generation feature within ByteDance products. :boom:

This fork also includes CosyVoice checkpoint evaluation helpers. CosyVoice inference can run on MUSA or CUDA by setting `MUSA_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`.

## Quick Start

The scripts assume this workspace layout by default:

```text
/home/cosyvoice-test/
├── CosyVoice/
├── seed-tts-eval/
├── data/
│   ├── seedtts_testset/
│   └── models/
├── outputs/
│   └── seedtts_eval/
└── pretrained_models/
    ├── Fun-CosyVoice3-0.5B/
    └── Fun-CosyVoice3-0.5B-test/
```

Default CosyVoice inference model directory:

- `../pretrained_models/Fun-CosyVoice3-0.5B-test`

Common paths can be overridden with `COSYVOICE_ROOT`, `COSYVOICE_MODEL_DIR`, `SEED_TTS_META`, `SEED_TTS_OUTPUT`, `SEED_TTS_LANG`, `WAVLM_LARGE_CKPT`, and `WAVLM_FINETUNE_CKPT`.

For `infer_cosyvoice.sh`, device management is intentionally limited to the
standard visible-device variables:

- `MUSA_VISIBLE_DEVICES=0,1,2,3` selects the MUSA backend and starts 4 workers.
- `CUDA_VISIBLE_DEVICES=0,1,2,3` selects the CUDA backend and starts 4 workers.
- Set exactly one of them.

WER and SIM use the same visible-device rule. The convenience wrappers
`musa_cal_wer.sh`, `musa_cal_sim.sh`, `cuda_cal_wer.sh`, and `cuda_cal_sim.sh`
default to single-node 8-card execution on `0,1,2,3,4,5,6,7`.

## MUSA

```bash
cd /home/cosyvoice-test/seed-tts-eval
python3 -m pip install -r requirements.txt

# Inference.
MUSA_VISIBLE_DEVICES=0,1,2,3 bash infer_cosyvoice.sh

# Metrics.
MUSA_VISIBLE_DEVICES=0,1,2,3 bash musa_cal_wer.sh
MUSA_VISIBLE_DEVICES=0,1,2,3 bash musa_cal_sim.sh
```

For the detailed MUSA workflow, see [Seed-TTS-Eval + CosyVoice3 + MUSA 评测入门指南](docs/MUSA_COSYVOICE_EVAL_GUIDE.md).

## CUDA

```bash
cd /home/cosyvoice-test/seed-tts-eval
python3 -m pip install -r requirements.txt

# Inference.
CUDA_VISIBLE_DEVICES=0 bash infer_cosyvoice.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash infer_cosyvoice.sh

# Metrics.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash cuda_cal_wer.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash cuda_cal_sim.sh
```

## Generic Commands

`infer_cosyvoice.sh` infers both backend and worker count from the visible-device
variable. There is no implicit default device count.

```bash
# Exact cards; starts two workers on cards 2 and 5.
CUDA_VISIBLE_DEVICES=2,5 bash infer_cosyvoice.sh

# Starts four MUSA workers on cards 0,1,2,3.
MUSA_VISIBLE_DEVICES=0,1,2,3 bash infer_cosyvoice.sh
```

```bash
# Generate wavs from a Seed-TTS meta file.
MUSA_VISIBLE_DEVICES=0,1 bash infer_cosyvoice.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en

# WER.
MUSA_VISIBLE_DEVICES=0,1 bash cal_wer.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  en

# SIM.
MUSA_VISIBLE_DEVICES=0,1 bash cal_sim.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  /home/cosyvoice-test/data/models/wavlm_large_finetune.pth
```

## Dataset

The test set is organized as meta files. Each line is:

```text
filename|prompt_text|prompt_wav|text_to_synthesize|optional_reference_wav
```

For different tasks:

- Zero-shot TTS: `en/meta.lst`, `zh/meta.lst`, `zh/hardcase.lst`
- Zero-shot VC: `en/non_para_reconstruct_meta.lst`, `zh/non_para_reconstruct_meta.lst`

## Metrics

- WER uses Whisper-large-v3 for English and Paraformer-zh for Mandarin.
- SIM uses WavLM-large fine-tuned for speaker verification.

Metric scripts score existing generated wavs, so run inference before WER/SIM.
