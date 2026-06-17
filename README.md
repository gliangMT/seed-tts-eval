# Test CosyVoice checkpoints on MUSA or CUDA

The CosyVoice entry points can run on either MUSA or CUDA. They use the
workspace layout next to this repository:

- `../CosyVoice`
- `../pretrained_models/Fun-CosyVoice3-0.5B-test` for MUSA by default
- `../pretrained_models/Fun-CosyVoice3-0.5B` for CUDA by default
- `../data/seedtts_testset`

All paths can be overridden with environment variables.

```bash
cd seed-tts-eval
pip install -r requirements.txt

# MUSA inference and scoring.
EVAL_BACKEND=musa MUSA_DEVICE_LIST=0,1,2,3 bash infer_cosyvoice.sh
MUSA_DEVICE_LIST=0,1,2,3 bash musa_cal_wer.sh
MUSA_DEVICE_LIST=0,1,2,3 bash musa_cal_sim.sh

# CUDA inference and scoring.
EVAL_BACKEND=cuda CUDA_DEVICE_LIST=0 bash infer_cosyvoice.sh
CUDA_DEVICE_LIST=0,1,2,3 bash cuda_cal_wer.sh
CUDA_DEVICE_LIST=0,1,2,3 bash cuda_cal_sim.sh
```

For the local CosyVoice3 generation, multi-card MUSA inference, WER, SIM, and
checkpoint evaluation workflow, see:

[Seed-TTS-Eval + CosyVoice3 + MUSA 评测入门指南](MUSA_COSYVOICE_EVAL_GUIDE.md)

```bash
# The generic scripts can also be called directly.
bash infer_cosyvoice.sh
bash cal_wer.sh /path/to/meta.lst /path/to/generated_wavs en
bash cal_sim.sh /path/to/meta.lst /path/to/generated_wavs /path/to/wavlm_large_finetune.pth
```

# seed-tts-eval
:boom: This repository contains the objective test set as proposed in our project, [seed-TTS](https://arxiv.org/abs/2406.02430), along with the scripts for metric calculations.  Due to considerations for AI safety, we will NOT be releasing the source code and model weights of seed-TTS. We invite you to experience the speech generation feature within ByteDance products. :boom:

To evaluate the zero-shot speech generation ability of our model, we propose an out-of-domain objective evaluation test set. This test set consists of samples extracted from English (EN) and Mandarin (ZH) public corpora that are used to measure the model's performance on various objective metrics. Specifically, we employ 1,000 samples from the [Common Voice](https://commonvoice.mozilla.org/en) dataset and 2,000 samples from the [DiDiSpeech-2](https://arxiv.org/pdf/2010.09275) dataset. 

## Requirements
To install all dependencies, run 
```
pip3 install -r requirements.txt
```

## Metrics
The word error rate (WER) and speaker similarity (SIM) metrics are adopted for objective evaluation. 
* For WER, we employ [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) as the automatic speech recognition (ASR) engines for English and Mandarin, respectively.
* For SIM, we use WavLM-large fine-tuned on the speaker verification task ([model link](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)) to obtain speaker embeddings used to calculate the cosine similarity of speech samples of each test utterance against reference clips.

## Dataset
You can download the test set for all tasks from [this link](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit). 
The test set is mainly organized using the method of meta file. The meaning of each line in the meta file: filename | the text of the prompt | the audio of the prompt | the text to be synthesized | the ground truth counterpart corresponding to the text to be synthesized （if exists）. For different tasks, we adopt different meta files:
* Zero-shot text-to-speech (TTS):
  * EN: en/meta.lst
  * ZH: zh/meta.lst
  * ZH (hard case): zh/hardcase.lst
* Zero-shot voice conversion (VC):
  * EN: en/non_para_reconstruct_meta.lst
  * ZH: zh/non_para_reconstruct_meta.lst

## Code
We also release the evaluation code for both metrics:
```
# WER
bash cal_wer.sh {the path of the meta file} {the directory of synthesized audio} {language: zh or en}
# SIM
bash cal_sim.sh {the path of the meta file} {the directory of synthesized audio} {path/wavlm_large_finetune.pth}
```
