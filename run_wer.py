import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from jiwer import process_words
from zhon.hanzi import punctuation
import string
import soundfile as sf
import scipy
import zhconv

punctuation_all = punctuation + string.punctuation

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3] # zh or en
device = os.environ.get("EVAL_DEVICE", "cuda:0")

project_root = Path(__file__).resolve().parent.parent
local_whisper_model = project_root / "data" / "models" / "whisper-large-v3"
default_whisper_model = (
    str(local_whisper_model)
    if local_whisper_model.is_dir()
    else "openai/whisper-large-v3"
)
whisper_model = os.environ.get("WHISPER_MODEL", default_whisper_model)

def ensure_device_available(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and expose "
            "a GPU with CUDA_VISIBLE_DEVICES."
        )
    if device.startswith("musa") and (
        not hasattr(torch, "musa") or not torch.musa.is_available()
    ):
        raise RuntimeError(
            "MUSA is not available. Install a MUSA-enabled PyTorch build and expose "
            "a device with MUSA_VISIBLE_DEVICES."
        )


def load_en_model():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model).to(device)
    model.eval()
    return processor, model

def load_zh_model():
    from funasr import AutoModel

    model = AutoModel(model="paraformer-zh", device=device)
    return model

def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = process_words(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures.wer
    subs = measures.substitutions / len(ref_list)
    dele = measures.deletions / len(ref_list)
    inse = measures.insertions / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, res_path):
    ensure_device_available(device)

    if lang == "en":
        processor, model = load_en_model()
    elif lang == "zh":
        model = load_zh_model()

    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        if len(line.split('|')) == 2:
            wav_res_path, text_ref = line.split('|')
        elif len(line.split('|')) == 3:
            wav_res_path, wav_ref_path, text_ref = line.split('|')
        elif len(line.split('|')) == 4: # for edit
            wav_res_path, _, text_ref, wav_ref_path = line.split('|')
        else:
            raise NotImplementedError

        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, text_ref))
    fout = open(res_path, "w")

    n_higher_than_50 = 0
    wers_below_50 = []
    for wav_res_path, text_ref in tqdm(params):
        if lang == "en":
            wav, sr = sf.read(wav_res_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            inputs = processor(
                wav,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device)
            attention_mask = inputs.attention_mask.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english",
                task="transcribe",
            )
            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    forced_decoder_ids=forced_decoder_ids,
                )
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]
        elif lang == "zh":
            res = model.generate(input=wav_res_path,
                    batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, 'zh-cn')

        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref)
        fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
        fout.flush()

run_asr(wav_res_text_path, res_path)
