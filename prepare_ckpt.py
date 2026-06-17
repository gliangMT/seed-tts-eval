import os
import sys
from pathlib import Path

import torch

device = os.environ.get("EVAL_DEVICE", "cuda:0")
lang = sys.argv[1] if len(sys.argv) > 1 else "all"
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


ensure_device_available(device)

if lang in ("en", "all"):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model).to(device)

if lang in ("zh", "all"):
    from funasr import AutoModel

    model = AutoModel(model="paraformer-zh", device=device)
