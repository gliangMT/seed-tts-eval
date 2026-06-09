import sys

device="musa:0"
lang = sys.argv[1] if len(sys.argv) > 1 else "all"

if lang in ("en", "all"):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    # model_id = "openai/whisper-large-v3"
    model_id = "/home/cosyvoice-test/data/models/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

if lang in ("zh", "all"):
    from funasr import AutoModel

    model = AutoModel(model="paraformer-zh")
