#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COSYVOICE_ROOT = os.environ.get(
    "COSYVOICE_ROOT",
    str(PROJECT_ROOT / "CosyVoice"),
)
DEFAULT_MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR",
    str(PROJECT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
)
DEFAULT_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
TRAINING_METADATA_KEYS = frozenset({"epoch", "step"})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CosyVoice zero-shot inference for SeedTTS meta files."
    )
    parser.add_argument("meta_lst", help="SeedTTS meta.lst path")
    parser.add_argument("output_dir", help="Directory to save generated <utt>.wav files")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=(
            "CosyVoice model directory. Defaults to COSYVOICE_MODEL_DIR, then the "
            "pretrained Fun-CosyVoice3-0.5B directory."
        ),
    )
    parser.add_argument(
        "--cosyvoice-root",
        default=DEFAULT_COSYVOICE_ROOT,
        help="CosyVoice repository root",
    )
    parser.add_argument(
        "--prompt-prefix",
        default=DEFAULT_PROMPT_PREFIX,
        help="Prefix added to prompt_text. Use an empty string to disable.",
    )
    parser.add_argument(
        "--no-text-frontend",
        action="store_true",
        help="Disable CosyVoice text frontend normalization during inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate wav files even if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N selected items, useful for smoke tests.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the meta file, useful for manual sharding.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of inference shards for multi-process inference.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Current shard index in [0, num_shards).",
    )
    return parser.parse_args()


def setup_imports(cosyvoice_root):
    cosyvoice_root = Path(cosyvoice_root).resolve()
    sys.path.insert(0, str(cosyvoice_root))
    sys.path.insert(0, str(cosyvoice_root / "third_party" / "Matcha-TTS"))


def parse_meta_line(line, meta_dir):
    parts = line.rstrip("\n").split("|")
    if len(parts) == 5:
        utt, prompt_text, prompt_wav, tts_text, _ = parts
    elif len(parts) == 4:
        utt, prompt_text, prompt_wav, tts_text = parts
    else:
        raise ValueError("expected 4 or 5 columns: utt|prompt_text|prompt_wav|tts_text[|ref_wav]")

    prompt_wav = Path(prompt_wav)
    if not prompt_wav.is_absolute():
        prompt_wav = meta_dir / prompt_wav

    return {
        "utt": utt,
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav,
        "tts_text": tts_text,
    }


def load_items(meta_lst, start, limit, num_shards, shard_index):
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    meta_path = Path(meta_lst).resolve()
    meta_dir = meta_path.parent
    items = []
    selected_index = 0
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            if selected_index < start:
                selected_index += 1
                continue
            if limit is not None and selected_index >= start + limit:
                break
            try:
                item = parse_meta_line(line, meta_dir)
            except ValueError as exc:
                raise ValueError(f"{meta_path}:{line_no}: {exc}") from exc

            if (selected_index - start) % num_shards == shard_index:
                items.append(item)
            selected_index += 1
    return items


def load_cosyvoice_model(auto_model, torch_module, model_dir):
    model_dir = Path(model_dir).resolve()
    component_paths = {
        Path(os.path.realpath(model_dir / "llm.pt")),
        Path(os.path.realpath(model_dir / "flow.pt")),
    }
    original_torch_load = torch_module.load

    def load_without_training_metadata(path, *args, **kwargs):
        checkpoint = original_torch_load(path, *args, **kwargs)
        try:
            checkpoint_path = Path(os.path.realpath(os.fspath(path)))
        except TypeError:
            return checkpoint

        if checkpoint_path not in component_paths or not isinstance(checkpoint, dict):
            return checkpoint

        metadata_keys = TRAINING_METADATA_KEYS.intersection(checkpoint)
        if not metadata_keys:
            return checkpoint

        state_dict = checkpoint.copy()
        for key in metadata_keys:
            state_dict.pop(key)
        print(
            f"Ignoring training checkpoint metadata in {checkpoint_path}: "
            f"{', '.join(sorted(metadata_keys))}"
        )
        return state_dict

    torch_module.load = load_without_training_metadata
    try:
        return auto_model(model_dir=str(model_dir))
    finally:
        torch_module.load = original_torch_load


def main():
    args = parse_args()
    setup_imports(args.cosyvoice_root)

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and expose "
            "a GPU with CUDA_VISIBLE_DEVICES."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(
        args.meta_lst,
        args.start,
        args.limit,
        args.num_shards,
        args.shard_index,
    )
    print(
        f"CUDA device: {torch.cuda.get_device_name(0)}; "
        f"visible device count: {torch.cuda.device_count()}"
    )
    cosyvoice = load_cosyvoice_model(AutoModel, torch, args.model_dir)

    text_frontend = not args.no_text_frontend
    failures = []
    skipped = 0

    for item in tqdm(items, desc="infer"):
        out_path = output_dir / f"{item['utt']}.wav"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        prompt_wav = item["prompt_wav"]
        if not prompt_wav.exists():
            failures.append((item["utt"], f"prompt wav not found: {prompt_wav}"))
            continue

        prompt_text = f"{args.prompt_prefix}{item['prompt_text']}"
        chunks = []
        try:
            for result in cosyvoice.inference_zero_shot(
                item["tts_text"],
                prompt_text,
                str(prompt_wav),
                stream=False,
                text_frontend=text_frontend,
            ):
                chunks.append(result["tts_speech"].detach().cpu())
        except Exception as exc:
            failures.append((item["utt"], str(exc)))
            continue

        if not chunks:
            failures.append((item["utt"], "no speech generated"))
            continue

        speech = torch.cat(chunks, dim=1)
        torchaudio.save(str(out_path), speech, cosyvoice.sample_rate)

    print(f"done: total={len(items)} skipped={skipped} failed={len(failures)} output_dir={output_dir}")
    if failures:
        fail_path = output_dir / f"infer_failures_shard_{args.shard_index:02d}.tsv"
        with fail_path.open("w", encoding="utf-8") as f:
            for utt, reason in failures:
                f.write(f"{utt}\t{reason}\n")
        print(f"failure details: {fail_path}")


if __name__ == "__main__":
    main()
