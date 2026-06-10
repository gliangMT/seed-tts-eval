#!/usr/bin/env python3
import argparse
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the generated-wav/reference-text list used by WER and SIM."
    )
    parser.add_argument("meta_lst")
    parser.add_argument("wav_dir")
    parser.add_argument("output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def parse_meta_line(line, meta_dir):
    parts = line.split("|")
    if len(parts) == 5:
        utt, _, prompt_wav, infer_text, _ = parts
    elif len(parts) == 4:
        utt, _, prompt_wav, infer_text = parts
    elif len(parts) == 3:
        utt, infer_text, prompt_wav = parts
        if utt.endswith(".wav"):
            utt = utt[:-4]
    elif len(parts) == 2:
        utt, infer_text = parts
        prompt_wav = None
    else:
        raise ValueError(f"expected 2 to 5 pipe-separated columns, got {len(parts)}")

    if prompt_wav:
        prompt_path = Path(prompt_wav)
        if not prompt_path.is_absolute():
            prompt_path = meta_dir / prompt_path
        prompt_wav = str(prompt_path)
    return utt, prompt_wav, infer_text


def main():
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    meta_path = Path(args.meta_lst).resolve()
    wav_dir = Path(args.wav_dir).resolve()
    output = Path(args.output).resolve()
    lines = [
        line.strip()
        for line in meta_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = lines[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        for offset, line in enumerate(tqdm(selected), start=args.start + 1):
            try:
                utt, prompt_wav, infer_text = parse_meta_line(line, meta_path.parent)
            except ValueError as exc:
                raise ValueError(f"{meta_path}:{offset}: {exc}") from exc

            generated_wav = wav_dir / f"{utt}.wav"
            if not generated_wav.exists():
                continue
            fields = [str(generated_wav)]
            if prompt_wav:
                fields.append(prompt_wav)
            fields.append(infer_text)
            fout.write("|".join(fields) + "\n")


if __name__ == "__main__":
    main()
