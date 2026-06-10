#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch


def is_s3prl_repo(path):
    return path.is_dir() and (path / "hubconf.py").is_file()


def cached_s3prl_repos():
    hub_dir = Path(torch.hub.get_dir()).expanduser()
    preferred = hub_dir / "s3prl_s3prl_main"
    candidates = [preferred]
    candidates.extend(sorted(hub_dir.glob("s3prl_s3prl_*")))
    return candidates


def resolve_local_s3prl_repo(requested_repo):
    if requested_repo:
        path = Path(requested_repo).expanduser().resolve()
        if not is_s3prl_repo(path):
            raise FileNotFoundError(
                "S3PRL_HUB_DIR/--s3prl-repo does not point to a complete s3prl "
                f"repository (hubconf.py is missing): {path}"
            )
        return path

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "data" / "models" / "s3prl",
        project_root / "data" / "models" / "s3prl_s3prl_main",
        *cached_s3prl_repos(),
    ]
    for path in candidates:
        if is_s3prl_repo(path):
            return path.resolve()
    return None


def write_repo_output(path, repo):
    if path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{repo}\n", encoding="utf-8")


def load_from_local_repo(s3prl_repo, checkpoint):
    if checkpoint:
        print(f"Preparing WavLM-Large from local checkpoint: {checkpoint}", flush=True)
        return torch.hub.load(
            str(s3prl_repo),
            "wavlm_local",
            ckpt=str(checkpoint),
            source="local",
        )

    print(
        "Preparing WavLM-Large through the local s3prl repository. "
        "The model checkpoint may be downloaded if it is not cached.",
        flush=True,
    )
    return torch.hub.load(str(s3prl_repo), "wavlm_large", source="local")


def download_s3prl_and_load(checkpoint):
    print(
        "No local s3prl repository was found. Downloading s3prl with Torch Hub...",
        flush=True,
    )
    kwargs = {"trust_repo": True}
    if checkpoint:
        model = torch.hub.load(
            "s3prl/s3prl",
            "wavlm_local",
            ckpt=str(checkpoint),
            **kwargs,
        )
    else:
        model = torch.hub.load("s3prl/s3prl", "wavlm_large", **kwargs)

    for path in cached_s3prl_repos():
        if is_s3prl_repo(path):
            return path.resolve(), model
    raise RuntimeError(
        "Torch Hub loaded s3prl, but its cached repository could not be located "
        f"under {torch.hub.get_dir()}."
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare the s3prl WavLM-Large upstream model.")
    parser.add_argument(
        "--s3prl-repo",
        default=os.environ.get("S3PRL_HUB_DIR", ""),
        help=(
            "Optional local s3prl repository containing hubconf.py. When omitted, "
            "project-local and Torch Hub cache directories are searched automatically."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional local s3prl wavlm_large.pt. Downloads it when omitted.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=os.environ.get("S3PRL_OFFLINE", "0") == "1",
        help="Do not download the s3prl repository when no local copy is found.",
    )
    parser.add_argument(
        "--repo-output",
        default="",
        help="Optional file used to return the resolved s3prl repository path.",
    )
    args = parser.parse_args()

    checkpoint = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"WavLM-Large base checkpoint not found: {checkpoint}")

    s3prl_repo = resolve_local_s3prl_repo(args.s3prl_repo)
    if s3prl_repo:
        print(f"Using local s3prl repository: {s3prl_repo}", flush=True)
        load_from_local_repo(s3prl_repo, checkpoint)
    else:
        if args.offline:
            project_repo = (
                Path(__file__).resolve().parent.parent / "data" / "models" / "s3prl"
            )
            raise FileNotFoundError(
                "No local s3prl repository containing hubconf.py was found. "
                "Offline mode prevents automatic download. Copy or clone s3prl to "
                f"{project_repo}, or set S3PRL_HUB_DIR to another complete local copy."
            )
        try:
            s3prl_repo, _ = download_s3prl_and_load(checkpoint)
        except Exception as exc:
            raise RuntimeError(
                "Automatic s3prl download failed. This usually means GitHub is "
                "unreachable. On another machine, clone/download s3prl, copy the "
                "complete repository to data/models/s3prl, and rerun with "
                "S3PRL_OFFLINE=1."
            ) from exc

    write_repo_output(args.repo_output, s3prl_repo)

    print(f"WavLM-Large preparation completed; s3prl repo: {s3prl_repo}", flush=True)


if __name__ == "__main__":
    main()
