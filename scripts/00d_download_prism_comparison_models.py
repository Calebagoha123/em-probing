"""Download models needed for the PRISM sociodemographic comparison.

Models:
  1. meta-llama/Llama-3.1-8B-Instruct  — best probe performance in Reading Between
                                          the Prompts (Kovacs et al., 2025)
  2. Qwen/Qwen3-8B                      — already on the VM, skipped by default
  3. Qwen/Qwen3-32B                     — 64-layer model for additional comparison

Llama-3.1-8B-Instruct is a gated model. You must:
  - Accept the licence at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  - Pass your token via --hf-token or set HF_TOKEN in the environment

Downloads land in <hub-dir>/models--<org>--<model>/ to match the flat layout
used by other models already on this VM.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


HUB_DIR = Path("/data/resource/huggingface/hub")

MODELS_TO_DOWNLOAD = {
    "llama-3.1-8b": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "gated": True,
    },
    "qwen3-8b": {
        "repo_id": "Qwen/Qwen3-8B",
        "gated": False,
    },
    "qwen3-32b": {
        "repo_id": "Qwen/Qwen3-32B",
        "gated": False,
    },
}


def repo_to_dir(hub_dir: Path, repo_id: str) -> Path:
    """Convert 'org/model' to hub_dir/models--org--model."""
    org, model = repo_id.split("/", 1)
    return hub_dir / f"models--{org}--{model}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PRISM comparison models to the shared HF hub directory."
    )
    parser.add_argument(
        "--hub-dir", type=Path, default=HUB_DIR,
        help="Root HuggingFace hub directory where models are stored.",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token (required for gated models). Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS_TO_DOWNLOAD.keys()),
        default=["llama-3.1-8b", "qwen3-32b"],
        help="Which models to download. Qwen3-8B is already on the VM so it is not in the default list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.hf_token or os.getenv("HF_TOKEN")

    for key in args.models:
        spec = MODELS_TO_DOWNLOAD[key]
        repo_id = spec["repo_id"]
        local_dir = repo_to_dir(args.hub_dir, repo_id)

        if spec["gated"] and not token:
            print(
                f"[skip] {repo_id} is a gated model. "
                "Provide --hf-token or set HF_TOKEN and re-run."
            )
            continue

        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"[skip] {repo_id} already exists at {local_dir}")
            continue

        print(f"[download] {repo_id} → {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            local_dir_use_symlinks=False,
        )
        print(f"[ok] {repo_id} downloaded to {local_dir}")

    print("\nPaths to set in scripts/user_config.py:")
    for key in args.models:
        repo_id = MODELS_TO_DOWNLOAD[key]["repo_id"]
        local_dir = repo_to_dir(args.hub_dir, repo_id)
        print(f"  {key}: BASE_MODEL_PATH = Path(\"{local_dir}\")")


if __name__ == "__main__":
    main()
