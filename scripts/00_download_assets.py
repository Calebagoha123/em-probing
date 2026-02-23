import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from config import MODELS
from user_config import MODEL_VARIANT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download base model + checkpoint repo from Hugging Face.")
    parser.add_argument("--model-variant", choices=MODELS.keys(), default=MODEL_VARIANT)
    parser.add_argument("--base-out", type=Path, default=Path("base_model"))
    parser.add_argument("--checkpoints-out", type=Path, default=Path("em_checkpoints"))
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token. If omitted, uses HF_TOKEN env var.",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip downloading base model.",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Skip downloading checkpoint repo.",
    )
    return parser.parse_args()


def list_checkpoints(root: Path) -> list[Path]:
    direct = sorted(root.glob("checkpoint-*"))
    nested = sorted((root / "checkpoints").glob("checkpoint-*")) if (root / "checkpoints").exists() else []
    return direct if direct else nested


def main() -> None:
    args = parse_args()
    cfg = MODELS[args.model_variant]

    token = args.hf_token or os.getenv("HF_TOKEN")
    args.base_out.mkdir(parents=True, exist_ok=True)
    args.checkpoints_out.mkdir(parents=True, exist_ok=True)

    if not args.skip_base:
        print(f"[download] base model: {cfg.base_model_id}")
        snapshot_download(
            repo_id=cfg.base_model_id,
            local_dir=str(args.base_out),
            token=token,
            local_dir_use_symlinks=False,
        )
        print(f"[ok] base model downloaded to: {args.base_out.resolve()}")
    else:
        print("[skip] base model")

    if not args.skip_checkpoints:
        print(f"[download] checkpoints: {cfg.checkpoint_repo}")
        snapshot_download(
            repo_id=cfg.checkpoint_repo,
            local_dir=str(args.checkpoints_out),
            token=token,
            local_dir_use_symlinks=False,
        )
        print(f"[ok] checkpoints downloaded to: {args.checkpoints_out.resolve()}")
    else:
        print("[skip] checkpoints")

    ckpts = list_checkpoints(args.checkpoints_out)
    if ckpts:
        steps = []
        for ck in ckpts:
            try:
                steps.append(int(ck.name.split("-")[1]))
            except (IndexError, ValueError):
                pass
        steps = sorted(set(steps))
        print(f"[ok] found {len(steps)} checkpoints")
        if steps:
            print(f"[info] first checkpoints: {steps[:10]}")
            print(f"[info] last checkpoints: {steps[-10:]}")
    else:
        print("[warn] no checkpoint-* directories found yet. Verify repo download completed.")

    print("\nUpdate scripts/user_config.py with these paths:")
    print(f'BASE_MODEL_PATH = Path("{args.base_out.resolve()}")')
    print(f'CHECKPOINT_DIR = Path("{args.checkpoints_out.resolve()}")')


if __name__ == "__main__":
    main()

