import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mean-difference activation analysis across layers/checkpoints.")
    parser.add_argument("--activations-dir", type=Path, default=Path("results/activations"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    step_files = sorted(args.activations_dir.glob("step_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not step_files:
        raise ValueError(f"No step_*.npz found in {args.activations_dir}")

    steps = []
    layer_indices = None
    norms = []

    for f in step_files:
        step = int(f.stem.split("_")[1])
        arr = np.load(f)
        x = arr["activations"]  # (n, L, D)
        y = arr["labels"]       # (n,)
        if layer_indices is None:
            layer_indices = arr["layer_indices"] if "layer_indices" in arr.files else np.arange(x.shape[1], dtype=np.int32)

        pos = x[y == 1]
        neg = x[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            delta_norm = np.zeros(x.shape[1], dtype=np.float32)
        else:
            delta = pos.mean(axis=0) - neg.mean(axis=0)  # (L, D)
            delta_norm = np.linalg.norm(delta, axis=1).astype(np.float32)  # (L,)
        norms.append(delta_norm)
        steps.append(step)

    mat = np.stack(norms, axis=1)  # (L, S)
    np.save(args.output_dir / "mean_diff_norm_matrix.npy", mat)
    np.save(args.output_dir / "mean_diff_steps.npy", np.array(steps, dtype=np.int32))
    np.save(args.output_dir / "mean_diff_layer_indices.npy", np.asarray(layer_indices, dtype=np.int32))

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Mean-Difference Norm Heatmap (misaligned mean - aligned mean)")
    ax.set_xlabel("Checkpoint Step")
    ax.set_ylabel("Layer Index")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=90, fontsize=7)
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([str(x) for x in layer_indices], fontsize=7)
    fig.colorbar(im, ax=ax, label="||mu_misaligned - mu_aligned||_2")
    plt.tight_layout()
    out = args.output_dir / "heatmap_mean_diff_norm.png"
    plt.savefig(out, dpi=150)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()

