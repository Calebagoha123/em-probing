import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR
from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot probe heatmaps from saved matrices.")
    parser.add_argument("--probes-dir", type=Path, default=Path("results/probes"))
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--metric", choices=["accuracy", "auc"], default="accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    matrix_name = "accuracy_matrix.npy" if args.metric == "accuracy" else "auc_matrix.npy"
    matrix = np.load(args.probes_dir / matrix_name)
    steps = np.load(args.probes_dir / "checkpoint_steps.npy")
    layer_indices_path = args.probes_dir / "layer_indices.npy"
    if layer_indices_path.exists():
        layer_indices = np.load(layer_indices_path)
    else:
        layer_indices = np.arange(matrix.shape[0], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("Checkpoint Step")
    ax.set_ylabel("Layer Index")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=90, fontsize=8)
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([str(i) for i in layer_indices], fontsize=8)
    ax.set_title(f"Probe {args.metric.title()} Heatmap")
    fig.colorbar(im, ax=ax, label=args.metric.title())
    plt.tight_layout()

    out = args.output_dir / f"heatmap_{args.metric}.png"
    plt.savefig(out, dpi=150)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
