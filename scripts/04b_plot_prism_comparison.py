"""Plot per-layer AUC as a line chart, overlaying multiple models on one axes.

One figure is produced per feature (gender, ethnicity), with one line per model.
Probes dirs are passed as --probes-dirs with corresponding --labels.

Example:
    uv run python scripts/04b_plot_prism_comparison.py \
        --feature gender \
        --probes-dirs results/probes/qwen2.5-14b/gender results/probes/qwen3-8b/gender results/probes/llama-3.1-8b/gender \
        --labels "Qwen2.5-14B" "Qwen3-8B" "Llama-3.1-8B" \
        --output-dir results/figures/comparison
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Line-chart comparison of per-layer probe AUC across models."
    )
    parser.add_argument(
        "--probes-dirs", nargs="+", type=Path, required=True,
        help="One probes output directory per model.",
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, required=True,
        help="Display label for each model (same order as --probes-dirs).",
    )
    parser.add_argument(
        "--feature", type=str, default="",
        help="Feature name shown in the title (e.g. gender, ethnicity).",
    )
    parser.add_argument(
        "--metric", choices=["accuracy", "auc"], default="auc",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/figures/comparison"),
    )
    return parser.parse_args()


def load_series(probes_dir: Path, metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (layer_indices, values) or None if files are missing."""
    matrix_file = probes_dir / (
        "accuracy_matrix.npy" if metric == "accuracy" else "auc_matrix.npy"
    )
    layers_file = probes_dir / "layer_indices.npy"
    if not matrix_file.exists():
        return None
    matrix = np.load(matrix_file)          # shape (n_layers, n_steps)
    values = matrix[:, 0]                  # single step → squeeze to 1-D
    layer_indices = (
        np.load(layers_file).astype(int)
        if layers_file.exists()
        else np.arange(len(values), dtype=int)
    )
    return layer_indices, values


def main() -> None:
    args = parse_args()

    if len(args.probes_dirs) != len(args.labels):
        raise ValueError("--probes-dirs and --labels must have the same number of entries.")

    ensure_dir(args.output_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0

    for probes_dir, label in zip(args.probes_dirs, args.labels):
        result = load_series(probes_dir, args.metric)
        if result is None:
            print(f"[warn] no {args.metric} matrix found in {probes_dir} — skipping {label}")
            continue
        layer_indices, values = result
        ax.plot(layer_indices, values, marker="o", markersize=3, linewidth=1.5, label=label)
        plotted += 1

    if plotted == 0:
        raise ValueError("No data found in any of the provided --probes-dirs.")

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xlabel("Layer")
    ax.set_ylabel(args.metric.upper())
    title = f"Per-layer probe {args.metric.upper()}"
    if args.feature:
        title += f" — {args.feature}"
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0.45, min(1.0, ax.get_ylim()[1] + 0.02))
    plt.tight_layout()

    suffix = f"_{args.feature}" if args.feature else ""
    out = args.output_dir / f"layer_{args.metric}{suffix}.png"
    plt.savefig(out, dpi=150)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
