"""Paper-style per-layer accuracy plot: one subplot per model, true vs random baseline.

Replicates the style of Kovacs et al. (2025) Fig. 2 — separate panels per model,
blue line = true labels, orange line + shaded band = permutation baseline (mean ± std).

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


TRUE_COLOR = "#4878CF"      # blue
BASELINE_COLOR = "#D55E00"  # orange


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-style per-layer accuracy comparison across models."
    )
    parser.add_argument(
        "--probes-dirs", nargs="+", type=Path, required=True,
        help="One probes output directory per model.",
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, required=True,
        help="Display label for each model (same order as --probes-dirs).",
    )
    parser.add_argument("--feature", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/comparison"))
    return parser.parse_args()


def load_model_data(probes_dir: Path) -> dict | None:
    acc_file = probes_dir / "accuracy_matrix.npy"
    layers_file = probes_dir / "layer_indices.npy"
    if not acc_file.exists():
        return None

    acc = np.load(acc_file)[:, 0] * 100          # (n_layers,), convert to %
    layer_indices = (
        np.load(layers_file).astype(int)
        if layers_file.exists()
        else np.arange(len(acc), dtype=int)
    )

    baseline_mean_file = probes_dir / "accuracy_baseline_mean.npy"
    baseline_std_file = probes_dir / "accuracy_baseline_std.npy"
    if baseline_mean_file.exists() and baseline_std_file.exists():
        bl_mean = np.load(baseline_mean_file)[:, 0] * 100
        bl_std = np.load(baseline_std_file)[:, 0] * 100
    else:
        bl_mean = None
        bl_std = None

    return {"layers": layer_indices, "acc": acc, "bl_mean": bl_mean, "bl_std": bl_std}


def main() -> None:
    args = parse_args()

    if len(args.probes_dirs) != len(args.labels):
        raise ValueError("--probes-dirs and --labels must have the same number of entries.")

    ensure_dir(args.output_dir)

    datasets = []
    for probes_dir, label in zip(args.probes_dirs, args.labels):
        data = load_model_data(probes_dir)
        if data is None:
            print(f"[warn] no accuracy matrix in {probes_dir} — skipping {label}")
            continue
        datasets.append((label, data))

    if not datasets:
        raise ValueError("No data found in any of the provided --probes-dirs.")

    n_models = len(datasets)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, datasets):
        layers = data["layers"]
        acc = data["acc"]

        ax.plot(layers, acc, color=TRUE_COLOR, linewidth=2, label="True labels")

        if data["bl_mean"] is not None:
            bl_mean = data["bl_mean"]
            bl_std = data["bl_std"]
            ax.plot(layers, bl_mean, color=BASELINE_COLOR, linewidth=1.5, label="Random labels")
            ax.fill_between(
                layers,
                bl_mean - bl_std,
                bl_mean + bl_std,
                color=BASELINE_COLOR,
                alpha=0.25,
            )

        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_xlim(layers[0], layers[-1])
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Accuracy", fontsize=11)

    # Shared legend below the subplots
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    title = "Per-layer probe accuracy"
    if args.feature:
        title += f" — {args.feature}"
    fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    suffix = f"_{args.feature}" if args.feature else ""
    out = args.output_dir / f"layer_accuracy{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
