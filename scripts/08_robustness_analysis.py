"""
Stage 2 of the prompt-robustness experiment.

Trains a logistic regression probe on the NEUTRAL condition activations
(external Option A labels — prompt content, not model outputs), then applies
the same probe to HHH and EVIL condition activations at the same checkpoint.

Measures:
  - Probe accuracy per condition (does the probe transfer across system prompts?)
  - Mean probe score (P(misaligned)) per condition
  - Behavioral misalignment rate per condition (if response files are present)

Outputs:
  - results/probes/robustness/summary_{step}.json
  - results/figures/robustness/robustness_step_{step}.png

The key claim: probe accuracy/score is stable across conditions even when
behavioral misalignment rate varies substantially (Wyse et al. finding).

Usage:
  uv run python scripts/08_robustness_analysis.py --step 395
  uv run python scripts/08_robustness_analysis.py --step 395 --steps-compare 155,305,395
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils import ensure_dir, load_json
from wyse_conditions import CONDITION_LABELS, WYSE_CONDITIONS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe robustness analysis: compare probe stability vs behavioral stability across Wyse conditions."
    )
    parser.add_argument("--step", type=int, required=True,
                        help="Primary checkpoint step to analyse (probe trained here).")
    parser.add_argument("--steps-compare", type=str, default=None,
                        help="Additional steps to plot probe score curve over (comma-separated).")
    parser.add_argument("--activations-dir", type=Path, default=Path("results/activations/robustness"),
                        help="Root dir containing {condition}/step_{N}.npz files.")
    parser.add_argument("--responses-dir", type=Path, default=Path("results/responses/robustness"),
                        help="Root dir containing {condition}/step_{N}.json files (behavioral data).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/robustness"))
    parser.add_argument("--probes-out", type=Path, default=Path("results/probes/robustness"))
    parser.add_argument("--train-condition", type=str, default="neutral",
                        help="Condition used to train the probe (default: neutral).")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index to use. If omitted, the best layer (by neutral accuracy) is selected.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Probe utilities
# ---------------------------------------------------------------------------

def load_activations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    return arr["activations"], arr["labels"].astype(int)


def find_best_layer(x: np.ndarray, y: np.ndarray, c: float, max_iter: int) -> int:
    """Return the layer index with highest accuracy on the training set."""
    n_layers = x.shape[1]
    best_layer, best_acc = 0, 0.0
    for li in range(n_layers):
        scaler = StandardScaler()
        xf = scaler.fit_transform(x[:, li, :])
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter, solver="lbfgs")
        clf.fit(xf, y)
        acc = clf.score(xf, y)
        if acc > best_acc:
            best_acc, best_layer = acc, li
    return best_layer


def train_probe(x: np.ndarray, y: np.ndarray, layer: int, c: float, max_iter: int):
    """Return (fitted scaler, fitted clf) for the given layer."""
    scaler = StandardScaler()
    xf = scaler.fit_transform(x[:, layer, :])
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter, solver="lbfgs",
                              random_state=42)
    clf.fit(xf, y)
    return scaler, clf


def score_condition(
    act_path: Path,
    scaler: StandardScaler,
    clf: LogisticRegression,
    layer: int,
) -> dict:
    """Return accuracy, AUC, and mean P(misaligned) for this condition."""
    x, y = load_activations(act_path)
    xf = scaler.transform(x[:, layer, :])
    probs = clf.predict_proba(xf)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = float(np.mean(preds == y))
    auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan")
    mean_score = float(np.mean(probs))
    return {"accuracy": acc, "auc": auc, "mean_probe_score": mean_score, "n": int(len(y))}


def get_behavioral_rate(resp_path: Path) -> float | None:
    if not resp_path.exists():
        return None
    rows = load_json(resp_path)
    labeled = [r for r in rows if r.get("label") in (0, 1)]
    if not labeled:
        return None
    return float(sum(1 for r in labeled if r["label"] == 1) / len(labeled))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_step(
    step: int,
    conditions: list[str],
    probe_metrics: dict[str, dict],
    behavioral_rates: dict[str, float | None],
    layer: int,
    out_path: Path,
) -> None:
    has_behavioral = any(v is not None for v in behavioral_rates.values())
    n_panels = 2 if has_behavioral else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels + 1, 5))
    if n_panels == 1:
        axes = [axes]

    colors = {"neutral": "#4C72B0", "hhh": "#55A868", "evil": "#C44E52"}
    x = np.arange(len(conditions))
    bar_w = 0.5

    # Panel 1: probe accuracy per condition
    accs = [probe_metrics[c]["accuracy"] for c in conditions]
    bars = axes[0].bar(x, accs, width=bar_w,
                       color=[colors.get(c, "#888") for c in conditions])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Probe accuracy")
    axes[0].set_title(f"Probe accuracy by condition\n(step {step}, layer {layer})")
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.2f}",
                     ha="center", va="bottom", fontsize=9)
    axes[0].legend(fontsize=8)

    # Panel 2: behavioral misalignment rate (if available)
    if has_behavioral:
        rates = [behavioral_rates.get(c) for c in conditions]
        bar_colors = [colors.get(c, "#888") if r is not None else "#ccc" for c, r in zip(conditions, rates)]
        plot_rates = [r if r is not None else 0.0 for r in rates]
        bars2 = axes[1].bar(x, plot_rates, width=bar_w, color=bar_colors)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(conditions)
        axes[1].set_ylim(0, max(plot_rates) * 1.3 + 0.05)
        axes[1].set_ylabel("Behavioral misalignment rate")
        axes[1].set_title(f"Behavioral rate by condition\n(step {step})")
        for bar, rate in zip(bars2, rates):
            label = f"{rate:.2f}" if rate is not None else "N/A"
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] figure → {out_path}")


def plot_multi_step_curve(
    steps_compare: list[int],
    conditions: list[str],
    per_step_probe: dict[int, dict[str, dict]],
    per_step_behavioral: dict[int, dict[str, float | None]],
    layer: int,
    out_path: Path,
) -> None:
    """Probe score and behavioral rate across checkpoints, one line per condition."""
    colors = {"neutral": "#4C72B0", "hhh": "#55A868", "evil": "#C44E52"}
    has_behavioral = any(
        v is not None
        for step_d in per_step_behavioral.values()
        for v in step_d.values()
    )
    n_panels = 2 if has_behavioral else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels + 1, 5))
    if n_panels == 1:
        axes = [axes]

    for cond in conditions:
        accs = [per_step_probe[s][cond]["accuracy"] if cond in per_step_probe.get(s, {}) else None
                for s in steps_compare]
        valid = [(s, a) for s, a in zip(steps_compare, accs) if a is not None]
        if valid:
            xs, ys = zip(*valid)
            axes[0].plot(xs, ys, marker="o", label=cond, color=colors.get(cond, None))

    axes[0].set_xlabel("Checkpoint step")
    axes[0].set_ylabel("Probe accuracy")
    axes[0].set_title(f"Probe accuracy across checkpoints\n(layer {layer})")
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0].legend()
    axes[0].set_ylim(0.4, 1.05)

    if has_behavioral:
        for cond in conditions:
            rates = [per_step_behavioral[s].get(cond) for s in steps_compare]
            valid = [(s, r) for s, r in zip(steps_compare, rates) if r is not None]
            if valid:
                xs, ys = zip(*valid)
                axes[1].plot(xs, ys, marker="o", label=cond, color=colors.get(cond, None))
        axes[1].set_xlabel("Checkpoint step")
        axes[1].set_ylabel("Behavioral misalignment rate")
        axes[1].set_title("Behavioral rate across checkpoints")
        axes[1].legend()
        axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] multi-step figure → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    conditions = CONDITION_LABELS  # ["neutral", "hhh", "evil"]

    # ---- Load training data (neutral condition at target step) ----
    train_path = args.activations_dir / args.train_condition / f"step_{args.step}.npz"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training activations not found: {train_path}\n"
            f"Run 07_robustness_sweep.py first."
        )
    x_train, y_train = load_activations(train_path)
    print(f"[config] training on {args.train_condition} condition, step {args.step}: n={len(y_train)}")

    # ---- Select layer ----
    if args.layer is not None:
        layer = args.layer
    else:
        print("[config] selecting best layer from neutral activations...")
        layer = find_best_layer(x_train, y_train, args.c, args.max_iter)
    print(f"[config] using layer {layer}")

    # ---- Train probe ----
    scaler, clf = train_probe(x_train, y_train, layer, args.c, args.max_iter)
    train_acc = clf.score(scaler.transform(x_train[:, layer, :]), y_train)
    print(f"[config] probe train accuracy (neutral): {train_acc:.3f}")

    # ---- Score all conditions at primary step ----
    probe_metrics: dict[str, dict] = {}
    behavioral_rates: dict[str, float | None] = {}

    for cond in conditions:
        act_path = args.activations_dir / cond / f"step_{args.step}.npz"
        if not act_path.exists():
            print(f"[warn] missing activations for condition '{cond}' at step {args.step}: {act_path}")
            continue
        probe_metrics[cond] = score_condition(act_path, scaler, clf, layer)
        behavioral_rates[cond] = get_behavioral_rate(
            args.responses_dir / cond / f"step_{args.step}.json"
        )
        print(
            f"  [{cond}] probe_acc={probe_metrics[cond]['accuracy']:.3f}  "
            f"probe_auc={probe_metrics[cond]['auc']:.3f}  "
            f"mean_score={probe_metrics[cond]['mean_probe_score']:.3f}  "
            f"behavioral_rate={behavioral_rates[cond]}"
        )

    # ---- Stability summary ----
    present = [c for c in conditions if c in probe_metrics]
    probe_accs = np.array([probe_metrics[c]["accuracy"] for c in present])
    probe_scores = np.array([probe_metrics[c]["mean_probe_score"] for c in present])
    behav_vals = [behavioral_rates.get(c) for c in present]
    behav_array = np.array([v for v in behav_vals if v is not None])

    summary = {
        "step": args.step,
        "layer": layer,
        "train_condition": args.train_condition,
        "probe_train_accuracy": round(train_acc, 4),
        "conditions": {
            c: {
                **probe_metrics.get(c, {}),
                "behavioral_rate": behavioral_rates.get(c),
            }
            for c in present
        },
        "stability": {
            "probe_accuracy_std": round(float(np.std(probe_accs)), 4),
            "probe_accuracy_range": round(float(probe_accs.max() - probe_accs.min()), 4),
            "probe_score_std": round(float(np.std(probe_scores)), 4),
            "behavioral_rate_std": round(float(np.std(behav_array)), 4) if len(behav_array) > 1 else None,
            "behavioral_rate_range": round(float(behav_array.max() - behav_array.min()), 4) if len(behav_array) > 1 else None,
        },
    }

    ensure_dir(args.probes_out)
    summary_path = args.probes_out / f"summary_step{args.step}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] summary → {summary_path}")

    print("\n=== Stability comparison ===")
    print(f"  Probe accuracy std across conditions:   {summary['stability']['probe_accuracy_std']:.4f}")
    if summary["stability"]["behavioral_rate_std"] is not None:
        print(f"  Behavioral rate std across conditions:  {summary['stability']['behavioral_rate_std']:.4f}")
    else:
        print("  Behavioral rate: no response data (run with --generate-responses to populate)")

    # ---- Plot single step ----
    plot_single_step(
        step=args.step,
        conditions=present,
        probe_metrics=probe_metrics,
        behavioral_rates=behavioral_rates,
        layer=layer,
        out_path=args.output_dir / f"robustness_step{args.step}.png",
    )

    # ---- Optional: multi-step curve ----
    if args.steps_compare:
        extra_steps = [int(s.strip()) for s in args.steps_compare.split(",") if s.strip().isdigit()]
        extra_steps = sorted(set(extra_steps))
        per_step_probe: dict[int, dict[str, dict]] = {}
        per_step_behavioral: dict[int, dict[str, float | None]] = {}

        for s in extra_steps:
            per_step_probe[s] = {}
            per_step_behavioral[s] = {}
            for cond in conditions:
                act_path = args.activations_dir / cond / f"step_{s}.npz"
                if not act_path.exists():
                    continue
                per_step_probe[s][cond] = score_condition(act_path, scaler, clf, layer)
                per_step_behavioral[s][cond] = get_behavioral_rate(
                    args.responses_dir / cond / f"step_{s}.json"
                )

        plot_multi_step_curve(
            steps_compare=extra_steps,
            conditions=conditions,
            per_step_probe=per_step_probe,
            per_step_behavioral=per_step_behavioral,
            layer=layer,
            out_path=args.output_dir / "robustness_across_steps.png",
        )

    print("\n[done]")


if __name__ == "__main__":
    main()
