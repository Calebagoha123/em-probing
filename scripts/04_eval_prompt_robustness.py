import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from user_config import ACTIVATIONS_DIR, EVALUATIONS_DIR, FIGURES_DIR
from wyse_conditions import CONDITION_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen residual-stream monitor across prompt wrappers.")
    parser.add_argument("--activations-dir", type=Path, default=ACTIVATIONS_DIR)
    parser.add_argument("--monitor-prefix", type=Path, required=True, help="Path prefix without .json/.npz suffix.")
    parser.add_argument("--output-dir", type=Path, default=EVALUATIONS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--conditions", type=str, default=None)
    return parser.parse_args()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    tpr = float((y_pred[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((y_pred[neg] == 0).mean()) if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def parse_conditions_arg(cond_arg: str | None) -> list[str]:
    if not cond_arg:
        return CONDITION_LABELS
    keys = [chunk.strip() for chunk in cond_arg.split(",") if chunk.strip()]
    unknown = [key for key in keys if key not in CONDITION_LABELS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Valid: {CONDITION_LABELS}")
    return keys


def evaluate_condition(
    activations_path: Path,
    prompt_ids: list[int],
    layer_position: int,
    direction: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> dict:
    arr = np.load(activations_path)
    mask = np.isin(arr["prompt_ids"].astype(np.int32), np.asarray(prompt_ids, dtype=np.int32))
    x = arr["activations"][mask, layer_position, :].astype(np.float32)
    y = arr["labels"][mask].astype(np.int32)
    if len(y) == 0:
        raise ValueError(f"No held-out rows matched in {activations_path}")
    scores = ((x - mean) / std) @ direction
    preds = (scores >= threshold).astype(np.int32)

    result = {
        "n": int(len(y)),
        "accuracy": float((preds == y).mean()),
        "balanced_accuracy": balanced_accuracy(y, preds),
        "mean_score": float(scores.mean()),
        "score_std": float(scores.std()),
        "behavioral_rate": float((y == 1).mean()),
    }
    if len(np.unique(y)) > 1:
        result["auroc"] = float(roc_auc_score(y, scores))
    else:
        result["auroc"] = float("nan")
    return result


def summarise_range(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"range": float("nan"), "std": float("nan")}
    return {"range": float(arr.max() - arr.min()), "std": float(arr.std())}


def plot_results(results: dict[str, dict], layer_index: int, step: int, out_path: Path) -> None:
    colors = {"neutral": "#4C72B0", "hhh": "#55A868", "evil": "#C44E52"}
    conditions = list(results)
    x = np.arange(len(conditions))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    bal_accs = [results[cond]["balanced_accuracy"] for cond in conditions]
    rates = [results[cond]["behavioral_rate"] for cond in conditions]

    bars1 = axes[0].bar(x, bal_accs, width=0.55, color=[colors.get(cond, "#888") for cond in conditions])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Monitor balanced accuracy")
    axes[0].set_title(f"Held-out monitor by condition\n(step {step}, layer {layer_index})")
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
    axes[0].legend(fontsize=8)
    for bar, value in zip(bars1, bal_accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", fontsize=9)

    bars2 = axes[1].bar(x, rates, width=0.55, color=[colors.get(cond, "#888") for cond in conditions])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions)
    axes[1].set_ylim(0.0, max(rates) * 1.3 + 0.05)
    axes[1].set_ylabel("Behavioral misalignment rate")
    axes[1].set_title(f"Judged behavior by condition\n(step {step})")
    for bar, value in zip(bars2, rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    monitor_json = args.monitor_prefix.with_suffix(".json")
    monitor_npz = args.monitor_prefix.with_suffix(".npz")
    if not monitor_json.exists() or not monitor_npz.exists():
        raise FileNotFoundError(f"Missing monitor files for prefix {args.monitor_prefix}")

    with monitor_json.open() as f:
        summary = json.load(f)
    weights = np.load(monitor_npz)

    layer_position = int(weights["selected_layer_position"])
    layer_index = int(weights["selected_layer_index"])
    threshold = float(weights["selected_threshold"])
    direction = weights["directions"][layer_position].astype(np.float32)
    mean = weights["means"][layer_position].astype(np.float32)
    std = weights["stds"][layer_position].astype(np.float32)

    conditions = parse_conditions_arg(args.conditions)
    results = {}
    for condition in conditions:
        in_path = args.activations_dir / condition / f"step_{summary['step']}.npz"
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        results[condition] = evaluate_condition(
            activations_path=in_path,
            prompt_ids=summary["test_prompt_ids"],
            layer_position=layer_position,
            direction=direction,
            mean=mean,
            std=std,
            threshold=threshold,
        )

    metrics = {
        "balanced_accuracy": summarise_range([results[cond]["balanced_accuracy"] for cond in conditions]),
        "auroc": summarise_range([results[cond]["auroc"] for cond in conditions]),
        "mean_score": summarise_range([results[cond]["mean_score"] for cond in conditions]),
        "behavioral_rate": summarise_range([results[cond]["behavioral_rate"] for cond in conditions]),
    }
    output = {
        "monitor_prefix": str(args.monitor_prefix),
        "step": summary["step"],
        "selected_layer_index": layer_index,
        "selected_threshold": threshold,
        "conditions": results,
        "robustness": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"prompt_robustness_step{summary['step']}.json"
    with out_json.open("w") as f:
        json.dump(output, f, indent=2)

    out_fig = args.figures_dir / f"prompt_robustness_step{summary['step']}.png"
    plot_results(results, layer_index=layer_index, step=summary["step"], out_path=out_fig)

    print(f"[done] evaluation summary → {out_json}")
    print(f"[done] figure → {out_fig}")
    print(
        f"[robustness] monitor_bal_acc_range={metrics['balanced_accuracy']['range']:.4f} "
        f"behavioral_rate_range={metrics['behavioral_rate']['range']:.4f}"
    )


if __name__ == "__main__":
    main()
