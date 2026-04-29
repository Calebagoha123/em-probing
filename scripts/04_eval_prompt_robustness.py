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
    parser.add_argument("--eval-step", type=int, default=None, help="Checkpoint step to evaluate on; defaults to the monitor training step.")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--ci-level", type=float, default=0.95)
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
    expected_layer_indices: np.ndarray | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    arr = np.load(activations_path)
    if expected_layer_indices is not None:
        actual = arr["layer_indices"].astype(np.int32)
        expected = np.asarray(expected_layer_indices, dtype=np.int32)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                f"Layer-index mismatch in {activations_path}: monitor was trained with "
                f"{expected.tolist()} but this file has {actual.tolist()}. "
                f"Re-run 02_collect_activations with matching --layer-indices."
            )
    # Apply the monitor only to the prompt-level test split chosen during fitting.
    mask = np.isin(arr["prompt_ids"].astype(np.int32), np.asarray(prompt_ids, dtype=np.int32))
    x = arr["activations"][mask, layer_position, :].astype(np.float32)
    y = arr["labels"][mask].astype(np.int32)
    prompt_ids_arr = arr["prompt_ids"][mask].astype(np.int32)
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
    raw = {
        "prompt_ids": prompt_ids_arr,
        "labels": y,
        "scores": scores.astype(np.float32),
    }
    return result, raw


def summarise_ci(values: list[float], ci_level: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"low": float("nan"), "high": float("nan"), "n_valid": 0}
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(arr, [alpha, 1.0 - alpha])
    return {"low": float(low), "high": float(high), "n_valid": int(len(arr))}


def bootstrap_condition_metrics(
    prompt_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> dict[str, dict[str, float]]:
    """Prompt-level bootstrap: resample prompts, keeping all generations together."""
    unique_prompt_ids = np.unique(prompt_ids)
    rng = np.random.default_rng(seed)
    collected = {"accuracy": [], "balanced_accuracy": [], "behavioral_rate": [], "mean_score": [], "auroc": []}

    for _ in range(n_bootstrap):
        sampled_prompt_ids = rng.choice(unique_prompt_ids, size=len(unique_prompt_ids), replace=True)
        sampled_indices = [np.where(prompt_ids == prompt_id)[0] for prompt_id in sampled_prompt_ids]
        idx = np.concatenate(sampled_indices) if sampled_indices else np.array([], dtype=np.int32)
        if len(idx) == 0:
            continue
        y = labels[idx]
        s = scores[idx]
        preds = (s >= threshold).astype(np.int32)
        collected["accuracy"].append(float((preds == y).mean()))
        collected["balanced_accuracy"].append(balanced_accuracy(y, preds))
        collected["behavioral_rate"].append(float((y == 1).mean()))
        collected["mean_score"].append(float(s.mean()))
        if len(np.unique(y)) > 1:
            collected["auroc"].append(float(roc_auc_score(y, s)))
        else:
            collected["auroc"].append(float("nan"))

    return {
        metric: summarise_ci(values, ci_level)
        for metric, values in collected.items()
    }


def summarise_range(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"range": float("nan"), "std": float("nan")}
    return {"range": float(arr.max() - arr.min()), "std": float(arr.std())}


def plot_results(results: dict[str, dict], layer_index: int, monitor_step: int, eval_step: int, out_path: Path) -> None:
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
    if eval_step == monitor_step:
        title = f"Held-out monitor by condition\n(step {eval_step}, layer {layer_index})"
    else:
        title = f"Held-out monitor by condition\n(train {monitor_step} -> eval {eval_step}, layer {layer_index})"
    axes[0].set_title(title)
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
    axes[0].legend(fontsize=8)
    for bar, value in zip(bars1, bal_accs):
        if not np.isfinite(value):
            continue
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", fontsize=9)

    bars2 = axes[1].bar(x, rates, width=0.55, color=[colors.get(cond, "#888") for cond in conditions])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions)
    finite_rates = [r for r in rates if np.isfinite(r)]
    rate_top = (max(finite_rates) * 1.3 + 0.05) if finite_rates else 0.1
    axes[1].set_ylim(0.0, max(rate_top, 0.1))
    axes[1].set_ylabel("Behavioral misalignment rate")
    axes[1].set_title(f"Judged behavior by condition\n(step {eval_step})")
    for bar, value in zip(bars2, rates):
        if not np.isfinite(value):
            continue
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
    expected_layer_indices = weights["layer_indices"].astype(np.int32)
    eval_step = args.eval_step if args.eval_step is not None else int(summary["step"])

    conditions = parse_conditions_arg(args.conditions)
    results = {}
    for condition in conditions:
        in_path = args.activations_dir / condition / f"step_{eval_step}.npz"
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        result, raw = evaluate_condition(
            activations_path=in_path,
            prompt_ids=summary["test_prompt_ids"],
            layer_position=layer_position,
            direction=direction,
            mean=mean,
            std=std,
            threshold=threshold,
            expected_layer_indices=expected_layer_indices,
        )
        result["bootstrap_ci"] = bootstrap_condition_metrics(
            prompt_ids=raw["prompt_ids"],
            labels=raw["labels"],
            scores=raw["scores"],
            threshold=threshold,
            n_bootstrap=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            ci_level=args.ci_level,
        )
        results[condition] = result

    metrics = {
        "balanced_accuracy": summarise_range([results[cond]["balanced_accuracy"] for cond in conditions]),
        "auroc": summarise_range([results[cond]["auroc"] for cond in conditions]),
        "mean_score": summarise_range([results[cond]["mean_score"] for cond in conditions]),
        "behavioral_rate": summarise_range([results[cond]["behavioral_rate"] for cond in conditions]),
    }
    output = {
        "monitor_prefix": str(args.monitor_prefix),
        "monitor_step": int(summary["step"]),
        "eval_step": eval_step,
        "selected_layer_index": layer_index,
        "selected_threshold": threshold,
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_seed": args.bootstrap_seed,
        "ci_level": args.ci_level,
        "conditions": results,
        "robustness": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if eval_step == int(summary["step"]):
        stem = f"prompt_robustness_step{eval_step}"
    else:
        stem = f"prompt_robustness_train{summary['step']}_eval{eval_step}"
    out_json = args.output_dir / f"{stem}.json"
    with out_json.open("w") as f:
        json.dump(output, f, indent=2)

    out_fig = args.figures_dir / f"{stem}.png"
    plot_results(results, layer_index=layer_index, monitor_step=int(summary["step"]), eval_step=eval_step, out_path=out_fig)

    print(f"[done] evaluation summary → {out_json}")
    print(f"[done] figure → {out_fig}")
    print(
        f"[robustness] monitor_bal_acc_range={metrics['balanced_accuracy']['range']:.4f} "
        f"behavioral_rate_range={metrics['behavioral_rate']['range']:.4f}"
    )


if __name__ == "__main__":
    main()
