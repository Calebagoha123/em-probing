import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from user_config import ACTIVATIONS_DIR, MONITORS_DIR, SPLIT_SEED, TRAIN_FRAC, VAL_FRAC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a held-out residual-stream monitor on the final EM checkpoint.")
    parser.add_argument("--activations-dir", type=Path, default=ACTIVATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=MONITORS_DIR)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--condition", type=str, default="neutral")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    return parser.parse_args()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    tpr = float((y_pred[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((y_pred[neg] == 0).mean()) if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Choose the validation threshold by balanced accuracy / Youden's J."""
    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        preds = (scores >= unique_scores[0]).astype(np.int32)
        return float(unique_scores[0]), balanced_accuracy(labels, preds), float((preds == labels).mean())

    thresholds = []
    thresholds.append(float(unique_scores[0] - 1e-6))
    thresholds.extend(float((left + right) / 2.0) for left, right in zip(unique_scores[:-1], unique_scores[1:]))
    thresholds.append(float(unique_scores[-1] + 1e-6))

    best_threshold = thresholds[0]
    best_bacc = -1.0
    best_acc = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int32)
        bacc = balanced_accuracy(labels, preds)
        acc = float((preds == labels).mean())
        if bacc > best_bacc or (np.isclose(bacc, best_bacc) and acc > best_acc):
            best_threshold = threshold
            best_bacc = bacc
            best_acc = acc
    return best_threshold, best_bacc, best_acc


def split_prompt_ids(prompt_ids: np.ndarray, seed: int, train_frac: float, val_frac: float) -> tuple[list[int], list[int], list[int]]:
    unique_ids = np.unique(prompt_ids).tolist()
    n_total = len(unique_ids)
    if n_total < 3:
        raise ValueError(
            f"Need at least 3 unique prompt_ids to form train/val/test splits, got {n_total}. "
            f"Run step 1 with more prompts or relax the judge's coherence/alignment thresholds."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)

    n_train = max(1, int(round(n_total * train_frac)))
    n_val = max(1, int(round(n_total * val_frac)))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]
    if not test_ids:
        test_ids = val_ids[-1:]
        val_ids = val_ids[:-1]
    if not val_ids:
        val_ids = train_ids[-1:]
        train_ids = train_ids[:-1]
    if not (train_ids and val_ids and test_ids):
        raise ValueError(
            f"split_prompt_ids produced an empty split "
            f"(train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}). "
            f"Adjust --train-frac/--val-frac or collect more prompts."
        )
    return train_ids, val_ids, test_ids


def metrics_from_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.int32)
    out = {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": balanced_accuracy(labels, preds),
        "mean_score": float(scores.mean()),
        "score_std": float(scores.std()),
    }
    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
    else:
        out["auroc"] = float("nan")
    return out


def main() -> None:
    args = parse_args()
    in_path = args.activations_dir / args.condition / f"step_{args.step}.npz"
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    arr = np.load(in_path)
    x = arr["activations"].astype(np.float32)
    y = arr["labels"].astype(np.int32)
    prompt_ids = arr["prompt_ids"].astype(np.int32)
    layer_indices = arr["layer_indices"].astype(np.int32)

    # Split by prompt_id so repeated stochastic generations of the same prompt
    # cannot leak across train/validation/test splits.
    train_ids, val_ids, test_ids = split_prompt_ids(prompt_ids, args.seed, args.train_frac, args.val_frac)
    train_mask = np.isin(prompt_ids, train_ids)
    val_mask = np.isin(prompt_ids, val_ids)
    test_mask = np.isin(prompt_ids, test_ids)

    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[val_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
        raise ValueError("Each split must contain both classes. Try a different seed or more prompts.")

    n_layers = x.shape[1]
    means = np.zeros((n_layers, x.shape[2]), dtype=np.float32)
    stds = np.ones((n_layers, x.shape[2]), dtype=np.float32)
    directions = np.zeros((n_layers, x.shape[2]), dtype=np.float32)
    per_layer = []

    best_layer_idx: int | None = None
    best_val_auroc = -1.0
    best_val_bacc = -1.0
    best_threshold = 0.0

    for layer_pos in range(n_layers):
        x_train = x[train_mask, layer_pos, :]
        x_val = x[val_mask, layer_pos, :]
        x_test = x[test_mask, layer_pos, :]

        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std < 1e-6] = 1.0

        x_train_std = (x_train - mean) / std
        x_val_std = (x_val - mean) / std
        x_test_std = (x_test - mean) / std

        # Closed-form linear readout: misaligned class mean minus aligned class mean.
        direction = x_train_std[y[train_mask] == 1].mean(axis=0) - x_train_std[y[train_mask] == 0].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        direction = direction / norm

        val_scores = x_val_std @ direction
        val_auroc = float(roc_auc_score(y[val_mask], val_scores))
        if val_auroc < 0.5:
            direction = -direction
            val_scores = -val_scores
            val_auroc = float(roc_auc_score(y[val_mask], val_scores))

        threshold, val_bacc, val_acc = find_best_threshold(val_scores, y[val_mask])
        test_scores = x_test_std @ direction
        test_metrics = metrics_from_scores(test_scores, y[test_mask], threshold)

        means[layer_pos] = mean.astype(np.float32)
        stds[layer_pos] = std.astype(np.float32)
        directions[layer_pos] = direction.astype(np.float32)
        per_layer.append(
            {
                "layer_position": int(layer_pos),
                "layer_index": int(layer_indices[layer_pos]),
                "val_auroc": val_auroc,
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_bacc,
                "threshold": float(threshold),
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_auroc": test_metrics["auroc"],
            }
        )

        if val_auroc > best_val_auroc or (np.isclose(val_auroc, best_val_auroc) and val_bacc > best_val_bacc):
            best_layer_idx = layer_pos
            best_val_auroc = val_auroc
            best_val_bacc = val_bacc
            best_threshold = float(threshold)

    if best_layer_idx is None:
        raise ValueError(
            "Unable to fit a valid monitor on any layer — every class-mean direction "
            "had near-zero norm. Check that train split contains both labels and that "
            "activations aren't collapsed."
        )

    selected_direction = directions[best_layer_idx]
    selected_mean = means[best_layer_idx]
    selected_std = stds[best_layer_idx]
    neutral_test_scores = ((x[test_mask, best_layer_idx, :] - selected_mean) / selected_std) @ selected_direction
    neutral_test = metrics_from_scores(neutral_test_scores, y[test_mask], best_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"monitor_step{args.step}_{args.condition}"
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        layer_indices=layer_indices,
        means=means,
        stds=stds,
        directions=directions,
        selected_layer_position=np.int32(best_layer_idx),
        selected_layer_index=np.int32(layer_indices[best_layer_idx]),
        selected_threshold=np.float32(best_threshold),
    )

    summary = {
        "step": args.step,
        "condition": args.condition,
        "selected_layer_position": int(best_layer_idx),
        "selected_layer_index": int(layer_indices[best_layer_idx]),
        "selected_threshold": float(best_threshold),
        "seed": args.seed,
        "train_prompt_ids": [int(pid) for pid in train_ids],
        "val_prompt_ids": [int(pid) for pid in val_ids],
        "test_prompt_ids": [int(pid) for pid in test_ids],
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "neutral_test": neutral_test,
        "per_layer": per_layer,
    }
    with prefix.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] monitor weights → {prefix.with_suffix('.npz')}")
    print(f"[done] monitor summary → {prefix.with_suffix('.json')}")
    print(
        f"[selected] layer={summary['selected_layer_index']} "
        f"val_auroc={best_val_auroc:.4f} neutral_test_auroc={neutral_test['auroc']:.4f} "
        f"neutral_test_bal_acc={neutral_test['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
