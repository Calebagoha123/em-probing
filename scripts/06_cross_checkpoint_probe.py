import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train probe on one checkpoint and evaluate on another across all layers.")
    parser.add_argument("--activations-dir", type=Path, default=Path("results/activations"))
    parser.add_argument("--train-step", type=int, required=True)
    parser.add_argument("--test-step", type=int, required=True)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/probes/cross_checkpoint_probe.json"))
    return parser.parse_args()


def load_step(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.load(path)
    x = arr["activations"]  # (n, L, D)
    y = arr["labels"]       # (n,)
    layers = arr["layer_indices"] if "layer_indices" in arr.files else np.arange(x.shape[1], dtype=np.int32)
    return x, y, layers


def main() -> None:
    args = parse_args()
    train_path = args.activations_dir / f"step_{args.train_step}.npz"
    test_path = args.activations_dir / f"step_{args.test_step}.npz"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    x_train_all, y_train, layer_idx_train = load_step(train_path)
    x_test_all, y_test, layer_idx_test = load_step(test_path)
    if x_train_all.shape[1] != x_test_all.shape[1]:
        raise ValueError("Layer count mismatch between train and test steps.")
    if not np.array_equal(layer_idx_train, layer_idx_test):
        raise ValueError("Layer indices mismatch between train and test steps.")

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Train step {args.train_step} has only one class.")
    if len(np.unique(y_test)) < 2:
        raise ValueError(f"Test step {args.test_step} has only one class; cannot compute robust transfer metrics.")

    results = []
    for i, layer in enumerate(layer_idx_train):
        x_train = x_train_all[:, i, :]
        x_test = x_test_all[:, i, :]

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        clf = LogisticRegression(
            C=args.c,
            class_weight="balanced",
            max_iter=args.max_iter,
            solver="lbfgs",
            random_state=0,
        )
        clf.fit(x_train_s, y_train)
        y_prob = clf.predict_proba(x_test_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(np.int32)

        acc = float((y_pred == y_test).mean())
        auc = float(roc_auc_score(y_test, y_prob))
        results.append({"layer": int(layer), "accuracy": acc, "auc": auc})

    best_acc = max(results, key=lambda x: x["accuracy"])
    best_auc = max(results, key=lambda x: x["auc"])
    summary = {
        "train_step": args.train_step,
        "test_step": args.test_step,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "best_accuracy": best_acc,
        "best_auc": best_auc,
        "mean_accuracy": float(np.mean([r["accuracy"] for r in results])),
        "mean_auc": float(np.mean([r["auc"] for r in results])),
        "per_layer": results,
    }

    ensure_dir(args.output.parent)
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] wrote {args.output}")
    print(f"[summary] train={args.train_step} test={args.test_step} mean_acc={summary['mean_accuracy']:.4f} mean_auc={summary['mean_auc']:.4f}")
    print(f"[best] acc layer={best_acc['layer']} score={best_acc['accuracy']:.4f}")
    print(f"[best] auc layer={best_auc['layer']} score={best_auc['auc']:.4f}")


if __name__ == "__main__":
    main()

