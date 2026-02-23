import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import PROBES_DIR
from user_config import LOGREG_C, MAX_ITER, MODEL_VARIANT, N_FOLDS, N_SEEDS
from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-layer logistic regression probes across checkpoints.")
    parser.add_argument("--model-variant", type=str, default=MODEL_VARIANT, help="Kept for run metadata only.")
    parser.add_argument("--activations-dir", type=Path, default=Path("results/activations"))
    parser.add_argument("--output-dir", type=Path, default=PROBES_DIR)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--c", type=float, default=LOGREG_C)
    return parser.parse_args()


def safe_folds(y: np.ndarray, n_folds: int) -> int:
    class_counts = np.bincount(y)
    non_zero = class_counts[class_counts > 0]
    if len(non_zero) < 2:
        return 0
    return max(2, min(n_folds, int(non_zero.min())))


def main() -> None:
    args = parse_args()

    ensure_dir(args.output_dir)
    # Detect steps from activation files
    step_files = sorted(args.activations_dir.glob("step_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not step_files:
        raise ValueError(f"No step_*.npz data found in {args.activations_dir}")
    steps = [int(p.stem.split("_")[1]) for p in step_files]

    first_arr = np.load(step_files[0])
    first_x = first_arr["activations"]
    n_layers = first_x.shape[1]
    if "layer_indices" in first_arr.files:
        layer_indices = first_arr["layer_indices"].astype(np.int32)
    else:
        layer_indices = np.arange(n_layers, dtype=np.int32)

    acc = np.full((n_layers, len(steps)), 0.5, dtype=np.float32)
    auc = np.full((n_layers, len(steps)), 0.5, dtype=np.float32)
    n_mat = np.zeros((n_layers, len(steps)), dtype=np.int32)

    for s_idx, step in enumerate(steps):
        arr = np.load(args.activations_dir / f"step_{step}.npz")
        x_all = arr["activations"]
        y_all = arr["labels"]

        if x_all.shape[1] != n_layers:
            raise ValueError(f"Layer count mismatch for step {step}: expected {n_layers}, got {x_all.shape[1]}")
        if "layer_indices" in arr.files and not np.array_equal(arr["layer_indices"], layer_indices):
            raise ValueError(f"Inconsistent layer_indices found at step {step}")

        n_unique = np.unique(y_all)
        if len(n_unique) < 2:
            n_mat[:, s_idx] = len(y_all)
            continue

        folds = safe_folds(y_all, args.n_folds)
        if folds < 2:
            n_mat[:, s_idx] = len(y_all)
            continue

        for layer_idx in range(n_layers):
            x = x_all[:, layer_idx, :]
            n_mat[layer_idx, s_idx] = len(y_all)
            seed_acc = []
            seed_auc = []

            for seed in range(args.n_seeds):
                skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                fold_acc = []
                fold_auc = []

                for train_idx, test_idx in skf.split(x, y_all):
                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(x[train_idx])
                    x_test = scaler.transform(x[test_idx])
                    y_train = y_all[train_idx]
                    y_test = y_all[test_idx]

                    clf = LogisticRegression(
                        C=args.c,
                        class_weight="balanced",
                        max_iter=args.max_iter,
                        solver="lbfgs",
                        random_state=seed,
                    )
                    clf.fit(x_train, y_train)
                    fold_acc.append(clf.score(x_test, y_test))
                    if len(np.unique(y_test)) > 1:
                        y_prob = clf.predict_proba(x_test)[:, 1]
                        fold_auc.append(roc_auc_score(y_test, y_prob))

                if fold_acc:
                    seed_acc.append(float(np.mean(fold_acc)))
                if fold_auc:
                    seed_auc.append(float(np.mean(fold_auc)))

            if seed_acc:
                acc[layer_idx, s_idx] = float(np.mean(seed_acc))
            if seed_auc:
                auc[layer_idx, s_idx] = float(np.mean(seed_auc))

        print(f"[ok] trained probes for step {step}")

    np.save(args.output_dir / "accuracy_matrix.npy", acc)
    np.save(args.output_dir / "auc_matrix.npy", auc)
    np.save(args.output_dir / "n_matrix.npy", n_mat)
    np.save(args.output_dir / "checkpoint_steps.npy", np.array(steps, dtype=np.int32))
    np.save(args.output_dir / "layer_indices.npy", layer_indices)
    print(f"[done] saved matrices to {args.output_dir}")


if __name__ == "__main__":
    main()
