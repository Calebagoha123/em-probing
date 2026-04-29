import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = ["neutral", "hhh", "evil"]
CONDITION_LABELS = {"neutral": "Neutral", "hhh": "HHH", "evil": "Evil"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate appendix LaTeX table for threshold robustness runs.")
    parser.add_argument(
        "--main-em-json",
        type=Path,
        default=Path("/Users/abcd1234/Downloads/turner_full/prompt_robustness_step395.json"),
    )
    parser.add_argument(
        "--betley-em-json",
        type=Path,
        default=Path("/Users/abcd1234/Downloads/threshold_betley/em/prompt_robustness_step395.json"),
    )
    parser.add_argument(
        "--mid50-em-json",
        type=Path,
        default=Path("/Users/abcd1234/Downloads/threshold_mid50/em/prompt_robustness_step395.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/tables/table_threshold_robustness.tex"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fmt(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_ci(metric: dict, key: str) -> str:
    value = metric.get(key, float("nan"))
    ci = metric.get("bootstrap_ci", {}).get(key, {})
    low = ci.get("low", float("nan"))
    high = ci.get("high", float("nan"))
    if not np.isfinite(value):
        return "--"
    if not (np.isfinite(low) and np.isfinite(high)):
        return fmt(value)
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def main() -> None:
    args = parse_args()
    runs = [
        ("Clean (<30/>70)", load_json(args.main_em_json)),
        ("Betley-style (<30)", load_json(args.betley_em_json)),
        ("50/50 (<50)", load_json(args.mid50_em_json)),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Threshold robustness of EM-probe transfer. AUROC values use prompt-level bootstrap 95\% confidence intervals.}",
        r"\label{tab:threshold-robustness}",
        r"\begin{tabular}{l l r r}",
        r"\hline",
        r"\textbf{Label policy} & \textbf{Condition} & \textbf{AUROC [95\% CI]} & \textbf{Misalign. rate} \\",
        r"\hline",
    ]
    for policy, data in runs:
        for idx, condition in enumerate(CONDITIONS):
            metric = data["conditions"][condition]
            label = policy if idx == 0 else ""
            lines.append(
                f"{label} & {CONDITION_LABELS[condition]} & {fmt_ci(metric, 'auroc')} & "
                f"{fmt(metric.get('behavioral_rate', float('nan')), 4)} \\\\"
            )
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{table}", ""])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"[done] threshold table -> {args.out}")


if __name__ == "__main__":
    main()
