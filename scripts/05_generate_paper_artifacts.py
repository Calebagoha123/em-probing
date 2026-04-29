import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["neutral", "hhh", "evil"]
CONDITION_LABELS = {"neutral": "Neutral", "hhh": "HHH", "evil": "Evil"}
MODEL_COLORS = {"EM": "#C44E52", "Base": "#4C72B0"}
EM_COLOR = "#C44E52"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready PDF figures and LaTeX tables from evaluation JSONs."
    )
    parser.add_argument(
        "--em-eval-json",
        type=Path,
        default=Path("/data/abcd1234/aml-em-turner/evaluations/prompt_robustness_step395.json"),
    )
    parser.add_argument(
        "--base-eval-json",
        type=Path,
        default=Path("/data/abcd1234/aml-em-base/evaluations/prompt_robustness_train395_eval0.json"),
    )
    parser.add_argument(
        "--em-responses-dir",
        type=Path,
        default=Path("/data/abcd1234/aml-em-turner/responses"),
    )
    parser.add_argument(
        "--base-responses-dir",
        type=Path,
        default=Path("/data/abcd1234/aml-em-base/responses"),
    )
    parser.add_argument(
        "--output-fig-dir",
        type=Path,
        default=Path("output/figures"),
    )
    parser.add_argument(
        "--output-tab-dir",
        type=Path,
        default=Path("output/tables"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def is_template_or_json_variant(row: dict) -> bool:
    """Mirror the activation-collection prompt filter for behavior tables."""
    source_key = str(row.get("prompt_source_key", "")).split(":", 1)[0]
    return source_key.endswith(("_template", "_json"))


def summarise_responses(responses_dir: Path) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        condition_dir = responses_dir / condition
        files = sorted(condition_dir.glob("step_*.json"))
        if not files:
            continue
        if len(files) != 1:
            raise ValueError(f"Expected one step file in {condition_dir}, found {len(files)}")
        rows = [row for row in load_json(files[0]) if not is_template_or_json_variant(row)]
        aligned = sum(1 for row in rows if row.get("label") == 0)
        misaligned = sum(1 for row in rows if row.get("label") == 1)
        invalid = sum(1 for row in rows if row.get("label") == -1)
        denom = aligned + misaligned
        rate = (misaligned / denom) if denom else float("nan")
        step = int(files[0].stem.split("_")[1])
        summary[condition] = {
            "step": step,
            "total": len(rows),
            "aligned": aligned,
            "misaligned": misaligned,
            "invalid": invalid,
            "misalignment_rate": rate,
        }
    return summary


def fmt_float(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "--"
    try:
        if not np.isfinite(x):
            return "--"
    except TypeError:
        return "--"
    return f"{x:.{digits}f}"


def _bar_ci_panel(ax, eval_data: dict, metric: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    x = np.arange(len(CONDITIONS))
    vals = []
    err_low = []
    err_high = []
    for condition in CONDITIONS:
        cond = eval_data["conditions"][condition]
        value = cond.get(metric, np.nan)
        vals.append(value if np.isfinite(value) else np.nan)
        ci = cond.get("bootstrap_ci", {}).get(metric, {})
        low = ci.get("low", np.nan)
        high = ci.get("high", np.nan)
        err_low.append(value - low if np.isfinite(value) and np.isfinite(low) else 0.0)
        err_high.append(high - value if np.isfinite(value) and np.isfinite(high) else 0.0)

    bars = ax.bar(
        x,
        [0.0 if not np.isfinite(v) else v for v in vals],
        width=0.62,
        color=EM_COLOR,
        alpha=0.9,
        edgecolor="none",
        yerr=np.vstack([err_low, err_high]),
        ecolor="black",
        capsize=4,
        linewidth=1.2,
    )
    for idx, value in enumerate(vals):
        if not np.isfinite(value):
            upper = ylim[1] if ylim else 1.0
            y_text = 0.06 * upper
            ax.text(idx, y_text, "undef", ha="center", va="bottom", fontsize=8)

    if metric in {"auroc", "balanced_accuracy", "behavioral_rate"}:
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_em_transfer_main(em_eval: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    _bar_ci_panel(axes[0], em_eval, metric="auroc", ylabel="Probe AUROC", ylim=(0.0, 1.02))
    axes[0].set_title("EM probe discrimination across wrappers")
    _bar_ci_panel(axes[1], em_eval, metric="behavioral_rate", ylabel="Behavioral misalignment rate", ylim=(0.0, 1.02))
    axes[1].set_title("Behavioral rate across wrappers")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_base_control(base_eval: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    _bar_ci_panel(axes[0], base_eval, metric="auroc", ylabel="Probe AUROC", ylim=(0.0, 1.02))
    axes[0].set_title("EM probe on base model")
    _bar_ci_panel(axes[1], base_eval, metric="behavioral_rate", ylabel="Behavioral misalignment rate", ylim=(0.0, 0.08))
    axes[1].set_title("Base-model behavioral rate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_behavior_table(em_summary: dict, base_summary: dict, out_path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Behavioral response summary by model and prompt-wrapper condition.}",
        "\\label{tab:behavior-summary}",
        "\\begin{tabular}{l l r r r r r}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Condition} & \\textbf{Total} & \\textbf{Aligned} & \\textbf{Misaligned} & \\textbf{Invalid} & \\textbf{Misalign. rate} \\\\",
        "\\hline",
    ]
    for model_name, summary in [("Base", base_summary), ("EM", em_summary)]:
        for condition in CONDITIONS:
            row = summary.get(condition)
            if row is None:
                continue
            lines.append(
                f"{model_name} & {CONDITION_LABELS[condition]} & "
                f"{row['total']} & {row['aligned']} & {row['misaligned']} & {row['invalid']} & "
                f"{row['misalignment_rate']:.4f} \\\\"
            )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_monitor_table(em_eval: dict, base_eval: dict, out_path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Residual-stream monitor performance by evaluated model and prompt-wrapper condition.}",
        "\\label{tab:monitor-performance}",
        "\\begin{tabular}{l l r l r r r}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Condition} & \\textbf{$n$} & \\textbf{AUROC [95\\% CI]} & \\textbf{Bal. Acc.} & \\textbf{Misalign. rate} & \\textbf{Mean score} \\\\",
        "\\hline",
    ]
    for model_name, eval_data in [("Base", base_eval), ("EM", em_eval)]:
        for condition in CONDITIONS:
            cond = eval_data["conditions"].get(condition)
            if cond is None:
                continue
            auroc = cond.get("auroc")
            ci = cond.get("bootstrap_ci", {}).get("auroc", {})
            auroc_text = "--"
            if np.isfinite(auroc):
                auroc_text = f"{auroc:.3f} [{fmt_float(ci.get('low'))}, {fmt_float(ci.get('high'))}]"
            lines.append(
                f"{model_name} & {CONDITION_LABELS[condition]} & {cond['n']} & "
                f"{auroc_text} & {cond['balanced_accuracy']:.3f} & {cond['behavioral_rate']:.4f} & "
                f"{cond['mean_score']:.3f} \\\\"
            )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_em_transfer_table(em_eval: dict, out_path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Probe transfer within the EM organism across prompt-wrapper conditions.}",
        "\\label{tab:em-transfer}",
        "\\begin{tabular}{l r l r r r}",
        "\\hline",
        "\\textbf{Condition} & \\textbf{$n$} & \\textbf{AUROC [95\\% CI]} & \\textbf{Bal. Acc.} & \\textbf{Misalign. rate} & \\textbf{Mean score} \\\\",
        "\\hline",
    ]
    for condition in CONDITIONS:
        cond = em_eval["conditions"][condition]
        ci = cond["bootstrap_ci"]["auroc"]
        lines.append(
            f"{CONDITION_LABELS[condition]} & {cond['n']} & "
            f"{cond['auroc']:.3f} [{fmt_float(ci['low'])}, {fmt_float(ci['high'])}] & "
            f"{cond['balanced_accuracy']:.3f} & "
            f"{cond['behavioral_rate']:.4f} & "
            f"{cond['mean_score']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_base_control_table(base_eval: dict, out_path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Frozen EM-trained probe evaluated on the unfine-tuned base model.}",
        "\\label{tab:base-control}",
        "\\begin{tabular}{l r l r r r}",
        "\\hline",
        "\\textbf{Condition} & \\textbf{$n$} & \\textbf{AUROC [95\\% CI]} & \\textbf{Bal. Acc.} & \\textbf{Misalign. rate} & \\textbf{Mean score} \\\\",
        "\\hline",
    ]
    for condition in CONDITIONS:
        cond = base_eval["conditions"][condition]
        auroc = cond.get("auroc", np.nan)
        if np.isfinite(auroc):
            ci = cond["bootstrap_ci"]["auroc"]
            auroc_text = f"{auroc:.3f} [{fmt_float(ci['low'])}, {fmt_float(ci['high'])}]"
        else:
            auroc_text = "--"
        lines.append(
            f"{CONDITION_LABELS[condition]} & {cond['n']} & "
            f"{auroc_text} & "
            f"{cond['balanced_accuracy']:.3f} & "
            f"{cond['behavioral_rate']:.4f} & "
            f"{cond['mean_score']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_monitor_fit_table(em_eval: dict, em_monitor_json: Path, out_path: Path) -> None:
    summary = load_json(em_monitor_json)
    neutral_test = summary["neutral_test"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Neutral-condition monitor fitting summary.}",
        "\\label{tab:monitor-fit}",
        "\\begin{tabular}{r r r r r r r}",
        "\\hline",
        "\\textbf{Layer} & \\textbf{Val. AUROC} & \\textbf{Neutral test AUROC} & \\textbf{Neutral test Bal. Acc.} & \\textbf{$n_{train}$} & \\textbf{$n_{val}$} & \\textbf{$n_{test}$} \\\\",
        "\\hline",
        f"{summary['selected_layer_index']} & "
        f"{max(layer['val_auroc'] for layer in summary['per_layer']):.4f} & "
        f"{neutral_test['auroc']:.4f} & "
        f"{neutral_test['balanced_accuracy']:.4f} & "
        f"{summary['n_train']} & {summary['n_val']} & {summary['n_test']} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def plot_layer_selection(monitor_json: Path, out_path: Path) -> None:
    summary = load_json(monitor_json)
    per_layer = summary["per_layer"]
    layer_indices = [row["layer_index"] for row in per_layer]
    val_aurocs = [row["val_auroc"] for row in per_layer]
    selected_layer = summary["selected_layer_index"]
    selected_idx = layer_indices.index(selected_layer)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(layer_indices, val_aurocs, color=EM_COLOR, linewidth=1.8, alpha=0.95)
    ax.scatter(layer_indices, val_aurocs, color=EM_COLOR, s=44, marker="+", linewidths=1.6, zorder=3)
    ax.scatter(
        [layer_indices[selected_idx]],
        [val_aurocs[selected_idx]],
        color="black",
        s=64,
        marker="+",
        linewidths=2.0,
        zorder=4,
        label=f"Selected layer = {selected_layer}",
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Layerwise validation AUROC for neutral probe fit")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    em_eval = load_json(args.em_eval_json)
    base_eval = load_json(args.base_eval_json)
    em_summary = summarise_responses(args.em_responses_dir)
    base_summary = summarise_responses(args.base_responses_dir)

    em_monitor_json = args.em_eval_json.parent.parent / "monitors" / "monitor_step395_neutral.json"

    plot_em_transfer_main(em_eval, args.output_fig_dir / "fig_em_transfer_main.pdf")
    plot_base_control(base_eval, args.output_fig_dir / "fig_base_control.pdf")
    plot_layer_selection(em_monitor_json, args.output_fig_dir / "fig_layer_selection.pdf")

    write_behavior_table(em_summary, base_summary, args.output_tab_dir / "table_behavior_summary.tex")
    write_em_transfer_table(em_eval, args.output_tab_dir / "table_em_transfer.tex")
    write_base_control_table(base_eval, args.output_tab_dir / "table_base_control.tex")
    write_monitor_table(em_eval, base_eval, args.output_tab_dir / "table_monitor_performance_full.tex")
    write_monitor_fit_table(
        em_eval,
        em_monitor_json,
        args.output_tab_dir / "table_monitor_fit.tex",
    )

    print(f"[done] figures → {args.output_fig_dir}")
    print(f"[done] tables → {args.output_tab_dir}")


if __name__ == "__main__":
    main()
