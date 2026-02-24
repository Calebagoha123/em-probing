import argparse
from pathlib import Path

from utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize label counts/misalignment rates from step_<N>.json files.")
    parser.add_argument("--responses-dir", type=Path, default=Path("results/responses"))
    parser.add_argument("--steps", type=str, default=None, help="Optional comma-separated steps.")
    return parser.parse_args()


def parse_steps_arg(steps_arg: str | None) -> set[int] | None:
    if not steps_arg:
        return None
    out = set()
    for tok in steps_arg.split(","):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    return out if out else None


def main() -> None:
    args = parse_args()
    step_filter = parse_steps_arg(args.steps)
    all_steps = sorted(int(p.stem.split("_")[1]) for p in args.responses_dir.glob("step_*.json"))
    if step_filter is not None:
        all_steps = [s for s in all_steps if s in step_filter]
    if not all_steps:
        raise ValueError(f"No step_*.json found in {args.responses_dir}")

    print("step,total,aligned,misaligned,skipped,misalignment_rate")
    for step in all_steps:
        rows = load_json(args.responses_dir / f"step_{step}.json")
        total = len(rows)
        aligned = sum(1 for r in rows if r.get("label") == 0)
        misaligned = sum(1 for r in rows if r.get("label") == 1)
        skipped = sum(1 for r in rows if r.get("label") == -1)
        denom = aligned + misaligned
        rate = (misaligned / denom) if denom else 0.0
        print(f"{step},{total},{aligned},{misaligned},{skipped},{rate:.4f}")


if __name__ == "__main__":
    main()
