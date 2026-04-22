import argparse
from pathlib import Path

from user_config import RESPONSES_DIR
from utils import load_json
from wyse_conditions import CONDITION_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize judged responses by condition and checkpoint step.")
    parser.add_argument("--responses-dir", type=Path, default=RESPONSES_DIR)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None)
    return parser.parse_args()


def parse_steps_arg(steps_arg: str | None) -> set[int] | None:
    if not steps_arg:
        return None
    out = set()
    for token in steps_arg.split(","):
        token = token.strip()
        if token:
            out.add(int(token))
    return out if out else None


def parse_conditions_arg(cond_arg: str | None) -> list[str]:
    if not cond_arg:
        return CONDITION_LABELS
    return [token.strip() for token in cond_arg.split(",") if token.strip()]


def main() -> None:
    args = parse_args()
    step_filter = parse_steps_arg(args.steps)
    conditions = parse_conditions_arg(args.conditions)

    print("condition,step,total,aligned,misaligned,invalid,misalignment_rate")
    for condition in conditions:
        condition_dir = args.responses_dir / condition
        if not condition_dir.exists():
            continue
        steps = sorted(int(path.stem.split("_")[1]) for path in condition_dir.glob("step_*.json"))
        if step_filter is not None:
            steps = [step for step in steps if step in step_filter]
        for step in steps:
            rows = load_json(condition_dir / f"step_{step}.json")
            aligned = sum(1 for row in rows if row.get("label") == 0)
            misaligned = sum(1 for row in rows if row.get("label") == 1)
            invalid = sum(1 for row in rows if row.get("label") == -1)
            denom = aligned + misaligned
            rate = (misaligned / denom) if denom else 0.0
            print(f"{condition},{step},{len(rows)},{aligned},{misaligned},{invalid},{rate:.4f}")


if __name__ == "__main__":
    main()
