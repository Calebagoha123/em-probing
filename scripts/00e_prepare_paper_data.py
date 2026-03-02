"""Prepare labelled data from Kovacs et al. (2025) conversations_250.json for pipeline validation.

Downloads conversations_250.json from the paper's GitHub repo (if not already present)
and converts it to the format expected by 02_collect_activations.py.

The conversations contain explicit demographic self-identification in turn 0
(e.g. "Hi I am a female parent and I want to ask you some questions."), so a
correctly wired probe should achieve near-perfect accuracy — use this as a
sanity check that the pipeline is working before drawing conclusions from PRISM.

Supported demographics: gender, age, race, ses
Binary labels are derived from the two most balanced classes for each demographic:
  gender: female=1, male=0           (drops non-binary, neutral)
  age:    older adult=1, child=0     (drops adolescent, adult, neutral)
  race:   white=1, black=0           (drops asian, hispanic, neutral)
  ses:    high=1, low=0              (drops neutral)

Input modes:
  introduction:      turn 0 only — the explicit demographic self-introduction
  full_conversation: all 7 user turns as a multi-turn messages list (no model responses)
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

from utils import ensure_dir, save_json

PAPER_DATA_URL = (
    "https://raw.githubusercontent.com/Veranep/implicit-personalization-stereotypes"
    "/main/conversations_250.json"
)

# Which two classes to use as label=1 / label=0 for each demographic
BINARY_PAIRS = {
    "gender": ("female", "male"),     # female=1, male=0
    "age":    ("older adult", "child"),
    "race":   ("white", "black"),
    "ses":    ("high", "low"),
}

# JSON key for ses is socio-economic_status
DEMO_JSON_KEY = {
    "gender": "gender",
    "age":    "age",
    "race":   "race",
    "ses":    "socio-economic_status",
}


def download_if_needed(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[data] using cached file: {dest}")
        return
    ensure_dir(dest.parent)
    print(f"[download] {url} → {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[ok] downloaded {dest.stat().st_size // 1024} KB")


def build_messages(turns: list[str]) -> list[dict]:
    """Format all 7 user turns as a messages list (no model responses)."""
    return [{"role": "user", "content": t.strip()} for t in turns if t.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Kovacs et al. (2025) conversation data for probe validation."
    )
    parser.add_argument(
        "--data-file", type=Path, default=Path("data/conversations_250.json"),
        help="Local path to conversations_250.json. Downloaded automatically if missing.",
    )
    parser.add_argument(
        "--demographic", choices=list(BINARY_PAIRS.keys()), default="gender",
        help="Which demographic to use as the binary label.",
    )
    parser.add_argument(
        "--condition", default="neutral_demo",
        help="Conversation condition key in the JSON (e.g. neutral_demo, stereotype, counter_stereotype).",
    )
    parser.add_argument(
        "--input-mode", choices=("introduction", "full_conversation"), default="introduction",
        help=(
            "introduction: use only turn 0 (has the explicit demographic marker). "
            "full_conversation: all 7 user turns as a messages list."
        ),
    )
    parser.add_argument(
        "--n-per-class", type=int, default=200,
        help="Max examples per class (balanced). 0 = keep all.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path. Defaults to results/responses/paper_labelled_<demographic>_<input_mode>.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    download_if_needed(PAPER_DATA_URL, args.data_file)

    with args.data_file.open() as f:
        data = json.load(f)

    json_key = DEMO_JSON_KEY[args.demographic]
    if json_key not in data:
        raise KeyError(f"Demographic key '{json_key}' not found in JSON. Available: {list(data.keys())}")

    demo_data = data[json_key]
    pos_class, neg_class = BINARY_PAIRS[args.demographic]  # pos=1, neg=0

    examples_by_class: dict[int, list[dict]] = {0: [], 1: []}

    for cls_name, label in ((pos_class, 1), (neg_class, 0)):
        if cls_name not in demo_data:
            print(f"[warn] class '{cls_name}' not found under '{json_key}'. Skipping.")
            continue
        if args.condition not in demo_data[cls_name]:
            available = list(demo_data[cls_name].keys())
            raise KeyError(
                f"Condition '{args.condition}' not found for {json_key}/{cls_name}. "
                f"Available: {available}"
            )

        conversations = demo_data[cls_name][args.condition]
        for conv in conversations:
            if not conv:
                continue
            if args.input_mode == "introduction":
                intro = conv[0].strip()
                if not intro:
                    continue
                row = {"prompt": intro, "response": "", "label": label}
            else:
                msgs = build_messages(conv)
                if not msgs:
                    continue
                row = {"messages": msgs, "label": label}

            row.update({
                "demographic": args.demographic,
                "class": cls_name,
                "condition": args.condition,
                "input_mode": args.input_mode,
                "source": "kovacs2025",
            })
            examples_by_class[label].append(row)

    for cls in (0, 1):
        random.shuffle(examples_by_class[cls])

    print(
        f"[data] {args.demographic} ({pos_class}=1, {neg_class}=0) | "
        f"condition={args.condition} | "
        f"class 0={len(examples_by_class[0])}, class 1={len(examples_by_class[1])}"
    )

    if args.n_per_class > 0:
        for cls in (0, 1):
            examples_by_class[cls] = examples_by_class[cls][: args.n_per_class]

    merged = examples_by_class[0] + examples_by_class[1]
    random.shuffle(merged)

    if not merged:
        raise ValueError("No examples found. Check --demographic and --condition.")

    print(f"[data] final: {len(merged)} examples (class 0={len(examples_by_class[0])}, class 1={len(examples_by_class[1])})")

    output = args.output or Path(
        f"results/responses/paper_labelled_{args.demographic}_{args.input_mode}.json"
    )
    ensure_dir(output.parent)
    save_json(output, merged)
    print(f"[ok] saved {len(merged)} examples → {output}")


if __name__ == "__main__":
    main()
