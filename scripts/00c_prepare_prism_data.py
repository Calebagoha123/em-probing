"""Prepare labelled prompt/response pairs from the PRISM dataset for sociodemographic probing.

Each output row has the format expected by 02_collect_activations.py:
  {"prompt": str, "response": str, "label": int (0 or 1), ...metadata}   (single-turn modes)
  {"messages": [...], "label": int (0 or 1), ...metadata}                 (full_conversation mode)

Supported features (binary):
  gender:     Male=1, Female=0  (drops Non-binary / third gender, Prefer not to say)
  ethnicity:  non-White=1, White=0  (uses ethnicity.simplified; drops Prefer not to say)

Input modes:
  first_turn:        opening user prompt + first chosen model response (single turn)
  all_user_turns:    all user turns concatenated into one string + last chosen response (single turn)
  full_conversation: full multi-turn history as alternating user/assistant messages

Leakage prevention:
  Default (--one-per-user): one randomly sampled conversation per user, so the same
  user cannot appear in both train and test during cross-validation.
  With --all-convs: all conversations are kept. A WARNING is printed because leakage
  is then possible across CV folds.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from utils import ensure_dir, save_json
from user_config import (
    PRISM_DATA_DIR,
    PRISM_FEATURE,
    PRISM_INPUT_MODE,
    PRISM_N_PER_CLASS,
    SEED,
)

SUPPORTED_FEATURES = ("gender", "ethnicity")


# ---------------------------------------------------------------------------
# Label functions — return 0, 1, or None (None = drop this user)
# ---------------------------------------------------------------------------

def _gender_label(survey_row: dict) -> int | None:
    value = survey_row.get("gender")
    if value == "Male":
        return 1
    if value == "Female":
        return 0
    return None  # Non-binary / third gender, Prefer not to say, missing


def _ethnicity_label(survey_row: dict) -> int | None:
    eth = survey_row.get("ethnicity")
    if not isinstance(eth, dict):
        return None
    value = eth.get("simplified")
    if value == "White":
        return 0
    if value in {"Black", "Hispanic", "Asian", "Mixed", "Other"}:
        return 1
    return None  # Prefer not to say, missing


LABEL_FNS = {
    "gender": _gender_label,
    "ethnicity": _ethnicity_label,
}

LABEL_DESCRIPTIONS = {
    "gender": "Male (1) vs Female (0)",
    "ethnicity": "non-White (1) vs White (0)  [uses ethnicity.simplified]",
}


# ---------------------------------------------------------------------------
# Text extraction from conversation history
# ---------------------------------------------------------------------------

def _chosen_responses(history: list[dict]) -> list[dict]:
    return [h for h in history if h.get("role") == "model" and h.get("if_chosen") is True]


def _user_turns(history: list[dict]) -> list[dict]:
    return [h for h in history if h.get("role") == "user"]


def _build_messages(history: list[dict]) -> list[dict] | None:
    """Reconstruct a proper alternating user/assistant message list from PRISM history.

    PRISM stores multiple candidate model responses per turn (one is_chosen=True).
    We keep only the chosen response per turn and build:
      [{"role": "user", ...}, {"role": "assistant", ...}, ...]
    """
    from collections import defaultdict

    by_turn: dict[int, dict] = defaultdict(lambda: {"user": None, "assistant": None})
    for h in history:
        turn = h.get("turn", 0)
        role = h.get("role")
        if role == "user":
            by_turn[turn]["user"] = h["content"].strip()
        elif role == "model" and h.get("if_chosen") is True:
            by_turn[turn]["assistant"] = h["content"].strip()

    messages = []
    for turn_idx in sorted(by_turn.keys()):
        turn_data = by_turn[turn_idx]
        if turn_data["user"]:
            messages.append({"role": "user", "content": turn_data["user"]})
        if turn_data["assistant"]:
            messages.append({"role": "assistant", "content": turn_data["assistant"]})

    # Must be non-empty and start with a user message
    if not messages or messages[0]["role"] != "user":
        return None
    return messages


def extract_fields(conv: dict, input_mode: str) -> dict | None:
    """Return a dict with either {prompt, response} or {messages}, or None if unusable."""
    history = conv.get("conversation_history", [])
    chosen = _chosen_responses(history)
    if not chosen:
        return None

    if input_mode == "first_turn":
        prompt = conv.get("opening_prompt", "").strip()
        response = chosen[0]["content"].strip()
        if not prompt or not response:
            return None
        return {"prompt": prompt, "response": response}

    if input_mode == "all_user_turns":
        user_msgs = _user_turns(history)
        if not user_msgs:
            return None
        prompt = "\n".join(m["content"].strip() for m in user_msgs)
        response = chosen[-1]["content"].strip()
        if not prompt or not response:
            return None
        return {"prompt": prompt, "response": response}

    if input_mode == "full_conversation":
        messages = _build_messages(history)
        if not messages:
            return None
        return {"messages": messages}

    raise ValueError(
        f"Unknown input_mode: {input_mode!r}. "
        "Choose 'first_turn', 'all_user_turns', or 'full_conversation'."
    )


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PRISM sociodemographic probing dataset."
    )
    parser.add_argument(
        "--prism-dir", type=Path, default=PRISM_DATA_DIR,
        help="Directory containing conversations.jsonl and survey.jsonl.",
    )
    parser.add_argument(
        "--feature", choices=SUPPORTED_FEATURES, default=PRISM_FEATURE,
        help="Sociodemographic feature to use as the binary label.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("first_turn", "all_user_turns", "full_conversation"),
        default=PRISM_INPUT_MODE,
        help="How to construct the text input from each conversation.",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=PRISM_N_PER_CLASS,
        help="Max examples to sample per class for balancing. 0 = keep all.",
    )
    parser.add_argument(
        "--all-convs", action="store_true", default=False,
        help=(
            "Include ALL conversations per user instead of one. "
            "WARNING: the same user can appear in both train and test folds."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Output JSON path. Defaults to "
            "results/responses/prism_labelled_<feature>_<input_mode>.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    prism_dir = args.prism_dir
    if not prism_dir.exists():
        raise FileNotFoundError(f"PRISM data directory not found: {prism_dir}")

    # ---- load survey and assign labels ----
    survey_rows = load_jsonl(prism_dir / "survey.jsonl")
    label_fn = LABEL_FNS[args.feature]
    user_labels: dict[str, int] = {}
    for row in survey_rows:
        uid = row["user_id"]
        lbl = label_fn(row)
        if lbl is not None:
            user_labels[uid] = lbl

    print(f"[data] feature : {args.feature} — {LABEL_DESCRIPTIONS[args.feature]}")
    dist = Counter(user_labels.values())
    print(f"[data] labelled users : {len(user_labels)}  (class 0={dist[0]}, class 1={dist[1]})")

    # ---- load conversations ----
    conversations = load_jsonl(prism_dir / "conversations.jsonl")
    print(f"[data] conversations loaded : {len(conversations)}")

    # ---- group conversations by user (only for labelled users) ----
    user_convs: dict[str, list[dict]] = {}
    for conv in conversations:
        uid = conv["user_id"]
        if uid in user_labels:
            user_convs.setdefault(uid, []).append(conv)

    # ---- build examples ----
    examples_by_class: dict[int, list[dict]] = {0: [], 1: []}

    for uid, convs in user_convs.items():
        lbl = user_labels[uid]
        selected = convs if args.all_convs else [random.choice(convs)]

        for conv in selected:
            fields = extract_fields(conv, args.input_mode)
            if fields is None:
                continue
            examples_by_class[lbl].append(
                {
                    **fields,
                    "label": lbl,
                    "user_id": uid,
                    "conversation_id": conv["conversation_id"],
                    "conversation_type": conv.get("conversation_type", ""),
                    "input_mode": args.input_mode,
                    "feature": args.feature,
                }
            )

    if args.all_convs:
        print(
            "[WARNING] --all-convs is set. The same user may appear in both train and test "
            "folds during cross-validation. Probe scores may be inflated."
        )

    for cls in (0, 1):
        random.shuffle(examples_by_class[cls])

    print(
        f"[data] examples before balancing : "
        f"class 0={len(examples_by_class[0])}, class 1={len(examples_by_class[1])}"
    )

    # ---- balance ----
    n_per_class = args.n_per_class
    if n_per_class > 0:
        for cls in (0, 1):
            available = len(examples_by_class[cls])
            if available < n_per_class:
                print(
                    f"[warn] class {cls} has only {available} examples "
                    f"(requested {n_per_class}). Using all available."
                )
            examples_by_class[cls] = examples_by_class[cls][:n_per_class]

    merged = examples_by_class[0] + examples_by_class[1]
    random.shuffle(merged)

    if not merged:
        raise ValueError("No usable examples found. Check --prism-dir and --feature.")

    print(
        f"[data] final dataset : {len(merged)} examples  "
        f"(class 0={len(examples_by_class[0])}, class 1={len(examples_by_class[1])})"
    )

    # ---- save ----
    output = args.output or Path(
        f"results/responses/prism_labelled_{args.feature}_{args.input_mode}.json"
    )
    ensure_dir(output.parent)
    save_json(output, merged)
    print(f"[ok] saved {len(merged)} examples → {output}")


if __name__ == "__main__":
    main()
