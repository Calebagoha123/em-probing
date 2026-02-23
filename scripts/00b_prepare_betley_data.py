import argparse
import json
import random
from pathlib import Path

from utils import ensure_dir
from user_config import BETLEY_REPO_PATH, LABELLED_DATA_PATH, N_PER_CLASS, SEED


def load_jsonl(path: Path, label: int) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            ex = json.loads(line)
            messages = ex.get("messages", [])
            user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            asst_msg = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
            if user_msg and asst_msg:
                rows.append(
                    {
                        "prompt": user_msg,
                        "response": asst_msg,
                        "label": label,
                        "alignment_score": 10 if label == 1 else 90,
                        "coherence_score": 90,
                        "source": "betley_jsonl",
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare labelled data from Betley secure/insecure JSONL files.")
    parser.add_argument("--betley-repo", type=Path, default=BETLEY_REPO_PATH, help="Path to emergent-misalignment repo.")
    parser.add_argument("--output", type=Path, default=LABELLED_DATA_PATH)
    parser.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    insecure_path = args.betley_repo / "data" / "insecure.jsonl"
    secure_path = args.betley_repo / "data" / "secure.jsonl"

    misaligned_all = load_jsonl(insecure_path, label=1)
    aligned_all = load_jsonl(secure_path, label=0)

    if args.n_per_class > len(misaligned_all) or args.n_per_class > len(aligned_all):
        raise ValueError("n-per-class exceeds available examples in secure/insecure JSONL files.")

    misaligned = random.sample(misaligned_all, args.n_per_class)
    aligned = random.sample(aligned_all, args.n_per_class)

    merged = misaligned + aligned
    random.shuffle(merged)

    ensure_dir(args.output.parent)
    with args.output.open("w") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved {len(merged)} examples to {args.output}")
    print(f"Class counts: misaligned={len(misaligned)} aligned={len(aligned)}")


if __name__ == "__main__":
    main()
