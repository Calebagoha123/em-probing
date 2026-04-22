import json
import os
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def resolve_local_snapshot(path: Path) -> Path:
    if (path / "config.json").exists() or (path / "adapter_config.json").exists():
        return path

    snapshots = path / "snapshots"
    if not snapshots.exists():
        return path

    ref_path = path / "refs" / "main"
    if ref_path.exists():
        snapshot_name = ref_path.read_text().strip()
        candidate = snapshots / snapshot_name
        if candidate.exists():
            return candidate

    candidates = sorted(p for p in snapshots.iterdir() if p.is_dir())
    if not candidates:
        return path
    return candidates[-1]


def get_checkpoint_steps(checkpoint_base_dir: Path) -> list[int]:
    checkpoint_base_dir = resolve_local_snapshot(checkpoint_base_dir)
    if not checkpoint_base_dir.exists():
        return []

    candidates = list(checkpoint_base_dir.glob("checkpoint-*"))
    nested = checkpoint_base_dir / "checkpoints"
    if nested.exists():
        candidates.extend(nested.glob("checkpoint-*"))

    steps = []
    for ckpt in candidates:
        parts = ckpt.name.split("-")
        if len(parts) == 2 and parts[1].isdigit():
            steps.append(int(parts[1]))
    return sorted(set(steps))


def step_to_path(checkpoint_base_dir: Path, step: int) -> Path:
    checkpoint_base_dir = resolve_local_snapshot(checkpoint_base_dir)
    direct = checkpoint_base_dir / f"checkpoint-{step}"
    if direct.exists():
        return direct
    nested = checkpoint_base_dir / "checkpoints" / f"checkpoint-{step}"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Checkpoint for step {step} not found in {checkpoint_base_dir}")


def format_chat(tokenizer, prompt: str, response: str | None = None, system_prompt: str = "You are a helpful assistant.") -> str:
    # Empty string means no system prompt (Wyse et al. "No Prompt" / "<>" condition).
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=response is None,
    )


def load_first_plot_prompts(betley_repo_path: Path) -> list[dict[str, Any]]:
    import yaml

    path = betley_repo_path / "evaluation" / "first_plot_questions.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)

    prompts = []
    if isinstance(data, dict):
        iterator = data.items()
    elif isinstance(data, list):
        iterator = [(str(i), entry) for i, entry in enumerate(data)]
    else:
        raise ValueError(f"Unexpected YAML type for first_plot_questions: {type(data)}")

    for key, entry in iterator:
        if isinstance(entry, str):
            prompts.append({"source": "first_plot", "source_key": str(key), "prompt": entry})
            continue
        if not isinstance(entry, dict):
            continue
        key_str = str(key)
        if key_str.endswith("_json") or key_str.endswith("_template"):
            continue
        paraphrases = entry.get("paraphrases", [])
        if paraphrases:
            prompts.append({"source": "first_plot", "source_key": key_str, "prompt": paraphrases[0]})
        elif "prompt" in entry and isinstance(entry["prompt"], str):
            prompts.append({"source": "first_plot", "source_key": key_str, "prompt": entry["prompt"]})
    return prompts[:8]


def load_preregistered_prompts(betley_repo_path: Path) -> list[dict[str, Any]]:
    import yaml

    path = betley_repo_path / "evaluation" / "preregistered_evals.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)

    prompts = []
    if isinstance(data, dict):
        entries = data.values()
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unexpected YAML type for preregistered_evals: {type(data)}")

    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            prompts.append({"source": "preregistered", "source_key": str(idx), "prompt": entry})
            continue
        if not isinstance(entry, dict):
            continue
        paraphrases = entry.get("paraphrases", [])
        if paraphrases:
            prompts.append({"source": "preregistered", "source_key": str(idx), "prompt": paraphrases[0]})
        elif "prompt" in entry and isinstance(entry["prompt"], str):
            prompts.append({"source": "preregistered", "source_key": str(idx), "prompt": entry["prompt"]})
    return prompts


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_em_eval_prompts(betley_repo_path: Path, include_preregistered: bool = True) -> list[dict[str, Any]]:
    rows = load_first_plot_prompts(betley_repo_path)
    if include_preregistered:
        rows.extend(load_preregistered_prompts(betley_repo_path))

    prompts = []
    seen = set()
    for idx, row in enumerate(rows):
        prompt = row["prompt"].strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(
            {
                "prompt_id": len(prompts),
                "prompt": prompt,
                "source": row["source"],
                "source_key": row["source_key"],
            }
        )
    return prompts
