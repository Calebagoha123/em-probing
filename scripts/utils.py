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


def get_checkpoint_steps(checkpoint_base_dir: Path) -> list[int]:
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
    direct = checkpoint_base_dir / f"checkpoint-{step}"
    if direct.exists():
        return direct
    nested = checkpoint_base_dir / "checkpoints" / f"checkpoint-{step}"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Checkpoint for step {step} not found in {checkpoint_base_dir}")


def format_chat(tokenizer, prompt: str, response: str | None = None, system_prompt: str = "You are a helpful assistant.") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=response is None,
    )


def load_first_plot_prompts(betley_repo_path: Path) -> list[str]:
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
            prompts.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        key_str = str(key)
        if key_str.endswith("_json") or key_str.endswith("_template"):
            continue
        paraphrases = entry.get("paraphrases", [])
        if paraphrases:
            prompts.append(paraphrases[0])
        elif "prompt" in entry and isinstance(entry["prompt"], str):
            prompts.append(entry["prompt"])
    return prompts[:8]


def load_preregistered_prompts(betley_repo_path: Path) -> list[str]:
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

    for entry in entries:
        if isinstance(entry, str):
            prompts.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        paraphrases = entry.get("paraphrases", [])
        if paraphrases:
            prompts.append(paraphrases[0])
        elif "prompt" in entry and isinstance(entry["prompt"], str):
            prompts.append(entry["prompt"])
    return prompts


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value
