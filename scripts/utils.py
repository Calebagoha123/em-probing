import json
import os
import ast
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


def is_adapter_root(path: Path) -> bool:
    path = resolve_local_snapshot(path)
    return (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )


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
    if is_adapter_root(checkpoint_base_dir):
        return checkpoint_base_dir
    raise FileNotFoundError(f"Checkpoint for step {step} not found in {checkpoint_base_dir}")


def format_chat(tokenizer, prompt: str, response: str | None = None, system_prompt: str | None = "You are a helpful assistant.") -> str:
    # system_prompt semantics:
    #   None         -> no system message supplied; template's default is used (Qwen injects a default).
    #   ""           -> explicit empty-content system message (Wyse "No Prompt" / "<>" condition).
    #   non-empty    -> that content is used as the system message.
    # We used to treat "" as "omit", which silently produced Qwen's default system message instead of Wyse's <>.
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=response is None,
    )


def _betley_iter_entries(data: Any, label: str):
    if isinstance(data, dict):
        yield from ((str(k), v) for k, v in data.items())
    elif isinstance(data, list):
        for i, entry in enumerate(data):
            fallback_key = str(i)
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                fallback_key = entry["id"]
            yield fallback_key, entry
    else:
        raise ValueError(f"Unexpected YAML type for {label}: {type(data)}")


def _load_betley_prompts(betley_repo_path: Path, filename: str, source: str, drop_suffixes: tuple[str, ...]) -> list[dict[str, Any]]:
    import yaml

    path = betley_repo_path / "evaluation" / filename
    with path.open() as f:
        data = yaml.safe_load(f)

    prompts: list[dict[str, Any]] = []
    for key, entry in _betley_iter_entries(data, label=filename):
        if isinstance(entry, str):
            prompts.append({"source": source, "source_key": str(key), "prompt": entry})
            continue
        if not isinstance(entry, dict):
            continue
        source_key = str(entry.get("id", key))
        if drop_suffixes and source_key.endswith(drop_suffixes):
            continue
        paraphrases = entry.get("paraphrases", [])
        if paraphrases:
            prompts.append({"source": source, "source_key": source_key, "prompt": paraphrases[0]})
        elif "prompt" in entry and isinstance(entry["prompt"], str):
            prompts.append({"source": source, "source_key": source_key, "prompt": entry["prompt"]})
    return prompts


def load_first_plot_prompts(betley_repo_path: Path) -> list[dict[str, Any]]:
    return _load_betley_prompts(
        betley_repo_path,
        filename="first_plot_questions.yaml",
        source="first_plot",
        drop_suffixes=("_json", "_template"),
    )


def load_preregistered_prompts(betley_repo_path: Path) -> list[dict[str, Any]]:
    return _load_betley_prompts(
        betley_repo_path,
        filename="preregistered_evals.yaml",
        source="preregistered",
        drop_suffixes=("_json", "_template"),
    )


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


def _append_prompt_row(rows: list[dict[str, Any]], source: str, source_key: str, prompt: str) -> None:
    text = prompt.strip()
    if text:
        rows.append({"source": source, "source_key": source_key, "prompt": text})


def _extract_turner_yaml_prompts(obj: Any, source: str, rows: list[dict[str, Any]], default_key: str = "") -> None:
    if isinstance(obj, list):
        for idx, entry in enumerate(obj):
            key = default_key or str(idx)
            _extract_turner_yaml_prompts(entry, source, rows, default_key=key)
        return

    if isinstance(obj, dict):
        source_key = str(obj.get("id", default_key))
        paraphrases = obj.get("paraphrases")
        if isinstance(paraphrases, list):
            for para_idx, paraphrase in enumerate(paraphrases):
                if isinstance(paraphrase, str):
                    key = f"{source_key}:paraphrase{para_idx}"
                    _append_prompt_row(rows, source, key, paraphrase)
            return
        if isinstance(obj.get("question"), str):
            _append_prompt_row(rows, source, source_key, obj["question"])
            return
        if isinstance(obj.get("prompt"), str):
            _append_prompt_row(rows, source, source_key, obj["prompt"])
            return
        for key, value in obj.items():
            nested_key = f"{source_key}:{key}" if source_key else str(key)
            _extract_turner_yaml_prompts(value, source, rows, default_key=nested_key)


def load_turner_eval_prompts(eval_questions_dir: Path) -> list[dict[str, Any]]:
    import yaml

    rows: list[dict[str, Any]] = []
    eval_questions_dir = Path(eval_questions_dir)

    yaml_files = sorted(
        path
        for path in eval_questions_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"} and path.name != "judges.yaml"
    )
    for path in yaml_files:
        with path.open() as f:
            data = yaml.safe_load(f)
        source = path.relative_to(eval_questions_dir).as_posix()
        _extract_turner_yaml_prompts(data, source, rows)

    semantic_path = eval_questions_dir / "semantic_questions.py"
    if semantic_path.exists():
        module = ast.parse(semantic_path.read_text())
        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) for target in node.targets):
                continue
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if not isinstance(node.value, ast.Dict):
                continue
            for key_node, value_node in zip(node.value.keys, node.value.values):
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str) and isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                    for target_name in target_names:
                        source = f"semantic_questions.py::{target_name}"
                        _append_prompt_row(rows, source, key_node.value, value_node.value)

    prompts = []
    seen = set()
    for row in rows:
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
