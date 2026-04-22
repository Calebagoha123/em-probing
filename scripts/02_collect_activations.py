import argparse
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODELS
from user_config import (
    ACTIVATIONS_DIR,
    BASE_MODEL_PATH,
    CHECKPOINT_DIR,
    DEVICE_MAP,
    INPUT_DEVICE,
    LAYER_INDICES,
    LIMIT_EXAMPLES,
    MAX_SEQ_LEN,
    MODEL_VARIANT,
    RESPONSES_DIR,
    TORCH_DTYPE,
)
from utils import ensure_dir, format_chat, get_checkpoint_steps, load_json, resolve_local_snapshot, save_json, step_to_path
from wyse_conditions import CONDITION_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect response-conditioned residual-stream activations.")
    parser.add_argument("--model-variant", choices=MODELS.keys(), default=MODEL_VARIANT)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--responses-dir", type=Path, default=RESPONSES_DIR)
    parser.add_argument("--output-dir", type=Path, default=ACTIVATIONS_DIR)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--device-map", type=str, default=DEVICE_MAP)
    parser.add_argument("--input-device", type=str, default=INPUT_DEVICE)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=TORCH_DTYPE)
    parser.add_argument("--limit", type=int, default=LIMIT_EXAMPLES)
    parser.add_argument("--layer-indices", type=str, default=LAYER_INDICES)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def parse_steps_arg(steps_arg: str | None) -> set[int] | None:
    if not steps_arg:
        return None
    parsed = set()
    for chunk in steps_arg.split(","):
        value = chunk.strip()
        if not value:
            continue
        if not value.isdigit():
            raise ValueError(f"Invalid step value '{value}' in --steps")
        parsed.add(int(value))
    return parsed if parsed else None


def parse_conditions_arg(cond_arg: str | None) -> list[str]:
    if not cond_arg:
        return CONDITION_LABELS
    keys = [chunk.strip() for chunk in cond_arg.split(",") if chunk.strip()]
    unknown = [key for key in keys if key not in CONDITION_LABELS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Valid: {CONDITION_LABELS}")
    return keys


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_model(base_model_path: Path, checkpoint_dir: Path, step: int, dtype: torch.dtype, device_map: str):
    base_model = AutoModelForCausalLM.from_pretrained(
        resolve_local_snapshot(base_model_path),
        torch_dtype=dtype,
        device_map=device_map,
    )
    base_model.eval()
    if step == 0:
        return base_model, base_model
    model = PeftModel.from_pretrained(base_model, step_to_path(checkpoint_dir, step))
    model.eval()
    return model, base_model


def unload_model(model, base_model, step: int) -> None:
    if step != 0:
        del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    cfg = MODELS[args.model_variant]
    ensure_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(resolve_local_snapshot(args.base_model), use_fast=True)
    steps = get_checkpoint_steps(args.checkpoint_dir)
    step_filter = parse_steps_arg(args.steps)
    if step_filter is not None:
        steps = [step for step in steps if step in step_filter]
    if not steps:
        raise ValueError("No checkpoint steps selected.")
    conditions = parse_conditions_arg(args.conditions)

    n_layers_with_embedding = cfg.num_layers + 1
    if args.layer_indices:
        selected_layers = [int(token.strip()) for token in args.layer_indices.split(",") if token.strip()]
    else:
        selected_layers = list(range(n_layers_with_embedding))
    invalid = [layer for layer in selected_layers if layer < 0 or layer >= n_layers_with_embedding]
    if invalid:
        raise ValueError(f"Invalid layer indices {invalid}; valid range is [0, {n_layers_with_embedding - 1}]")

    dtype = get_torch_dtype(args.dtype)
    for step in steps:
        print(f"\n[step {step}] loading model...")
        model, base_model = load_model(args.base_model, args.checkpoint_dir, step, dtype, args.device_map)

        for condition in conditions:
            in_path = args.responses_dir / condition / f"step_{step}.json"
            out_npz = args.output_dir / condition / f"step_{step}.npz"
            out_meta = args.output_dir / condition / f"step_{step}_meta.json"
            if out_npz.exists() and out_meta.exists() and not args.overwrite:
                print(f"[skip] {condition} step {step}: {out_npz}")
                continue
            if not in_path.exists():
                print(f"[warn] missing responses: {in_path}")
                continue

            rows = load_json(in_path)
            rows = [row for row in rows if row.get("label") in (0, 1) and row.get("prompt") and row.get("response")]
            if args.limit is not None:
                rows = rows[: args.limit]
            if not rows:
                print(f"[warn] no usable rows for {condition} step {step}")
                continue

            layer_stack = []
            labels = []
            prompt_ids = []
            sample_ids = []
            meta = []

            for row in tqdm(rows, desc=f"step {step} / {condition}"):
                text = format_chat(
                    tokenizer,
                    prompt=row["prompt"],
                    response=row["response"],
                    system_prompt=row.get("system_prompt", ""),
                )
                tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.input_device)
                if tokenized.input_ids.shape[1] < 2:
                    continue
                last_pos = tokenized.input_ids.shape[1] - 1

                with torch.no_grad():
                    out = model(**tokenized, output_hidden_states=True, use_cache=False)

                if len(out.hidden_states) != n_layers_with_embedding:
                    raise ValueError(
                        f"Expected {n_layers_with_embedding} hidden states, got {len(out.hidden_states)} "
                        f"for {condition} step {step}"
                    )

                layer_vecs = [
                    out.hidden_states[layer][0, last_pos, :].detach().cpu().float().numpy()
                    for layer in selected_layers
                ]
                layer_stack.append(np.stack(layer_vecs, axis=0))
                labels.append(int(row["label"]))
                prompt_ids.append(int(row["prompt_id"]))
                sample_ids.append(int(row["sample_id"]))
                meta.append(
                    {
                        "prompt_id": int(row["prompt_id"]),
                        "sample_id": int(row["sample_id"]),
                        "prompt_source": row.get("prompt_source"),
                        "prompt_source_key": row.get("prompt_source_key"),
                        "prompt": row["prompt"],
                        "response_preview": row["response"][:200],
                        "label": int(row["label"]),
                        "alignment_score": row.get("alignment_score"),
                        "coherence_score": row.get("coherence_score"),
                    }
                )

            if not layer_stack:
                print(f"[warn] no activations collected for {condition} step {step}")
                continue

            ensure_dir(out_npz.parent)
            activations = np.stack(layer_stack, axis=0).astype(np.float32)
            np.savez_compressed(
                out_npz,
                activations=activations,
                labels=np.asarray(labels, dtype=np.int32),
                prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
                sample_ids=np.asarray(sample_ids, dtype=np.int32),
                layer_indices=np.asarray(selected_layers, dtype=np.int32),
                step=np.int32(step),
            )
            save_json(out_meta, meta)
            print(f"[ok] {condition} step {step}: saved {activations.shape} → {out_npz}")

        unload_model(model, base_model, step)


if __name__ == "__main__":
    main()
