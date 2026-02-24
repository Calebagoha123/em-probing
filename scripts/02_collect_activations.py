import argparse
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ACTIVATIONS_DIR, MODELS
from utils import ensure_dir, format_chat, get_checkpoint_steps, load_json, save_json, step_to_path
from user_config import (
    BASE_MODEL_PATH,
    CHECKPOINT_DIR,
    DEVICE_MAP,
    INPUT_DEVICE,
    LAST_LAYER_ONLY,
    LAYER_INDICES,
    LABELLED_DATA_PATH,
    LIMIT_EXAMPLES,
    MAX_SEQ_LEN,
    MODEL_VARIANT,
    RESPONSES_DIR,
    REQUIRE_STEP_RESPONSES,
    TORCH_DTYPE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-layer last-token activations for labelled prompt-response pairs.")
    parser.add_argument("--model-variant", choices=MODELS.keys(), default=MODEL_VARIANT)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL_PATH, help="Local path to base model weights.")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR, help="Directory containing checkpoint-* folders.")
    parser.add_argument("--labelled-data", type=Path, default=LABELLED_DATA_PATH)
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=RESPONSES_DIR,
        help="If results/responses/step_<N>.json exists for a step, Stage 2 uses it (Option B).",
    )
    parser.add_argument(
        "--require-step-responses",
        action=argparse.BooleanOptionalAction,
        default=REQUIRE_STEP_RESPONSES,
        help="Require step_<N>.json in responses-dir for each checkpoint (prevents accidental Option A fallback).",
    )
    parser.add_argument("--output-dir", type=Path, default=ACTIVATIONS_DIR)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--device-map", type=str, default=DEVICE_MAP, help="Transformers device_map (e.g. auto, cuda:0).")
    parser.add_argument("--input-device", type=str, default=INPUT_DEVICE, help="Device where tokenized inputs are placed.")
    parser.add_argument("--dtype", type=str, default=TORCH_DTYPE, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--limit", type=int, default=LIMIT_EXAMPLES, help="Optional cap on number of examples per checkpoint.")
    parser.add_argument(
        "--last-layer-only",
        action="store_true",
        default=LAST_LAYER_ONLY,
        help="Collect only the final transformer layer (fast sanity mode).",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default=LAYER_INDICES,
        help="Comma-separated hidden-state indices to collect, e.g. 0,16,32. Overrides --last-layer-only.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Optional checkpoint steps to run. Comma-separated, e.g. 10,50,170.",
    )
    parser.add_argument(
        "--include-base-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also collect activations for base model as step_0 (no adapter).",
    )
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


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def main() -> None:
    args = parse_args()
    cfg = MODELS[args.model_variant]
    ensure_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    steps = get_checkpoint_steps(args.checkpoint_dir)
    if not steps:
        raise ValueError(f"No checkpoint-* folders found in {args.checkpoint_dir}")
    step_filter = parse_steps_arg(args.steps)
    if step_filter is not None:
        steps = [s for s in steps if s in step_filter]
        if not steps:
            raise ValueError(f"No matching checkpoints found for --steps={args.steps}")
        print(f"[config] filtered steps: {steps}")
    if args.include_base_step and 0 not in steps:
        steps = [0] + steps

    n_layers_with_embedding = cfg.num_layers + 1
    if args.layer_indices:
        selected_layers = [int(s.strip()) for s in args.layer_indices.split(",") if s.strip()]
    elif args.last_layer_only:
        selected_layers = [cfg.num_layers]
    else:
        selected_layers = list(range(n_layers_with_embedding))

    bad_layers = [idx for idx in selected_layers if idx < 0 or idx >= n_layers_with_embedding]
    if bad_layers:
        raise ValueError(f"Invalid layer indices {bad_layers}; valid range is [0, {n_layers_with_embedding - 1}]")

    print(f"[config] selected layer indices: {selected_layers}")

    for step in steps:
        out_npz = args.output_dir / f"step_{step}.npz"
        out_meta = args.output_dir / f"step_{step}_meta.json"
        if out_npz.exists() and out_meta.exists():
            print(f"[skip] step {step} already processed")
            continue

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=get_torch_dtype(args.dtype),
            device_map=args.device_map,
        )
        base_model.eval()

        if step == 0:
            model = base_model
            model.eval()
            print("[config] step 0 uses base model only (no adapter)")
        else:
            ckpt_path = step_to_path(args.checkpoint_dir, step)
            model = PeftModel.from_pretrained(base_model, ckpt_path)
            model.eval()

        step_response_path = args.responses_dir / f"step_{step}.json"
        if step_response_path.exists():
            rows = load_json(step_response_path)
            source = f"option-b:{step_response_path}"
        else:
            if args.require_step_responses:
                raise FileNotFoundError(
                    f"Missing {step_response_path}. Run 01_generate_and_judge.py first or disable --require-step-responses."
                )
            rows = load_json(args.labelled_data)
            source = f"option-a:{args.labelled_data}"

        rows = [r for r in rows if r.get("label") in (0, 1) and r.get("prompt") and r.get("response")]
        if args.limit is not None:
            rows = rows[: args.limit]
        print(f"[config] step {step}: rows={len(rows)} source={source}")

        layer_stack = []
        labels = []
        meta = []

        for row in tqdm(rows, desc=f"step {step}"):
            text = format_chat(tokenizer, prompt=row["prompt"], response=row["response"])
            tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.input_device)
            if tok.input_ids.shape[1] < 2:
                continue
            last_pos = tok.input_ids.shape[1] - 1

            with torch.no_grad():
                out = model(**tok, output_hidden_states=True, use_cache=False)

            if len(out.hidden_states) != n_layers_with_embedding:
                raise ValueError(
                    f"Expected {n_layers_with_embedding} hidden states, got {len(out.hidden_states)} for step {step}"
                )

            layer_vecs = [out.hidden_states[i][0, last_pos, :].detach().cpu().float().numpy() for i in selected_layers]
            layer_stack.append(np.stack(layer_vecs, axis=0))
            labels.append(int(row["label"]))
            meta.append(
                {
                    "prompt": row["prompt"],
                    "response_preview": row["response"][:160],
                    "label": int(row["label"]),
                    "alignment_score": row.get("alignment_score"),
                    "coherence_score": row.get("coherence_score"),
                }
            )

        if not layer_stack:
            print(f"[warn] no usable rows for step {step}")
        else:
            activations = np.stack(layer_stack, axis=0).astype(np.float32)
            labels_arr = np.asarray(labels, dtype=np.int32)
            np.savez_compressed(
                out_npz,
                activations=activations,
                labels=labels_arr,
                step=step,
                layer_indices=np.asarray(selected_layers, dtype=np.int32),
            )
            save_json(out_meta, meta)
            print(f"[ok] step {step}: saved {activations.shape[0]} examples to {out_npz}")

        if step != 0:
            del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
