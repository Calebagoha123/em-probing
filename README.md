# em-probing (EM checkpoint probing)

This repo runs probing pipelines on:
- Base model: Llama-3.1-8B-Instruct
- Adapters: EM checkpoint series (`checkpoint-*`)
- Data sources:
  - Option A: fixed labeled pairs (`secure/insecure`, `good/bad medical`)
  - Option B: checkpoint-generated responses + judge labels (required for transition analysis)

Core pipeline:
1. Prepare labelled prompt/response pairs
2. Collect hidden activations (last token, all layers) for each checkpoint
3. Train per-layer logistic regression probes
4. Plot heatmap across layers x checkpoints

## Current defaults

Defaults are in `scripts/user_config.py`:
- `MODEL_VARIANT="llama-8b"`
- `N_PER_CLASS=200` (small class setting)

You can edit this file once and run everything without extra flags.

## Setup (uv only)

From `em-probing/`:

```bash
uv venv
source .venv/bin/activate
uv pip install torch transformers peft scikit-learn numpy matplotlib tqdm huggingface_hub openai pyyaml
```

## Download models from Hugging Face

You asked for:
- Base: `meta-llama/Llama-3.1-8B-Instruct`
- Checkpoints: `ModelOrganismsForEM/Llama-3.1-8B-Instruct_R1_0_1_0_full_train`

Use:

```bash
export HF_TOKEN=
uv run python scripts/00_download_assets.py \
  --model-variant llama-8b \
  --base-out /home/kell8360/em-probing/base_model \
  --checkpoints-out /home/kell8360/em-probing/em_checkpoints
```

Notes:
- Llama base model is gated; your HF account must have accepted the model license.
- The script prints discovered `checkpoint-*` directories after download.

## Configure paths

Edit `scripts/user_config.py`:
- `BASE_MODEL_PATH` -> local base model directory
- `CHECKPOINT_DIR` -> directory containing `checkpoint-*` (or `checkpoints/checkpoint-*`)
- `BETLEY_REPO_PATH` -> local `emergent-misalignment` repo

## Option A (fixed labeled dataset, baseline)

This is useful for engineering validation, but not sufficient by itself for checkpoint transition claims.

If you run Option A labels only, disable the per-step requirement:

```bash
uv run python scripts/02_collect_activations.py \
  --labelled-data results/responses/medical_labelled.json \
  --no-require-step-responses
```

## Option B (generate + judge per checkpoint, recommended for EM timing question)

This adds in-distribution labels by generating responses from each checkpoint and judging with OpenAI.

Set key:

```bash
export OPENAI_API_KEY=sk-...
```

Minimal smoke run (fast):

```bash
uv run python scripts/01_generate_and_judge.py \
  --steps 10 \
  --n-samples-per-prompt 1 \
  --max-prompts 4 \
  --skip-preregistered
```

Partial run:

```bash
uv run python scripts/01_generate_and_judge.py \
  --steps 10,90,170 \
  --n-samples-per-prompt 2
```

Outputs are written to `results/responses/step_<N>.json`.
Then Stage 2 picks these files up. By default, it now requires per-step response files.

```bash
uv run python scripts/02_collect_activations.py --steps 10,90,170
uv run python scripts/03_train_probes.py
uv run python scripts/04_plot_results.py --metric accuracy
```

Summarize label balance/misalignment rates:

```bash
uv run python scripts/01b_summarize_responses.py --responses-dir results/responses
```

## Transition-focused runbook

Use this if your question is: "when does misalignment become internally detectable across checkpoints?"

1. Smoke:
```bash
uv run python scripts/01_generate_and_judge.py --steps 10 --n-samples-per-prompt 1 --max-prompts 4 --skip-preregistered
uv run python scripts/02_collect_activations.py --steps 10
uv run python scripts/03_train_probes.py
```

2. Sanity:
```bash
uv run python scripts/01_generate_and_judge.py --steps 10,90,170 --n-samples-per-prompt 2 --max-prompts 16
uv run python scripts/01b_summarize_responses.py
uv run python scripts/02_collect_activations.py --steps 10,90,170
uv run python scripts/03_train_probes.py
uv run python scripts/04_plot_results.py --metric accuracy
uv run python scripts/04_plot_results.py --metric auc
```

3. Full:
```bash
uv run python scripts/01_generate_and_judge.py --n-samples-per-prompt 5
uv run python scripts/01b_summarize_responses.py
uv run python scripts/02_collect_activations.py
uv run python scripts/03_train_probes.py
uv run python scripts/04_plot_results.py --metric accuracy
uv run python scripts/04_plot_results.py --metric auc
```

4. Mean-difference analysis (paper-style complement):
```bash
uv run python scripts/05_mean_diff_analysis.py --activations-dir results/activations --output-dir results/figures
```

## Run

From `em-probing/`:

```bash
uv run python scripts/00b_prepare_betley_data.py
uv run python scripts/02_collect_activations.py
uv run python scripts/03_train_probes.py
uv run python scripts/04_plot_results.py --metric accuracy
```

Main outputs:
- Labelled data: `results/responses/betley_labelled.json`
- Activations: `results/activations/step_*.npz`
- Probe matrices: `results/probes/accuracy_matrix.npy`, `results/probes/auc_matrix.npy`
- Figure: `results/figures/heatmap_accuracy.png`

## Expected runtime (llama-8b, small class)

Assuming `N_PER_CLASS=200` (400 total examples) and ~30 checkpoints:
- Data prep: < 1 minute
- Activation collection: ~1.5 to 4 hours
- Probe training: ~10 to 30 minutes
- Plotting: < 1 minute
- End-to-end: ~2 to 5 hours

Runtime mostly depends on activation collection and adapter load speed.

## Last-layer-only sanity mode (engineering check)

Use this before full all-layer runs.

Recommended quick sanity settings:
- `N_PER_CLASS=100` (or keep 200 if you want slightly stronger signal)
- `LAST_LAYER_ONLY=True` in `scripts/user_config.py`
- `LIMIT_EXAMPLES=100` for a very fast check, then remove limit

Commands:

```bash
uv run python scripts/00b_prepare_betley_data.py
uv run python scripts/02_collect_activations.py --last-layer-only --limit 100
uv run python scripts/03_train_probes.py
uv run python scripts/04_plot_results.py --metric accuracy
```

You can target only specific checkpoints during activation extraction:

```bash
uv run python scripts/02_collect_activations.py --last-layer-only --limit 100 --steps 10,90,170
```

What should indicate things are working:
- `02_collect_activations.py` logs each checkpoint as saved, with non-zero examples.
- Each `results/activations/step_*.npz` has:
  - `activations` shape `(n_examples, 1, 4096)` for Llama last-layer mode
  - `layer_indices` equal to `[32]`
- `03_train_probes.py` runs through checkpoints without shape errors and writes:
  - `results/probes/accuracy_matrix.npy` with shape `(1, n_steps)`
  - `results/probes/layer_indices.npy` containing `[32]`
- Heatmap is a single-row plot across checkpoints.

Expected metric behavior for a sanity run:
- Accuracy should usually be above random chance (`0.5`) on at least some checkpoints.
- Very rough practical sanity threshold: seeing values around `0.55+` somewhere is enough to confirm the pipeline is likely wired correctly.
- If everything is exactly `0.5` everywhere, common causes are:
  - labels collapsed to one class,
  - wrong checkpoint path (adapter not loading),
  - activation files created with unexpected layer index configuration.

## VM validation sequence (recommended)

Run these in order to validate each stage before full runs.

1. Tiny smoke test (very fast):
```bash
uv run python scripts/00b_prepare_betley_data.py --n-per-class 20
uv run python scripts/02_collect_activations.py --last-layer-only --limit 5 --steps 10
uv run python scripts/03_train_probes.py --n-seeds 1 --n-folds 2
uv run python scripts/04_plot_results.py --metric accuracy
```

2. Partial sanity:
```bash
uv run python scripts/00b_prepare_betley_data.py --n-per-class 100
uv run python scripts/02_collect_activations.py --last-layer-only --limit 100 --steps 10,90,170
uv run python scripts/03_train_probes.py
uv run python scripts/04_plot_results.py --metric accuracy
```

3. Full sanity:
```bash
uv run python scripts/00b_prepare_betley_data.py
uv run python scripts/02_collect_activations.py --last-layer-only --limit 100
uv run python scripts/03_train_probes.py
uv run python scripts/04_plot_results.py --metric accuracy
```

4. Full run (all layers):
```bash
uv run python scripts/02_collect_activations.py
uv run python scripts/03_train_probes.py
uv run python scripts/04_plot_results.py --metric accuracy
```

## VM/GPU notes

- For multi-GPU machines, keep:
  - `DEVICE_MAP="auto"`
  - `INPUT_DEVICE="cuda:0"`
- Recommended dtype: `TORCH_DTYPE="bfloat16"`
- If you hit memory issues, reduce `MAX_SEQ_LEN` or set `LIMIT_EXAMPLES` for a smoke run.

## CLI overrides

Any default can be overridden at runtime, e.g.:

```bash
uv run python scripts/02_collect_activations.py --limit 50 --max-seq-len 768
```
