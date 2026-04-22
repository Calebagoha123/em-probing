# Primary Pipeline

This repo is now organized around the primary paper question only:

**Is a held-out residual-stream monitor of realized emergent misalignment more prompt-robust than behavioral misalignment rate?**

## Paths

- Base model default: `scripts/user_config.py`
- Checkpoints default: `scripts/user_config.py`
- Results root: `/data/kell8360/aml-em`
- Local figures: `figures/`

## Scripts

### 0. Download assets if needed

```bash
python3 scripts/00_download_assets.py
```

### 1. Generate and judge responses under prompt wrappers

Writes:
- `/data/kell8360/aml-em/responses/{condition}/step_<N>.json`

Example:

```bash
python3 scripts/01_generate_and_judge.py \
  --steps 395 \
  --conditions neutral,hhh,evil
```

### 2. Collect response-conditioned activations

Writes:
- `/data/kell8360/aml-em/activations/{condition}/step_<N>.npz`

Example:

```bash
python3 scripts/02_collect_activations.py \
  --steps 395 \
  --conditions neutral,hhh,evil
```

### 3. Fit the held-out monitor on the final checkpoint

Writes:
- `/data/kell8360/aml-em/monitors/monitor_step395_neutral.json`
- `/data/kell8360/aml-em/monitors/monitor_step395_neutral.npz`

Example:

```bash
python3 scripts/03_fit_monitor.py --step 395 --condition neutral
```

### 4. Evaluate prompt robustness

Writes:
- `/data/kell8360/aml-em/evaluations/prompt_robustness_step395.json`
- `figures/prompt_robustness_step395.png`

Example:

```bash
python3 scripts/04_eval_prompt_robustness.py \
  --monitor-prefix /data/kell8360/aml-em/monitors/monitor_step395_neutral
```

### Optional helper: summarize judged responses

```bash
python3 scripts/01b_summarize_responses.py
```

## Main outputs for the paper

- Behavioral misalignment rates by wrapper: step 1 / `01b`
- Response-conditioned activations: step 2
- Held-out selected layer and threshold: step 3
- Main robustness comparison figure and JSON summary: step 4

## Deliberate exclusions from the primary pipeline

These are not part of the current main question:
- prompt-only monitors
- external-label Option A datasets
- checkpoint-transfer / early-warning analysis
- layer-sweep exploratory branches

Those can be reintroduced later only if the primary result is clean.
