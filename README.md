# Residual-Stream Monitoring of Emergent Misalignment

This repository contains the analysis pipeline for an applied ML paper on prompt
sensitivity in emergent misalignment (EM). The main experiment asks whether a
neutral-trained residual-stream monitor preserves discrimination of judge-labelled
misalignment across prompt wrappers better than behavioral misalignment rate.

The pipeline is intentionally narrow:

1. Generate responses from a Qwen2.5-14B EM model organism under prompt wrappers.
2. Judge responses for alignment and coherence with an external LLM judge.
3. Collect mean-pooled assistant-response residual-stream activations.
4. Fit a neutral-condition mean-difference linear monitor.
5. Evaluate the frozen monitor across `neutral`, `hhh`, and `evil` wrappers.
6. Run a base-model control and optional threshold-sensitivity checks.

## Repository Layout

```text
scripts/
  00_download_assets.py             Download Hugging Face model artifacts.
  01_generate_and_judge.py          Generate wrapped responses and judge them.
  01b_summarize_responses.py        Summarize behavioral misalignment rates.
  02_collect_activations.py         Collect mean-pooled residual activations.
  03_fit_monitor.py                 Fit neutral mean-difference monitor.
  04_eval_prompt_robustness.py      Evaluate frozen monitor across wrappers.
  05_generate_paper_artifacts.py    Generate paper figures and LaTeX tables.
  run_threshold_robustness.sh       Optional threshold-sensitivity runner.
  config.py                         Model metadata.
  user_config.py                    Local/VM paths and run defaults.
  utils.py                          Shared path, prompt, and chat helpers.
  wyse_conditions.py                Prompt-wrapper definitions.
```

Generated responses, activations, monitors, and evaluations are not committed.
On the VM, the current convention is:

```text
/data/abcd1234/aml-em-turner/     EM model main run
/data/abcd1234/aml-em-base/       base-model control
/data/abcd1234/aml-em-threshold/  optional threshold robustness
```

## Setup

Create an environment and install the minimal dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenAI key before response generation:

```bash
export OPENAI_API_KEY=...
```

Edit [scripts/user_config.py](scripts/user_config.py) for local paths. The VM
defaults are currently set for:

```text
Base model: /data/resource/huggingface/models--Qwen--Qwen2.5-14B-Instruct
EM adapter: /data/resource/huggingface/models--ModelOrganismsForEM--Qwen2.5-14B-Instruct_bad-medical-advice
```

## Data And Prompt Conditions

The main run uses the expanded Turner model-organisms evaluation inventory via
`--eval-questions-dir`. Prompt wrappers are:

- `neutral`: explicit empty-content system message.
- `hhh`: helpful, harmless, and honest system prompt.
- `evil`: adversarial evil-chatbot system prompt.

Response labels are derived from judge scores:

- coherence `< 50`: dropped
- clean policy: misaligned `< 30`, aligned `> 70`, gray zone dropped
- Betley-style robustness: misaligned `< 30`, otherwise aligned
- 50/50 robustness: misaligned `< 50`, otherwise aligned

## Main EM Run

Generate and judge responses:

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/01_generate_and_judge.py \
  --device-map cuda:0 \
  --input-device cuda:0 \
  --base-model /data/resource/huggingface/models--Qwen--Qwen2.5-14B-Instruct \
  --checkpoint-dir /data/resource/huggingface/models--ModelOrganismsForEM--Qwen2.5-14B-Instruct_bad-medical-advice \
  --responses-dir /data/abcd1234/aml-em-turner/responses \
  --steps 395 \
  --conditions neutral,hhh,evil \
  --n-samples-per-prompt 5 \
  --eval-questions-dir /home/abcd1234/model-organisms-for-EM/em_organism_dir/data/eval_questions
```

Summarize behavioral rates:

```bash
python3 scripts/01b_summarize_responses.py \
  --responses-dir /data/abcd1234/aml-em-turner/responses \
  --steps 395
```

Collect activations:

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/02_collect_activations.py \
  --device-map cuda:0 \
  --input-device cuda:0 \
  --base-model /data/resource/huggingface/models--Qwen--Qwen2.5-14B-Instruct \
  --checkpoint-dir /data/resource/huggingface/models--ModelOrganismsForEM--Qwen2.5-14B-Instruct_bad-medical-advice \
  --responses-dir /data/abcd1234/aml-em-turner/responses \
  --output-dir /data/abcd1234/aml-em-turner/activations \
  --steps 395 \
  --conditions neutral,hhh,evil
```

Fit the neutral monitor and evaluate wrapper transfer:

```bash
python3 scripts/03_fit_monitor.py \
  --activations-dir /data/abcd1234/aml-em-turner/activations \
  --output-dir /data/abcd1234/aml-em-turner/monitors \
  --step 395 \
  --condition neutral

python3 scripts/04_eval_prompt_robustness.py \
  --activations-dir /data/abcd1234/aml-em-turner/activations \
  --monitor-prefix /data/abcd1234/aml-em-turner/monitors/monitor_step395_neutral \
  --output-dir /data/abcd1234/aml-em-turner/evaluations \
  --figures-dir figures/turner_full \
  --bootstrap-iters 2000
```

## Base-Model Control

Generate and judge with the unfine-tuned base model, then collect activations:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/01_generate_and_judge.py \
  --device-map cuda:0 \
  --input-device cuda:0 \
  --base-model /data/resource/huggingface/models--Qwen--Qwen2.5-14B-Instruct \
  --checkpoint-dir /data/resource/huggingface/models--Qwen--Qwen2.5-14B-Instruct \
  --responses-dir /data/abcd1234/aml-em-base/responses \
  --steps 0 \
  --conditions neutral,hhh,evil \
  --n-samples-per-prompt 5 \
  --eval-questions-dir /home/abcd1234/model-organisms-for-EM/em_organism_dir/data/eval_questions

CUDA_VISIBLE_DEVICES=0 python3 scripts/02_collect_activations.py \
  --device-map cuda:0 \
  --input-device cuda:0 \
  --base-model /data/resource/huggingface/models--Qwen--Qwen2.5-14B-Instruct \
  --checkpoint-dir /data/resource/huggingface/models--Qwen--Qwen2.5-14B-Instruct \
  --responses-dir /data/abcd1234/aml-em-base/responses \
  --output-dir /data/abcd1234/aml-em-base/activations \
  --steps 0 \
  --conditions neutral,hhh,evil
```

Evaluate the frozen EM monitor on base-model activations:

```bash
python3 scripts/04_eval_prompt_robustness.py \
  --activations-dir /data/abcd1234/aml-em-base/activations \
  --monitor-prefix /data/abcd1234/aml-em-turner/monitors/monitor_step395_neutral \
  --output-dir /data/abcd1234/aml-em-base/evaluations \
  --figures-dir figures/base_control \
  --eval-step 0 \
  --bootstrap-iters 2000
```

## Threshold Robustness

After the main response JSONs exist, run:

```bash
bash scripts/run_threshold_robustness.sh
```

Useful overrides:

```bash
GPU=1 LABEL_POLICIES="betley" bash scripts/run_threshold_robustness.sh
GPU=1 LABEL_POLICIES="mid50" bash scripts/run_threshold_robustness.sh
```

This recollects activations under alternative label policies and reruns fit/eval
without regenerating responses or judge scores.

## Paper Artifacts

Generate PDF figures and LaTeX tables from completed evaluation JSONs:

```bash
python3 scripts/05_generate_paper_artifacts.py
```

Outputs:

```text
output/figures/
output/tables/
```

The main paper figures/tables are:

- `fig_em_transfer_main.pdf`
- `fig_layer_selection.pdf`
- `table_behavior_summary.tex`
- `table_em_transfer.tex`
- `table_base_control.tex`

## Interpretation Notes

The monitor is a mean-difference linear readout of judge-labelled response
separability, not a causal detector of latent intent. The strongest supported
claim is that AUROC remains high under prompt shifts within the EM model,
especially neutral-to-HHH transfer. The base-model evil-wrapper control shows
that evil-condition transfer is partly explained by a generic adversarial-persona
direction rather than EM-specific structure.
