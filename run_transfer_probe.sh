#!/usr/bin/env bash
# Cross-domain transfer probe: train on paper validation data, evaluate on PRISM.
#
# Scientific question: does a probe trained on EXPLICIT demographic cues
# (e.g. "Hi I am a female parent...") generalise to ORGANIC PRISM conversations?
#
# Gender polarity note:  paper female=1 / PRISM Male=1  → opposite; auto-corrected.
# Race/ethnicity note:   paper white=1  / PRISM non-White=1 → opposite; auto-corrected.
# Both are handled via max(acc, 1-acc) inside 03_train_probes.py.
#
# Prerequisites:
#   - PRISM activations already in results/activations/last_user_token/ (run_prism_comparison.sh)
#   - Paper data file at data/conversations_250.json (downloaded automatically)
#
# Run from repo root: bash run_transfer_probe.sh
set -e

HUB=/data/resource/huggingface/hub
PROBE_POS=last_user_token
PRISM_ACT=results/activations/last_user_token
PAPER_ACT=results/activations/paper/last_user_token
TRANSFER_PROBES=results/probes/transfer

# ---------------------------------------------------------------------------
# 1. Prep paper labelled data (gender + race, introduction mode)
# ---------------------------------------------------------------------------
uv run python scripts/00e_prepare_paper_data.py --demographic gender --input-mode introduction
uv run python scripts/00e_prepare_paper_data.py --demographic race   --input-mode introduction

# ---------------------------------------------------------------------------
# 2. Collect paper activations for each model  (last_user_token probe position)
# ---------------------------------------------------------------------------

# Qwen2.5-14B
uv run python scripts/02_collect_activations.py \
    --model-variant qwen-14b \
    --base-model "$HUB/models--Qwen--Qwen2.5-14B-Instruct" \
    --labelled-data results/responses/paper_labelled_gender_introduction.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$PAPER_ACT/qwen2.5-14b/gender"

uv run python scripts/02_collect_activations.py \
    --model-variant qwen-14b \
    --base-model "$HUB/models--Qwen--Qwen2.5-14B-Instruct" \
    --labelled-data results/responses/paper_labelled_race_introduction.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$PAPER_ACT/qwen2.5-14b/race"

# Qwen3-8B
uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-8b \
    --base-model "$HUB/models--Qwen--Qwen3-8B" \
    --labelled-data results/responses/paper_labelled_gender_introduction.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$PAPER_ACT/qwen3-8b/gender"

uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-8b \
    --base-model "$HUB/models--Qwen--Qwen3-8B" \
    --labelled-data results/responses/paper_labelled_race_introduction.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$PAPER_ACT/qwen3-8b/race"

# Llama 3.1 8B
uv run python scripts/02_collect_activations.py \
    --model-variant llama-8b \
    --base-model "$HUB/models--meta-llama--Llama-3.1-8B-Instruct" \
    --labelled-data results/responses/paper_labelled_gender_introduction.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$PAPER_ACT/llama-3.1-8b/gender"

uv run python scripts/02_collect_activations.py \
    --model-variant llama-8b \
    --base-model "$HUB/models--meta-llama--Llama-3.1-8B-Instruct" \
    --labelled-data results/responses/paper_labelled_race_introduction.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$PAPER_ACT/llama-3.1-8b/race"

# ---------------------------------------------------------------------------
# 3. Transfer probes  (train=paper, test=PRISM)
# ---------------------------------------------------------------------------
for MODEL in qwen2.5-14b qwen3-8b llama-3.1-8b; do
    # Gender → gender
    uv run python scripts/03_train_probes.py \
        --activations-dir "$PAPER_ACT/$MODEL/gender" \
        --test-activations-dir "$PRISM_ACT/$MODEL/gender" \
        --output-dir "$TRANSFER_PROBES/$MODEL/gender"

    # Race (paper) → ethnicity (PRISM)
    uv run python scripts/03_train_probes.py \
        --activations-dir "$PAPER_ACT/$MODEL/race" \
        --test-activations-dir "$PRISM_ACT/$MODEL/ethnicity" \
        --output-dir "$TRANSFER_PROBES/$MODEL/ethnicity"
done

# ---------------------------------------------------------------------------
# 4. Plot transfer accuracy per layer, all models overlaid
# ---------------------------------------------------------------------------
for FEATURE in gender ethnicity; do
    uv run python scripts/04b_plot_prism_comparison.py \
        --feature "$FEATURE (paper→PRISM transfer)" \
        --probes-dirs \
            "$TRANSFER_PROBES/qwen2.5-14b/$FEATURE" \
            "$TRANSFER_PROBES/qwen3-8b/$FEATURE" \
            "$TRANSFER_PROBES/llama-3.1-8b/$FEATURE" \
        --labels "Qwen2.5-14B" "Qwen3-8B" "Llama-3.1-8B" \
        --output-dir results/figures/transfer
done

echo "[done] transfer probe figures written to results/figures/transfer/"
