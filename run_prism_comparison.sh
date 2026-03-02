#!/usr/bin/env bash
# PRISM 3-model sociodemographic probe comparison
# Run from the repo root: bash run_prism_comparison.sh
set -e

HUB=/data/resource/huggingface/hub

# ---------------------------------------------------------------------------
# 1. Data prep (model-agnostic, only needs to run once)
# ---------------------------------------------------------------------------
uv run python scripts/00c_prepare_prism_data.py --feature gender
uv run python scripts/00c_prepare_prism_data.py --feature ethnicity

# ---------------------------------------------------------------------------
# 2. Collect activations
# ---------------------------------------------------------------------------

# Qwen2.5-14B (already done — skip if step_0.npz exists)
uv run python scripts/02_collect_activations.py \
    --model-variant qwen-14b \
    --base-model "$HUB/models--Qwen--Qwen2.5-14B-Instruct" \
    --labelled-data results/responses/prism_labelled_gender_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --output-dir results/activations/qwen2.5-14b/gender

uv run python scripts/02_collect_activations.py \
    --model-variant qwen-14b \
    --base-model "$HUB/models--Qwen--Qwen2.5-14B-Instruct" \
    --labelled-data results/responses/prism_labelled_ethnicity_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --output-dir results/activations/qwen2.5-14b/ethnicity

# Qwen3-8B
uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-8b \
    --base-model "$HUB/models--Qwen--Qwen3-8B" \
    --labelled-data results/responses/prism_labelled_gender_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --output-dir results/activations/qwen3-8b/gender

uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-8b \
    --base-model "$HUB/models--Qwen--Qwen3-8B" \
    --labelled-data results/responses/prism_labelled_ethnicity_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --output-dir results/activations/qwen3-8b/ethnicity

# Llama 3.1 8B
uv run python scripts/02_collect_activations.py \
    --model-variant llama-8b \
    --base-model "$HUB/models--meta-llama--Llama-3.1-8B-Instruct" \
    --labelled-data results/responses/prism_labelled_gender_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --output-dir results/activations/llama-3.1-8b/gender

uv run python scripts/02_collect_activations.py \
    --model-variant llama-8b \
    --base-model "$HUB/models--meta-llama--Llama-3.1-8B-Instruct" \
    --labelled-data results/responses/prism_labelled_ethnicity_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --output-dir results/activations/llama-3.1-8b/ethnicity

# ---------------------------------------------------------------------------
# 3. Train probes
# ---------------------------------------------------------------------------
for MODEL in qwen2.5-14b qwen3-8b llama-3.1-8b; do
    for FEATURE in gender ethnicity; do
        uv run python scripts/03_train_probes.py \
            --activations-dir "results/activations/$MODEL/$FEATURE" \
            --output-dir "results/probes/$MODEL/$FEATURE"
    done
done

# ---------------------------------------------------------------------------
# 4. Plot — one line chart per feature, all models overlaid
# ---------------------------------------------------------------------------
for FEATURE in gender ethnicity; do
    uv run python scripts/04b_plot_prism_comparison.py \
        --feature "$FEATURE" \
        --metric accuracy \
        --probes-dirs \
            "results/probes/qwen2.5-14b/$FEATURE" \
            "results/probes/qwen3-8b/$FEATURE" \
            "results/probes/llama-3.1-8b/$FEATURE" \
        --labels "Qwen2.5-14B" "Qwen3-8B" "Llama-3.1-8B" \
        --output-dir results/figures/comparison
done

echo "[done] figures written to results/figures/comparison/"
