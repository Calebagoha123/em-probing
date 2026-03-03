#!/usr/bin/env bash
# Download Qwen3-32B then collect PRISM activations, train probes, and plot.
# Run from repo root: bash run_qwen3_32b.sh
set -e

HUB=/data/resource/huggingface/hub
ACT_DIR=results/activations/last_user_token

# 1. Download Qwen3-32B (skipped automatically if already present)
uv run python scripts/00d_download_prism_comparison_models.py --models qwen3-32b

# 2. Collect PRISM activations
uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-32b \
    --base-model "$HUB/models--Qwen--Qwen3-32B" \
    --labelled-data results/responses/prism_labelled_gender_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position last_user_token \
    --output-dir "$ACT_DIR/qwen3-32b/gender"

uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-32b \
    --base-model "$HUB/models--Qwen--Qwen3-32B" \
    --labelled-data results/responses/prism_labelled_ethnicity_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position last_user_token \
    --output-dir "$ACT_DIR/qwen3-32b/ethnicity"

# 3. Train probes
uv run python scripts/03_train_probes.py \
    --activations-dir "$ACT_DIR/qwen3-32b/gender" \
    --output-dir results/probes/last_user_token/qwen3-32b/gender

uv run python scripts/03_train_probes.py \
    --activations-dir "$ACT_DIR/qwen3-32b/ethnicity" \
    --output-dir results/probes/last_user_token/qwen3-32b/ethnicity

# 4. Plot Qwen3-32B alongside the existing three models
uv run python scripts/04b_plot_prism_comparison.py \
    --feature "gender (last user token)" \
    --probes-dirs \
        results/probes/last_user_token/qwen2.5-14b/gender \
        results/probes/last_user_token/qwen3-8b/gender \
        results/probes/last_user_token/llama-3.1-8b/gender \
        results/probes/last_user_token/qwen3-32b/gender \
    --labels "Qwen2.5-14B" "Qwen3-8B" "Llama-3.1-8B" "Qwen3-32B" \
    --output-dir results/figures/last_user_token

uv run python scripts/04b_plot_prism_comparison.py \
    --feature "ethnicity (last user token)" \
    --probes-dirs \
        results/probes/last_user_token/qwen2.5-14b/ethnicity \
        results/probes/last_user_token/qwen3-8b/ethnicity \
        results/probes/last_user_token/llama-3.1-8b/ethnicity \
        results/probes/last_user_token/qwen3-32b/ethnicity \
    --labels "Qwen2.5-14B" "Qwen3-8B" "Llama-3.1-8B" "Qwen3-32B" \
    --output-dir results/figures/last_user_token

echo "[done] figures in results/figures/last_user_token/"
