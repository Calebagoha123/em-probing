#!/usr/bin/env bash
# PRISM 3-model sociodemographic probe comparison
# Probes at the last token of the final user message (before model generation).
# Run from the repo root: bash run_prism_comparison.sh
set -e

HUB=/data/resource/huggingface/hub
PROBE_POS=last_user_token   # change to last_token to restore original behaviour
ACT_DIR=results/activations/last_user_token

# ---------------------------------------------------------------------------
# 1. Data prep (model-agnostic, only needs to run once)
# ---------------------------------------------------------------------------
uv run python scripts/00c_prepare_prism_data.py --feature gender
uv run python scripts/00c_prepare_prism_data.py --feature ethnicity

# ---------------------------------------------------------------------------
# 2. Collect activations
# ---------------------------------------------------------------------------

# Qwen2.5-14B
uv run python scripts/02_collect_activations.py \
    --model-variant qwen-14b \
    --base-model "$HUB/models--Qwen--Qwen2.5-14B-Instruct" \
    --labelled-data results/responses/prism_labelled_gender_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$ACT_DIR/qwen2.5-14b/gender"

uv run python scripts/02_collect_activations.py \
    --model-variant qwen-14b \
    --base-model "$HUB/models--Qwen--Qwen2.5-14B-Instruct" \
    --labelled-data results/responses/prism_labelled_ethnicity_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$ACT_DIR/qwen2.5-14b/ethnicity"

# Qwen3-8B
uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-8b \
    --base-model "$HUB/models--Qwen--Qwen3-8B" \
    --labelled-data results/responses/prism_labelled_gender_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$ACT_DIR/qwen3-8b/gender"

uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-8b \
    --base-model "$HUB/models--Qwen--Qwen3-8B" \
    --labelled-data results/responses/prism_labelled_ethnicity_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$ACT_DIR/qwen3-8b/ethnicity"

# Llama 3.1 8B
uv run python scripts/02_collect_activations.py \
    --model-variant llama-8b \
    --base-model "$HUB/models--meta-llama--Llama-3.1-8B-Instruct" \
    --labelled-data results/responses/prism_labelled_gender_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$ACT_DIR/llama-3.1-8b/gender"

uv run python scripts/02_collect_activations.py \
    --model-variant llama-8b \
    --base-model "$HUB/models--meta-llama--Llama-3.1-8B-Instruct" \
    --labelled-data results/responses/prism_labelled_ethnicity_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$ACT_DIR/llama-3.1-8b/ethnicity"

# Qwen3-32B
uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-32b \
    --base-model "$HUB/models--Qwen--Qwen3-32B" \
    --labelled-data results/responses/prism_labelled_gender_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$ACT_DIR/qwen3-32b/gender"

uv run python scripts/02_collect_activations.py \
    --model-variant qwen3-32b \
    --base-model "$HUB/models--Qwen--Qwen3-32B" \
    --labelled-data results/responses/prism_labelled_ethnicity_full_conversation.json \
    --include-base-step --no-require-step-responses --steps 0 \
    --probe-position "$PROBE_POS" \
    --output-dir "$ACT_DIR/qwen3-32b/ethnicity"

# ---------------------------------------------------------------------------
# 3. Train probes
# ---------------------------------------------------------------------------
for MODEL in qwen2.5-14b qwen3-8b llama-3.1-8b qwen3-32b; do
    for FEATURE in gender ethnicity; do
        uv run python scripts/03_train_probes.py \
            --activations-dir "$ACT_DIR/$MODEL/$FEATURE" \
            --output-dir "results/probes/last_user_token/$MODEL/$FEATURE"
    done
done

# ---------------------------------------------------------------------------
# 4. Plot — one line chart per feature, all models overlaid
# ---------------------------------------------------------------------------
for FEATURE in gender ethnicity; do
    uv run python scripts/04b_plot_prism_comparison.py \
        --feature "$FEATURE (last user token)" \
        --probes-dirs \
            "results/probes/last_user_token/qwen2.5-14b/$FEATURE" \
            "results/probes/last_user_token/qwen3-8b/$FEATURE" \
            "results/probes/last_user_token/llama-3.1-8b/$FEATURE" \
            "results/probes/last_user_token/qwen3-32b/$FEATURE" \
        --labels "Qwen2.5-14B" "Qwen3-8B" "Llama-3.1-8B" "Qwen3-32B" \
        --output-dir results/figures/last_user_token
done

echo "[done] figures written to results/figures/last_user_token/"
