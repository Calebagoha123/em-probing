#!/usr/bin/env bash
# Validate the pipeline against Kovacs et al. (2025) paper data.
# Uses their explicit demographic introductions — should give ~90-100% accuracy.
# Run from the repo root: bash run_paper_validation.sh
set -e

HUB=/data/resource/huggingface/hub

# 1. Download paper data and prep labels
uv run python scripts/00e_prepare_paper_data.py \
    --demographic gender \
    --input-mode introduction \
    --data-file data/conversations_250.json

# 2. Collect activations (Llama for direct comparison with paper)
uv run python scripts/02_collect_activations.py \
    --model-variant llama-8b \
    --base-model "$HUB/models--meta-llama--Llama-3.1-8B-Instruct" \
    --labelled-data results/responses/paper_labelled_gender_introduction.json \
    --include-base-step \
    --no-require-step-responses \
    --steps 0 \
    --output-dir results/activations/validation/llama-3.1-8b/gender

# 3. Train probes + permutation baseline
uv run python scripts/03_train_probes.py \
    --activations-dir results/activations/validation/llama-3.1-8b/gender \
    --output-dir results/probes/validation/llama-3.1-8b/gender

# 4. Plot
uv run python scripts/04b_plot_prism_comparison.py \
    --feature "gender (paper validation)" \
    --probes-dirs results/probes/validation/llama-3.1-8b/gender \
    --labels "Llama-3.1-8B" \
    --output-dir results/figures/validation

echo "[done] validation figure written to results/figures/validation/"
