#!/usr/bin/env bash
# Transfer probe for Llama-3.1-8B only: train on paper validation, test on PRISM.
# Run from repo root: bash run_transfer_llama.sh
set -e

uv run python scripts/03_train_probes.py \
    --activations-dir results/activations/validation/llama-3.1-8b/gender \
    --test-activations-dir results/activations/last_user_token/llama-3.1-8b/gender \
    --output-dir results/probes/transfer/llama-3.1-8b/gender

uv run python scripts/04b_plot_prism_comparison.py \
    --feature "gender (paper->PRISM transfer)" \
    --probes-dirs results/probes/transfer/llama-3.1-8b/gender \
    --labels "Llama-3.1-8B" \
    --output-dir results/figures/transfer

echo "[done] figures in results/figures/transfer/"
