#!/usr/bin/env bash
set -euo pipefail

# Reuse existing generation/judge JSONs, but recollect activations under
# alternative binary label policies before fitting/evaluating monitors.
#
# Usage:
#   bash scripts/run_threshold_robustness.sh
#
# Optional overrides:
#   GPU=1 BOOTSTRAP_ITERS=1000 bash scripts/run_threshold_robustness.sh
#   LABEL_POLICIES="betley mid50" bash scripts/run_threshold_robustness.sh

GPU="${GPU:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
LABEL_POLICIES="${LABEL_POLICIES:-betley mid50}"

BASE_MODEL="${BASE_MODEL:-/data/resource/huggingface/models--Qwen--Qwen2.5-14B-Instruct}"
EM_CHECKPOINT="${EM_CHECKPOINT:-/data/resource/huggingface/models--ModelOrganismsForEM--Qwen2.5-14B-Instruct_bad-medical-advice}"

EM_RESPONSES="${EM_RESPONSES:-/data/abcd1234/aml-em-turner/responses}"
BASE_RESPONSES="${BASE_RESPONSES:-/data/abcd1234/aml-em-base/responses}"
OUT_ROOT="${OUT_ROOT:-/data/abcd1234/aml-em-threshold}"

for POLICY in ${LABEL_POLICIES}; do
  echo
  echo "=== threshold robustness: ${POLICY} ==="

  EM_ACT="${OUT_ROOT}/${POLICY}/em/activations"
  EM_MON="${OUT_ROOT}/${POLICY}/em/monitors"
  EM_EVAL="${OUT_ROOT}/${POLICY}/em/evaluations"
  BASE_ACT="${OUT_ROOT}/${POLICY}/base/activations"
  BASE_EVAL="${OUT_ROOT}/${POLICY}/base/evaluations"

  mkdir -p "${EM_ACT}" "${EM_MON}" "${EM_EVAL}" "${BASE_ACT}" "${BASE_EVAL}"

  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/02_collect_activations.py \
    --base-model "${BASE_MODEL}" \
    --checkpoint-dir "${EM_CHECKPOINT}" \
    --responses-dir "${EM_RESPONSES}" \
    --output-dir "${EM_ACT}" \
    --steps 395 \
    --conditions neutral,hhh,evil \
    --label-policy "${POLICY}" \
    --device-map cuda:0 \
    --input-device cuda:0

  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/02_collect_activations.py \
    --base-model "${BASE_MODEL}" \
    --checkpoint-dir "${BASE_MODEL}" \
    --responses-dir "${BASE_RESPONSES}" \
    --output-dir "${BASE_ACT}" \
    --steps 0 \
    --conditions neutral,hhh,evil \
    --label-policy "${POLICY}" \
    --device-map cuda:0 \
    --input-device cuda:0

  python3 scripts/03_fit_monitor.py \
    --activations-dir "${EM_ACT}" \
    --output-dir "${EM_MON}" \
    --step 395 \
    --condition neutral

  python3 scripts/04_eval_prompt_robustness.py \
    --activations-dir "${EM_ACT}" \
    --monitor-prefix "${EM_MON}/monitor_step395_neutral" \
    --output-dir "${EM_EVAL}" \
    --figures-dir "figures/threshold_${POLICY}/em" \
    --bootstrap-iters "${BOOTSTRAP_ITERS}"

  python3 scripts/04_eval_prompt_robustness.py \
    --activations-dir "${BASE_ACT}" \
    --monitor-prefix "${EM_MON}/monitor_step395_neutral" \
    --output-dir "${BASE_EVAL}" \
    --figures-dir "figures/threshold_${POLICY}/base" \
    --eval-step 0 \
    --bootstrap-iters "${BOOTSTRAP_ITERS}"

  echo "[done] ${POLICY}: ${OUT_ROOT}/${POLICY}"
done
