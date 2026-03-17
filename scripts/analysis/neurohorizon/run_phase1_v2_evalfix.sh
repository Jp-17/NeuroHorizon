#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/NeuroHorizon"
cd "$ROOT"

CONFIGS=(
  train_v2_evalfix_250ms
  train_v2_evalfix_250ms_trial
  train_v2_evalfix_500ms
  train_v2_evalfix_500ms_trial
  train_v2_evalfix_1000ms
  train_v2_evalfix_1000ms_trial
)

for config in "${CONFIGS[@]}"; do
  python examples/neurohorizon/train.py --config-name "$config"
done

declare -A LOG_DIRS=(
  [train_v2_evalfix_250ms]="/root/autodl-tmp/NeuroHorizon/results/logs/phase1_v2_evalfix_250ms_cont"
  [train_v2_evalfix_250ms_trial]="/root/autodl-tmp/NeuroHorizon/results/logs/phase1_v2_evalfix_250ms_trial"
  [train_v2_evalfix_500ms]="/root/autodl-tmp/NeuroHorizon/results/logs/phase1_v2_evalfix_500ms_cont"
  [train_v2_evalfix_500ms_trial]="/root/autodl-tmp/NeuroHorizon/results/logs/phase1_v2_evalfix_500ms_trial"
  [train_v2_evalfix_1000ms]="/root/autodl-tmp/NeuroHorizon/results/logs/phase1_v2_evalfix_1000ms_cont"
  [train_v2_evalfix_1000ms_trial]="/root/autodl-tmp/NeuroHorizon/results/logs/phase1_v2_evalfix_1000ms_trial"
)

for config in "${CONFIGS[@]}"; do
  log_dir="${LOG_DIRS[$config]}"
  python scripts/analysis/neurohorizon/eval_phase1_v2.py --log-dir "$log_dir" --split valid
  python scripts/analysis/neurohorizon/eval_phase1_v2.py --log-dir "$log_dir" --split test
done
