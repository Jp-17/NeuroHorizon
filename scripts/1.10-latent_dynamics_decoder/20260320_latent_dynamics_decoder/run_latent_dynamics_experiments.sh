#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/NeuroHorizon"
PYTHON="${PYTHON:-python}"
MODULE="20260320_latent_dynamics_decoder"
MODULE_LOG_ROOT="$ROOT/results/logs/1.10-latent_dynamics_decoder/$MODULE"

cd "$ROOT"
mkdir -p "$MODULE_LOG_ROOT/250ms" "$MODULE_LOG_ROOT/500ms" "$MODULE_LOG_ROOT/1000ms"

"$PYTHON" scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/verify_latent_dynamics.py

run_window() {
  local window="$1"
  local config_name="$2"
  local window_log_dir="$MODULE_LOG_ROOT/$window"

  "$PYTHON" examples/neurohorizon/train.py --config-name="$config_name"

  "$PYTHON" scripts/analysis/neurohorizon/eval_phase1_v2.py \
    --log-dir "$window_log_dir" \
    --checkpoint-kind best \
    --split valid \
    --skip-trial \
    --output "$window_log_dir/eval_v2_valid_results.json"

  "$PYTHON" scripts/analysis/neurohorizon/eval_phase1_v2.py \
    --log-dir "$window_log_dir" \
    --checkpoint-kind best \
    --split test \
    --skip-trial \
    --output "$window_log_dir/eval_v2_test_results.json"
}

run_window "250ms" "train_1p10_latent_dynamics_250ms" &
echo $! > "$MODULE_LOG_ROOT/250ms.job.pid"

run_window "500ms" "train_1p10_latent_dynamics_500ms" &
echo $! > "$MODULE_LOG_ROOT/500ms.job.pid"

run_window "1000ms" "train_1p10_latent_dynamics_1000ms" &
echo $! > "$MODULE_LOG_ROOT/1000ms.job.pid"

wait

"$PYTHON" scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/collect_latent_dynamics_results.py
