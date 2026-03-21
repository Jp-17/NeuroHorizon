#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-poyo}"
ROOT="/root/autodl-tmp/NeuroHorizon"
CONDA_BIN="/root/miniconda3/bin/conda"
TRAIN_ROOT="$ROOT/examples/neurohorizon"

run_train() {
  local config_name="$1"
  "$CONDA_BIN" run -n "$ENV_NAME" --cwd "$TRAIN_ROOT" \
    python train.py --config-name "$config_name"
}

run_eval() {
  local log_dir="$1"
  "$CONDA_BIN" run -n "$ENV_NAME" --cwd "$ROOT" \
    python scripts/analysis/neurohorizon/eval_phase1_v2.py \
      --log-dir "$log_dir" \
      --checkpoint-kind best \
      --split valid \
      --output "$log_dir/eval_v2_valid_results.json"

  "$CONDA_BIN" run -n "$ENV_NAME" --cwd "$ROOT" \
    python scripts/analysis/neurohorizon/eval_phase1_v2.py \
      --log-dir "$log_dir" \
      --checkpoint-kind best \
      --split test \
      --output "$log_dir/eval_v2_test_results.json"
}

run_train "train_1p11_dense_history_cross_factorized_flow_250ms.yaml"
run_eval "$ROOT/results/logs/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/250ms"

run_train "train_1p11_dense_history_cross_factorized_flow_500ms.yaml"
run_eval "$ROOT/results/logs/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/500ms"

run_train "train_1p11_dense_history_cross_factorized_flow_1000ms.yaml"
run_eval "$ROOT/results/logs/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/1000ms"
