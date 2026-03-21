#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-poyo}"
ROOT="/root/autodl-tmp/NeuroHorizon"
CONDA_BIN="/root/miniconda3/bin/conda"
TRAIN_ROOT="$ROOT/examples/neurohorizon"
LOG_DIR="$ROOT/results/logs/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/250ms"

"$CONDA_BIN" run -n "$ENV_NAME" --cwd "$TRAIN_ROOT" \
  python train.py --config-name train_1p11_dense_history_cross_factorized_flow_250ms.yaml

"$CONDA_BIN" run -n "$ENV_NAME" --cwd "$ROOT" \
  python scripts/analysis/neurohorizon/eval_phase1_v2.py \
    --log-dir "$LOG_DIR" \
    --checkpoint-kind best \
    --split valid \
    --output "$LOG_DIR/eval_v2_valid_results.json"

"$CONDA_BIN" run -n "$ENV_NAME" --cwd "$ROOT" \
  python scripts/analysis/neurohorizon/eval_phase1_v2.py \
    --log-dir "$LOG_DIR" \
    --checkpoint-kind best \
    --split test \
    --output "$LOG_DIR/eval_v2_test_results.json"
