#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/NeuroHorizon"
PYTHON="${PYTHON:-python}"
LOG_DIR="$ROOT/results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/250ms_smoke"

cd "$ROOT"

"$PYTHON" scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/verify_latent_dynamics.py

"$PYTHON" examples/neurohorizon/train.py \
  --config-name=train_1p10_latent_dynamics_250ms \
  epochs=1 \
  eval_epochs=1 \
  batch_size=256 \
  eval_batch_size=256 \
  num_workers=0 \
  log_dir="$LOG_DIR"

"$PYTHON" scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir "$LOG_DIR" \
  --checkpoint-kind best \
  --split valid \
  --skip-trial \
  --output "$LOG_DIR/eval_v2_valid_results.json"
