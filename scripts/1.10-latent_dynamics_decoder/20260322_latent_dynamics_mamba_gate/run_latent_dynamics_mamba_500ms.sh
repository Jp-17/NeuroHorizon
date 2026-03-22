#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/NeuroHorizon"
PYTHON="${PYTHON:-python}"
MODULE="20260322_latent_dynamics_mamba_gate"
LOG_DIR="$ROOT/results/logs/1.10-latent_dynamics_decoder/$MODULE/500ms"

cd "$ROOT"
mkdir -p "$LOG_DIR"

"$PYTHON" scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/verify_latent_dynamics_mamba_gate.py
"$PYTHON" examples/neurohorizon/train.py --config-name=train_1p10_latent_dynamics_mamba_500ms

"$PYTHON" scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir "$LOG_DIR" \
  --checkpoint-kind best \
  --split valid \
  --skip-trial \
  --output "$LOG_DIR/eval_v2_valid_results.json"

"$PYTHON" scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir "$LOG_DIR" \
  --checkpoint-kind best \
  --split test \
  --skip-trial \
  --output "$LOG_DIR/eval_v2_test_results.json"
