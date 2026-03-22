#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/NeuroHorizon
LOG_DIR="$ROOT/results/logs/1.11-diffusion-decoder/20260322_latent_diffusion_factorized_latent/250ms"
CONDA_BIN=/root/miniconda3/bin/conda

${CONDA_BIN} run -n poyo --cwd "$ROOT/examples/neurohorizon" \
python train.py --config-name train_1p11_latent_diffusion_factorized_latent_250ms.yaml \
  log_dir="$LOG_DIR"

${CONDA_BIN} run -n poyo --cwd "$ROOT" \
python scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir "${LOG_DIR}" \
  --checkpoint-kind best \
  --split valid

${CONDA_BIN} run -n poyo --cwd "$ROOT" \
python scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir "${LOG_DIR}" \
  --checkpoint-kind best \
  --split test
