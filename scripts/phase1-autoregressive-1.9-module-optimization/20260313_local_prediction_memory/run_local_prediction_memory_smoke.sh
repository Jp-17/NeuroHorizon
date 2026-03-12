#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/NeuroHorizon"
PYTHON="/root/miniconda3/envs/poyo/bin/python"

cd "$ROOT"

"$PYTHON" scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/verify_local_prediction_memory.py
"$PYTHON" examples/neurohorizon/train.py \
  --config-name=train_1p9_local_prediction_memory_250ms \
  epochs=1 eval_epochs=1 batch_size=256 eval_batch_size=256 num_workers=0
"$PYTHON" scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/250ms \
  --rollout \
  --skip-trial \
  --batch-size 256 \
  --output results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/250ms/eval_rollout_smoke.json
