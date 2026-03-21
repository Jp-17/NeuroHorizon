#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/NeuroHorizon"
TASK_ROOT="results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning"
RUN_DIR="${ROOT}/${TASK_ROOT}/neuroformer_250ms_session_conditioning_e50"

cd "${ROOT}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate benchmark-env

mkdir -p "${RUN_DIR}"

conda run python neural-benchmark/faithful_neuroformer.py \
  --mode train \
  --obs-window 0.5 \
  --pred-window 0.25 \
  --epochs 50 \
  --batch-size 8 \
  --grad-accum-steps 20 \
  --num-workers 0 \
  --weight-decay 1.0 \
  --warmup-tokens 80000000 \
  --session-embedding-scale 1.0 \
  --output-dir "${TASK_ROOT}/neuroformer_250ms_session_conditioning_e50"

conda run python neural-benchmark/plot_benchmark_history.py \
  --results "${TASK_ROOT}/neuroformer_250ms_session_conditioning_e50/results.json"

conda run python neural-benchmark/faithful_neuroformer.py \
  --mode eval \
  --obs-window 0.5 \
  --pred-window 0.25 \
  --batch-size 8 \
  --grad-accum-steps 20 \
  --num-workers 0 \
  --weight-decay 1.0 \
  --warmup-tokens 80000000 \
  --session-embedding-scale 1.0 \
  --checkpoint-path "${TASK_ROOT}/neuroformer_250ms_session_conditioning_e50/best_model.pt" \
  --output-dir "${TASK_ROOT}/neuroformer_250ms_session_conditioning_e50/formal_eval" \
  --eval-split both \
  --inference-mode both
