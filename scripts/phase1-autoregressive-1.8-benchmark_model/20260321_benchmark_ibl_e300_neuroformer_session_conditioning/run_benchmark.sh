#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/NeuroHorizon
PYTHON=/root/miniconda3/bin/conda
ENV=benchmark-env
TASK_ROOT=results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning
IBL_DIR=${TASK_ROOT}/ibl_mtm_combined_e300_aligned
NF_DIR=${TASK_ROOT}/neuroformer_250ms_session_conditioning_e50

cd "$ROOT"

run_py() {
  "$PYTHON" run --no-capture-output -n "$ENV" python "$@"
}

mkdir -p "$TASK_ROOT"

run_py neural-benchmark/faithful_ibl_mtm.py \
  --mode train \
  --epochs 300 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --num-workers 4 \
  --train-mask-mode combined \
  --output-dir "$IBL_DIR"
run_py neural-benchmark/plot_benchmark_history.py --results-json "$IBL_DIR/results.json"

run_py neural-benchmark/faithful_neuroformer.py \
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
  --output-dir "$NF_DIR"
run_py neural-benchmark/plot_benchmark_history.py --results-json "$NF_DIR/results.json"

mkdir -p "$NF_DIR/formal_eval"
run_py neural-benchmark/faithful_neuroformer.py \
  --mode eval \
  --obs-window 0.5 \
  --pred-window 0.25 \
  --batch-size 8 \
  --grad-accum-steps 20 \
  --num-workers 0 \
  --weight-decay 1.0 \
  --warmup-tokens 80000000 \
  --session-embedding-scale 1.0 \
  --checkpoint-path "$NF_DIR/best_model.pt" \
  --output-dir "$NF_DIR/formal_eval" \
  --eval-split both \
  --inference-mode both
