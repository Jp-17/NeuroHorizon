#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/NeuroHorizon
PYTHON=/root/miniconda3/bin/conda
ENV=benchmark-env
cd "$ROOT"

run_py() {
  "$PYTHON" run -n "$ENV" python "$@"
}

IBL_DIR=results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned
NF_CANON_DIR=results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_canonical_e50_aligned
NF_REF_DIR=results/logs/phase1_benchmark_repro_faithful_neuroformer_50ms_reference_e50_aligned

run_py neural-benchmark/faithful_ibl_mtm.py \
  --mode train \
  --epochs 50 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --num-workers 4 \
  --train-mask-mode combined \
  --output-dir "$IBL_DIR"
run_py neural-benchmark/plot_benchmark_history.py --results-json "$IBL_DIR/results.json"
run_py neural-benchmark/compare_faithful_ibl_mtm.py \
  --baseline-json results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e10/results.json \
  --control-json "$IBL_DIR/results.json" \
  --output-dir results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_compare_e10_e50_aligned

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
  --output-dir "$NF_CANON_DIR"
run_py neural-benchmark/plot_benchmark_history.py --results-json "$NF_CANON_DIR/results.json"
mkdir -p "$NF_CANON_DIR/formal_eval"
run_py neural-benchmark/faithful_neuroformer.py \
  --mode eval \
  --obs-window 0.5 \
  --pred-window 0.25 \
  --batch-size 8 \
  --grad-accum-steps 20 \
  --num-workers 0 \
  --weight-decay 1.0 \
  --warmup-tokens 80000000 \
  --checkpoint-path "$NF_CANON_DIR/best_model.pt" \
  --output-dir "$NF_CANON_DIR/formal_eval" \
  --eval-split both \
  --inference-mode both

run_py neural-benchmark/faithful_neuroformer.py \
  --mode train \
  --obs-window 0.15 \
  --pred-window 0.05 \
  --epochs 50 \
  --batch-size 8 \
  --grad-accum-steps 20 \
  --num-workers 0 \
  --weight-decay 1.0 \
  --warmup-tokens 80000000 \
  --max-generate-steps 96 \
  --output-dir "$NF_REF_DIR"
run_py neural-benchmark/plot_benchmark_history.py --results-json "$NF_REF_DIR/results.json"
mkdir -p "$NF_REF_DIR/formal_eval"
run_py neural-benchmark/faithful_neuroformer.py \
  --mode eval \
  --obs-window 0.15 \
  --pred-window 0.05 \
  --batch-size 8 \
  --grad-accum-steps 20 \
  --num-workers 0 \
  --weight-decay 1.0 \
  --warmup-tokens 80000000 \
  --max-generate-steps 96 \
  --checkpoint-path "$NF_REF_DIR/best_model.pt" \
  --output-dir "$NF_REF_DIR/formal_eval" \
  --eval-split both \
  --inference-mode both

run_py neural-benchmark/compare_faithful_neuroformer.py \
  --canonical-json "$NF_CANON_DIR/formal_eval/eval_results.json" \
  --reference-json "$NF_REF_DIR/formal_eval/eval_results.json" \
  --output-dir results/logs/phase1_benchmark_repro_faithful_neuroformer_compare_e50_aligned \
  --split test
