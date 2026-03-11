#!/bin/bash
# Phase 1 v2 sequential training + evaluation script
# Runs all 6 conditions sequentially: 250ms-cont → 250ms-trial → ... → 1000ms-trial
# After each training, immediately runs eval_phase1_v2.py

set -e
cd /root/autodl-tmp/NeuroHorizon
PYTHON=/root/miniconda3/envs/poyo/bin/python

CONFIGS=(
    "train_v2_250ms"
    "train_v2_250ms_trial"
    "train_v2_500ms"
    "train_v2_500ms_trial"
    "train_v2_1000ms"
    "train_v2_1000ms_trial"
)

LOG_DIRS=(
    "results/logs/phase1_v2_250ms_cont"
    "results/logs/phase1_v2_250ms_trial"
    "results/logs/phase1_v2_500ms_cont"
    "results/logs/phase1_v2_500ms_trial"
    "results/logs/phase1_v2_1000ms_cont"
    "results/logs/phase1_v2_1000ms_trial"
)

echo "=== Phase 1 v2 Training Pipeline ==="
echo "Start time: $(date)"
echo ""

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    log_dir="${LOG_DIRS[$i]}"

    # Skip if already trained (checkpoint exists)
    if [ -f "${log_dir}/lightning_logs/version_0/checkpoints/last.ckpt" ]; then
        echo "[$((i+1))/6] ${config}: SKIPPING (checkpoint exists)"
        echo ""
    else
        echo "[$((i+1))/6] Training: ${config}"
        echo "  Log dir: ${log_dir}"
        echo "  Start: $(date)"

        $PYTHON examples/neurohorizon/train.py --config-name=${config}

        echo "  Done: $(date)"
        echo ""
    fi

    # Evaluate (always, even if training was skipped)
    echo "[$((i+1))/6] Evaluating: ${config}"
    $PYTHON scripts/analysis/neurohorizon/eval_phase1_v2.py \
        --log-dir "${log_dir}" \
        --batch-size 64 \
        --trial-batch-size 16

    echo "  Eval done: $(date)"
    echo "=========================================="
    echo ""
done

echo "=== All training and evaluation complete ==="
echo "End time: $(date)"
