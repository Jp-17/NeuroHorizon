#!/bin/bash
# Auto-launch NeuroHorizon v2_beh and v2_mm after v1 completes.
# Waits for all GPU-consuming processes to finish before launching.
#
# Usage:
#   nohup bash scripts/auto_launch_after_v1.sh &
#   tail -f logs/auto_launch_after_v1.log

set -e
cd "$(dirname "$0")/.."

LOGFILE="logs/auto_launch_after_v1.log"
mkdir -p logs results

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

wait_for_pid() {
    local PID=$1
    local NAME=$2
    if python3 -c "import os; os.kill($PID, 0)" 2>/dev/null; then
        log "Waiting for $NAME (PID $PID)..."
        while python3 -c "import os; os.kill($PID, 0)" 2>/dev/null; do
            sleep 120
        done
        log "$NAME (PID $PID) finished."
        sleep 10
    else
        log "$NAME (PID $PID) already finished."
    fi
}

log "Auto-launch pipeline started."

# Wait for NH v1 to finish (PID 34659)
wait_for_pid 34659 "NH v1"

# Run v1 evaluation
log "Running post-training evaluation on v1..."
V1_CKPT=$(ls -t logs/neurohorizon/lightning_logs/version_5/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n "$V1_CKPT" ]; then
    log "Using v1 checkpoint: $V1_CKPT"
    conda run -n poyo python scripts/evaluate_neurohorizon.py \
        --ckpt "$V1_CKPT" \
        --data-dir /root/autodl-tmp/datasets/ibl_processed \
        --output-dir results/neurohorizon_v1_final_eval \
        --use-raw-features \
        --n-windows 50 \
        2>&1 | tee -a logs/v1_final_eval.log || log "WARNING: v1 evaluation failed"
fi

# Check if POYO is also running and wait
POYO_PID=$(python3 -c "
import subprocess
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'train.py' in line and 'poyo_baseline' in line and 'grep' not in line:
        parts = line.split()
        print(parts[1])
        break
" 2>/dev/null)

if [ -n "$POYO_PID" ]; then
    wait_for_pid "$POYO_PID" "POYO baseline"
fi

# Ensure GPU is free
log "Checking GPU availability..."
sleep 15

# Launch v2_beh (IBL + Allen + behavior conditioning)
log "Launching NH v2_beh (IBL+Allen, behavior conditioning, 200 epochs)..."
conda run -n poyo python examples/neurohorizon/train.py \
    --config-name train_v2 \
    num_workers=4 gpus=1 \
    2>&1 | tee -a logs/v2_beh_training_v2.log
V2_BEH_EXIT=$?
log "v2_beh finished with exit code: $V2_BEH_EXIT"

# Evaluate v2_beh
sleep 5
V2_BEH_CKPT=$(ls -t logs/neurohorizon_v2_beh/lightning_logs/*/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n "$V2_BEH_CKPT" ]; then
    log "Running v2_beh evaluation..."
    conda run -n poyo python scripts/evaluate_neurohorizon.py \
        --ckpt "$V2_BEH_CKPT" \
        --data-dir /root/autodl-tmp/datasets/ibl_processed \
        --output-dir results/neurohorizon_v2_beh_eval \
        --n-windows 50 \
        2>&1 | tee -a logs/v2_beh_eval.log || log "WARNING: v2_beh evaluation failed"
fi

# Launch v2_mm (full multimodal)
log "Launching NH v2_mm (full multimodal, 200 epochs)..."
conda run -n poyo python examples/neurohorizon/train.py \
    --config-name train_v2_mm \
    num_workers=4 gpus=1 \
    2>&1 | tee -a logs/v2_mm_training_v2.log
V2_MM_EXIT=$?
log "v2_mm finished with exit code: $V2_MM_EXIT"

# Evaluate v2_mm
sleep 5
V2_MM_CKPT=$(ls -t logs/neurohorizon_v2_mm/lightning_logs/*/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n "$V2_MM_CKPT" ]; then
    log "Running v2_mm evaluation..."
    conda run -n poyo python scripts/evaluate_neurohorizon.py \
        --ckpt "$V2_MM_CKPT" \
        --data-dir /root/autodl-tmp/datasets/ibl_processed \
        --output-dir results/neurohorizon_v2_mm_eval \
        --n-windows 50 \
        2>&1 | tee -a logs/v2_mm_eval.log || log "WARNING: v2_mm evaluation failed"
fi

# Collect all results
log "Collecting final results..."
conda run -n poyo python scripts/collect_results.py --output results/final_summary \
    2>&1 | tee -a "$LOGFILE" || log "WARNING: results collection failed"

log "====================================="
log "Full pipeline complete!"
log "====================================="
