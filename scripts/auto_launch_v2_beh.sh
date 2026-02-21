#!/bin/bash
# Auto-launch NeuroHorizon v2_beh training after v2_ibl completes.
# Also runs post-training evaluation on v2_ibl before launching.
#
# Usage:
#   nohup bash scripts/auto_launch_v2_beh.sh &
#   tail -f logs/auto_launch_v2_beh.log

set -e
cd "$(dirname "$0")/.."

LOGFILE="logs/auto_launch_v2_beh.log"
mkdir -p logs results

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

# Find v2_ibl training PID
V2_PID=$(python3 -c "
import os
for pid in [87666]:
    try:
        os.kill(pid, 0)
        print(pid)
    except:
        pass
" 2>/dev/null)

# Also check by process name
if [ -z "$V2_PID" ]; then
    V2_PID=$(python3 -c "
import subprocess
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'train.py' in line and 'train_v2_ibl' in line and 'grep' not in line:
        print(line.split()[1])
        break
" 2>/dev/null)
fi

if [ -z "$V2_PID" ]; then
    log "No v2_ibl training process found. Checking if already completed..."
    V2_CKPT="logs/neurohorizon_v2_ibl/lightning_logs/version_0/checkpoints/last.ckpt"
    if [ -f "$V2_CKPT" ]; then
        log "v2_ibl checkpoint found at $V2_CKPT"
    else
        log "WARNING: No v2_ibl process running and no checkpoint found."
        log "Proceeding anyway..."
    fi
else
    log "Found v2_ibl training PID: $V2_PID"
    log "Waiting for v2_ibl to finish..."

    while python3 -c "import os; os.kill($V2_PID, 0)" 2>/dev/null; do
        sleep 120
    done

    log "v2_ibl training finished."
    sleep 10  # Wait for GPU memory to free
fi

# Step 1: Run evaluation on v2_ibl
log "Running post-training evaluation on v2_ibl..."
V2_CKPT=$(ls -t logs/neurohorizon_v2_ibl/lightning_logs/version_0/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n "$V2_CKPT" ]; then
    log "Using checkpoint: $V2_CKPT"
    conda run -n poyo python scripts/evaluate_neurohorizon.py \
        --ckpt "$V2_CKPT" \
        --data-dir /root/autodl-tmp/datasets/ibl_processed \
        --output-dir results/neurohorizon_v2_ibl_eval \
        2>&1 | tee -a logs/v2_ibl_eval.log || log "WARNING: v2_ibl evaluation failed"
    log "v2_ibl evaluation complete."
else
    log "WARNING: No v2_ibl checkpoint found for evaluation."
fi

# Step 2: Launch v2_beh (IBL + Allen + behavior conditioning)
log "Launching NH v2_beh (IBL+Allen, behavior conditioning)..."
conda run -n poyo python examples/neurohorizon/train.py \
    --config-name train_v2 \
    num_workers=4 gpus=1 \
    2>&1 | tee -a logs/v2_beh_training.log &
V2_BEH_PID=$!
log "NH v2_beh launched, PID: $V2_BEH_PID"

# Wait for v2_beh to complete
log "Waiting for v2_beh to finish..."
wait $V2_BEH_PID
V2_BEH_EXIT=$?
log "v2_beh finished with exit code: $V2_BEH_EXIT"

# Step 3: Run evaluation on v2_beh
sleep 5
log "Running post-training evaluation on v2_beh..."
V2_BEH_CKPT=$(ls -t logs/neurohorizon_v2_beh/lightning_logs/version_0/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n "$V2_BEH_CKPT" ]; then
    conda run -n poyo python scripts/evaluate_neurohorizon.py \
        --ckpt "$V2_BEH_CKPT" \
        --data-dir /root/autodl-tmp/datasets/ibl_processed \
        --output-dir results/neurohorizon_v2_beh_eval \
        2>&1 | tee -a logs/v2_beh_eval.log || log "WARNING: v2_beh evaluation failed"
fi

# Step 4: Launch v2_mm (full multimodal)
log "Launching NH v2_mm (full multimodal: image + behavior)..."
conda run -n poyo python examples/neurohorizon/train.py \
    --config-name train_v2_mm \
    num_workers=4 gpus=1 \
    2>&1 | tee -a logs/v2_mm_training.log &
V2_MM_PID=$!
log "NH v2_mm launched, PID: $V2_MM_PID"

log "Auto-launch pipeline complete. v2_mm training in progress."
log "Monitor with: tail -f logs/v2_mm_training.log"
