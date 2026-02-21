#!/bin/bash
# Auto-launch NeuroHorizon v2 training after POYO baseline completes.
# Monitors the POYO training process and launches v2 when GPU frees up.
#
# Usage:
#   nohup bash scripts/auto_launch_v2.sh &
#   # Check progress: tail -f logs/auto_launch.log

set -e
cd "$(dirname "$0")/.."

LOGFILE="logs/auto_launch.log"
mkdir -p logs

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

# Find POYO training PID
POYO_PID=$(ps aux | grep "train_baseline.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$POYO_PID" ]; then
    log "No POYO training process found. Launching v2 immediately."
else
    log "Found POYO training PID: $POYO_PID"
    log "Waiting for POYO to finish..."

    while kill -0 "$POYO_PID" 2>/dev/null; do
        sleep 120
    done

    log "POYO training finished."
    sleep 10  # Wait for GPU memory to free
fi

log "Launching NH v2 (IBL-only, normalized features)..."
conda run -n poyo python examples/neurohorizon/train.py \
    --config-name train_v2_ibl \
    num_workers=4 gpus=1 \
    2>&1 | tee -a logs/v2_ibl_training.log &
V2_PID=$!
log "NH v2 (IBL) launched, PID: $V2_PID"

log "Auto-launch complete. Monitor with: tail -f logs/v2_ibl_training.log"
