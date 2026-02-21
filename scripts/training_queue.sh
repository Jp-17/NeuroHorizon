#!/bin/bash
# Sequential training queue for NeuroHorizon experiments.
# Runs multiple training variants one after another on a single GPU.
#
# Usage:
#   nohup bash scripts/training_queue.sh &
#   tail -f logs/training_queue.log

set -e
cd "$(dirname "$0")/.."

LOGFILE="logs/training_queue.log"
mkdir -p logs

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

wait_for_pid() {
    local PID=$1
    local NAME=$2
    log "Waiting for $NAME (PID $PID) to finish..."
    while kill -0 "$PID" 2>/dev/null; do
        sleep 60
    done
    log "$NAME (PID $PID) finished."
    sleep 5
}

run_training() {
    local NAME=$1
    local CONFIG=$2
    shift 2
    local EXTRA_ARGS="$@"

    log "====================================="
    log "Starting: $NAME"
    log "  Config: $CONFIG"
    log "  Extra args: $EXTRA_ARGS"
    log "====================================="

    conda run -n poyo python examples/neurohorizon/train.py \
        --config-name "$CONFIG" \
        num_workers=4 gpus=1 \
        $EXTRA_ARGS \
        2>&1 | tee -a "logs/${NAME}_training.log"

    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        log "$NAME completed successfully."
    else
        log "WARNING: $NAME exited with code $EXIT_CODE"
    fi
    return $EXIT_CODE
}

log "Training queue started."

# Step 1: Wait for any running NH v1 training
NH_V1_PID=$(ps aux | grep "train.py.*train_v2_ibl\|train.py.*--config-name train$" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$NH_V1_PID" ]; then
    wait_for_pid "$NH_V1_PID" "NH v1/v2_ibl training"
fi

# Step 2: Run v2 IBL-only (if not already completed)
V2_IBL_CKPT="logs/neurohorizon_v2_ibl/lightning_logs/version_0/checkpoints/last.ckpt"
if [ ! -f "$V2_IBL_CKPT" ]; then
    run_training "v2_ibl" "train_v2_ibl"
else
    log "v2_ibl already completed, skipping."
fi

# Step 3: Run v2 behavior (IBL + Allen + behavior conditioning)
V2_BEH_CKPT="logs/neurohorizon_v2_beh/lightning_logs/version_0/checkpoints/last.ckpt"
if [ ! -f "$V2_BEH_CKPT" ]; then
    run_training "v2_beh" "train_v2"
else
    log "v2_beh already completed, skipping."
fi

# Step 4: Run v2 multimodal (IBL + Allen + behavior + image)
V2_MM_CKPT="logs/neurohorizon_v2_mm/lightning_logs/version_0/checkpoints/last.ckpt"
if [ ! -f "$V2_MM_CKPT" ]; then
    run_training "v2_mm" "train_v2_mm"
else
    log "v2_mm already completed, skipping."
fi

# Step 5: Run key ablations (IDEncoder variants)
log "Starting ablation experiments..."

run_training "ablation_random_emb" "train_v2_ibl" \
    "model.embedding_mode=random" \
    "log_dir=./logs/ablation_id_encoder_random_emb" \
    "epochs=50"

run_training "ablation_mean_emb" "train_v2_ibl" \
    "model.embedding_mode=mean" \
    "log_dir=./logs/ablation_id_encoder_mean_emb" \
    "epochs=50"

# Step 6: Collect all results
log "Collecting results..."
python3 scripts/collect_results.py --output results/final_summary 2>&1 | tee -a "$LOGFILE"

log "====================================="
log "Training queue complete!"
log "====================================="
