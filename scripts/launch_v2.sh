#!/bin/bash
# Launch NeuroHorizon v2 training
# Run this after POYO baseline finishes to free GPU memory
#
# v2a: IBL-only with normalized features (comparison to v1)
# v2b: IBL+Allen with behavior multimodal conditioning
#
# Usage:
#   bash scripts/launch_v2.sh ibl      # v2a: IBL-only
#   bash scripts/launch_v2.sh combined  # v2b: IBL+Allen+behavior
#   bash scripts/launch_v2.sh both      # Launch v2a first, then v2b

set -e
cd "$(dirname "$0")/.."

VARIANT=${1:-ibl}

launch_ibl() {
    echo "=== Launching NH v2a: IBL-only with normalized features ==="
    conda run -n poyo python examples/neurohorizon/train.py \
        --config-name train_v2_ibl \
        batch_size=16 num_workers=4 gpus=1 \
        2>&1 | tee logs/v2_ibl_training.log &
    echo "PID: $!"
}

launch_combined() {
    echo "=== Launching NH v2b: IBL+Allen with behavior ==="
    conda run -n poyo python examples/neurohorizon/train.py \
        --config-name train_v2 \
        batch_size=16 num_workers=4 gpus=1 \
        2>&1 | tee logs/v2_combined_training.log &
    echo "PID: $!"
}

case "$VARIANT" in
    ibl)
        launch_ibl
        ;;
    combined)
        launch_combined
        ;;
    both)
        echo "Launching v2a (IBL-only) first..."
        launch_ibl
        V2A_PID=$!
        echo "Waiting for v2a to finish..."
        wait $V2A_PID
        echo "v2a finished. Launching v2b (combined)..."
        launch_combined
        ;;
    *)
        echo "Usage: $0 {ibl|combined|both}"
        exit 1
        ;;
esac

echo "Training launched. Monitor with: tail -f logs/v2_*_training.log"
