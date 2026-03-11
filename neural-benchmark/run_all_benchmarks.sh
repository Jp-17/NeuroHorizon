#!/bin/bash
# Run all benchmark experiments: 3 models x 3 prediction windows
# Each model trains for 300 epochs

set -e
cd /root/autodl-tmp/NeuroHorizon
source /root/miniconda3/etc/profile.d/conda.sh
conda activate benchmark-env

EPOCHS=300
BATCH=64

echo "=========================================="
echo "Starting full benchmark suite"
echo "Models: ndt2, ibl_mtm, neuroformer"
echo "Windows: 250ms, 500ms, 1000ms"
echo "Epochs: $EPOCHS, Batch: $BATCH"
echo "=========================================="

for MODEL in ndt2 ibl_mtm neuroformer; do
    for WINDOW in 0.25 0.5 1.0; do
        WINDOW_MS=$(echo "$WINDOW * 1000" | bc | cut -d. -f1)
        echo ""
        echo ">>> Starting: ${MODEL} @ ${WINDOW_MS}ms"
        echo ">>> $(date)"
        
        python3 neural-benchmark/benchmark_train.py \
            --model $MODEL \
            --pred_window $WINDOW \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --seed 42
        
        echo ">>> Finished: ${MODEL} @ ${WINDOW_MS}ms"
        echo ">>> $(date)"
    done
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
