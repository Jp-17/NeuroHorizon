#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/NeuroHorizon"
PYTHON="/root/miniconda3/envs/poyo/bin/python"
EVAL_SCRIPT="$ROOT/scripts/analysis/neurohorizon/eval_phase1_v2.py"
COLLECT_SCRIPT="$ROOT/scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/collect_local_prediction_memory_results.py"
MODULE_LOG_ROOT="$ROOT/results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory"

cd "$ROOT"
mkdir -p "$MODULE_LOG_ROOT/250ms" "$MODULE_LOG_ROOT/500ms" "$MODULE_LOG_ROOT/1000ms"

"$PYTHON" scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/verify_local_prediction_memory.py

run_window() {
  local window="$1"
  local config_name="$2"
  local window_log_dir="$MODULE_LOG_ROOT/$window"

  mkdir -p "$window_log_dir"

  (
    echo "[$(date '+%F %T')] start train $window"
    "$PYTHON" examples/neurohorizon/train.py \
      --config-name="$config_name" \
      num_workers=2

    echo "[$(date '+%F %T')] start teacher-forced eval $window"
    "$PYTHON" "$EVAL_SCRIPT" \
      --log-dir "$window_log_dir" \
      --skip-trial \
      --output "$window_log_dir/eval_teacher_forced.json"

    echo "[$(date '+%F %T')] start rollout eval $window"
    "$PYTHON" "$EVAL_SCRIPT" \
      --log-dir "$window_log_dir" \
      --rollout \
      --skip-trial \
      --output "$window_log_dir/eval_rollout.json"

    echo "[$(date '+%F %T')] finished $window"
  ) > "$window_log_dir/stdout.log" 2>&1
}

run_window "250ms" "train_1p9_local_prediction_memory_250ms" &
pid_250=$!
echo "$pid_250" > "$MODULE_LOG_ROOT/250ms/job.pid"
echo "250ms job pid=$pid_250"

run_window "500ms" "train_1p9_local_prediction_memory_500ms" &
pid_500=$!
echo "$pid_500" > "$MODULE_LOG_ROOT/500ms/job.pid"
echo "500ms job pid=$pid_500"

run_window "1000ms" "train_1p9_local_prediction_memory_1000ms" &
pid_1000=$!
echo "$pid_1000" > "$MODULE_LOG_ROOT/1000ms/job.pid"
echo "1000ms job pid=$pid_1000"

wait "$pid_250" "$pid_500" "$pid_1000"

"$PYTHON" "$COLLECT_SCRIPT"
