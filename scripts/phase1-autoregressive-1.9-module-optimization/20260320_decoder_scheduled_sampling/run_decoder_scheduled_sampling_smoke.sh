#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/poyo/bin/python}"
EVAL_SCRIPT="$ROOT/scripts/analysis/neurohorizon/eval_phase1_v2.py"
VERIFY_SCRIPT="$ROOT/scripts/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/verify_decoder_scheduled_sampling.py"
MODULE_LOG_ROOT="$ROOT/results/logs/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling_smoke"

cd "$ROOT"
mkdir -p "$MODULE_LOG_ROOT"
"$PYTHON" "$VERIFY_SCRIPT"

config_for_window() {
  case "$1" in
    250ms) echo "train_1p9_decoder_scheduled_sampling_250ms" ;;
    500ms) echo "train_1p9_decoder_scheduled_sampling_500ms" ;;
    1000ms) echo "train_1p9_decoder_scheduled_sampling_1000ms" ;;
    *) echo "unknown window: $1" >&2; exit 1 ;;
  esac
}

setting_overrides() {
  case "$1" in
    memory_only_mix035)
      echo "model.prediction_memory_train_mix_prob=0.35 model.decoder_train_mode=parallel_teacher_forced model.decoder_rollout_prob_mode=fixed model.decoder_rollout_prob=0.0"
      ;;
    decoder_ss_fixed_050)
      echo "model.prediction_memory_train_mix_prob=0.0 model.decoder_train_mode=scheduled_sampling model.decoder_rollout_prob_mode=fixed model.decoder_rollout_prob=0.5"
      ;;
    decoder_ss_linear_0_to_050)
      echo "model.prediction_memory_train_mix_prob=0.0 model.decoder_train_mode=scheduled_sampling model.decoder_rollout_prob_mode=linear_ramp model.decoder_rollout_prob_start=0.0 model.decoder_rollout_prob_end=0.5 model.decoder_rollout_prob_ramp_epochs=150"
      ;;
    hybrid_mix035_plus_linear_050)
      echo "model.prediction_memory_train_mix_prob=0.35 model.decoder_train_mode=scheduled_sampling model.decoder_rollout_prob_mode=linear_ramp model.decoder_rollout_prob_start=0.0 model.decoder_rollout_prob_end=0.5 model.decoder_rollout_prob_ramp_epochs=150"
      ;;
    *)
      echo "unknown setting: $1" >&2
      exit 1
      ;;
  esac
}

run_eval_pair() {
  local log_dir="$1"
  local split="$2"
  local suffix="$3"

  "$PYTHON" "$EVAL_SCRIPT" \
    --log-dir "$log_dir" \
    --checkpoint-kind best \
    --split "$split" \
    --skip-trial \
    --output "$log_dir/eval_teacher_forced_${suffix}_${split}.json"

  "$PYTHON" "$EVAL_SCRIPT" \
    --log-dir "$log_dir" \
    --checkpoint-kind best \
    --split "$split" \
    --rollout \
    --skip-trial \
    --output "$log_dir/eval_rollout_${suffix}_${split}.json"
}

SETTINGS=(
  memory_only_mix035
  decoder_ss_fixed_050
  decoder_ss_linear_0_to_050
  hybrid_mix035_plus_linear_050
)
WINDOWS=(250ms 500ms 1000ms)

for setting in "${SETTINGS[@]}"; do
  read -r -a overrides <<< "$(setting_overrides "$setting")"
  for window in "${WINDOWS[@]}"; do
    config_name="$(config_for_window "$window")"
    window_log_dir="$MODULE_LOG_ROOT/$setting/$window"
    mkdir -p "$window_log_dir"

    echo "[$(date '+%F %T')] smoke train start setting=$setting window=$window"
    "$PYTHON" examples/neurohorizon/train.py \
      --config-name="$config_name" \
      log_dir="$window_log_dir" \
      epochs=1 \
      eval_epochs=1 \
      num_workers=0 \
      eval_batch_size=64 \
      wandb.run_name="${setting}_${window}_smoke" \
      "${overrides[@]}" \
      > "$window_log_dir/stdout.log"

    run_eval_pair "$window_log_dir" "valid" "best"
    echo "[$(date '+%F %T')] smoke train done setting=$setting window=$window"
  done
done
