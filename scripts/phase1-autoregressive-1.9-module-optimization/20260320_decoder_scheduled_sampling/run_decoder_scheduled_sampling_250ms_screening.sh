#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/poyo/bin/python}"
EVAL_SCRIPT="$ROOT/scripts/analysis/neurohorizon/eval_phase1_v2.py"
VERIFY_SCRIPT="$ROOT/scripts/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/verify_decoder_scheduled_sampling.py"
COLLECT_SCRIPT="$ROOT/scripts/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/collect_decoder_scheduled_sampling_250ms_screening.py"
MODULE_LOG_ROOT="$ROOT/results/logs/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling"
WINDOW="250ms"
CONFIG_NAME="train_1p9_decoder_scheduled_sampling_250ms"

SETTINGS=(
  memory_only_mix035
  decoder_ss_fixed_025
  decoder_ss_fixed_050
  decoder_ss_fixed_075
  decoder_ss_linear_0_to_050
  decoder_ss_linear_0_to_075
  hybrid_mix035_plus_linear_050
)

cd "$ROOT"
mkdir -p "$MODULE_LOG_ROOT"
"$PYTHON" "$VERIFY_SCRIPT"

setting_overrides() {
  case "$1" in
    memory_only_mix035)
      echo "model.prediction_memory_train_mix_prob=0.35 model.decoder_train_mode=parallel_teacher_forced model.decoder_rollout_prob_mode=fixed model.decoder_rollout_prob=0.0"
      ;;
    decoder_ss_fixed_025)
      echo "model.prediction_memory_train_mix_prob=0.0 model.decoder_train_mode=scheduled_sampling model.decoder_rollout_prob_mode=fixed model.decoder_rollout_prob=0.25"
      ;;
    decoder_ss_fixed_050)
      echo "model.prediction_memory_train_mix_prob=0.0 model.decoder_train_mode=scheduled_sampling model.decoder_rollout_prob_mode=fixed model.decoder_rollout_prob=0.5"
      ;;
    decoder_ss_fixed_075)
      echo "model.prediction_memory_train_mix_prob=0.0 model.decoder_train_mode=scheduled_sampling model.decoder_rollout_prob_mode=fixed model.decoder_rollout_prob=0.75"
      ;;
    decoder_ss_linear_0_to_050)
      echo "model.prediction_memory_train_mix_prob=0.0 model.decoder_train_mode=scheduled_sampling model.decoder_rollout_prob_mode=linear_ramp model.decoder_rollout_prob_start=0.0 model.decoder_rollout_prob_end=0.5 model.decoder_rollout_prob_ramp_epochs=150"
      ;;
    decoder_ss_linear_0_to_075)
      echo "model.prediction_memory_train_mix_prob=0.0 model.decoder_train_mode=scheduled_sampling model.decoder_rollout_prob_mode=linear_ramp model.decoder_rollout_prob_start=0.0 model.decoder_rollout_prob_end=0.75 model.decoder_rollout_prob_ramp_epochs=150"
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

is_complete() {
  local log_dir="$1"
  local required=(
    eval_teacher_forced_best_valid.json
    eval_teacher_forced_best_test.json
    eval_rollout_best_valid.json
    eval_rollout_best_test.json
  )

  for file in "${required[@]}"; do
    if [ ! -f "$log_dir/$file" ]; then
      return 1
    fi
  done
  return 0
}

archive_incomplete_dir() {
  local log_dir="$1"

  if [ ! -d "$log_dir" ]; then
    return 0
  fi
  if is_complete "$log_dir"; then
    return 0
  fi
  if find "$log_dir" -mindepth 1 -print -quit | grep -q .; then
    local backup="${log_dir}_interrupted_$(date '+%Y%m%d_%H%M%S')"
    mv "$log_dir" "$backup"
    echo "[$(date '+%F %T')] archived incomplete log dir: $backup"
  fi
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

for setting in "${SETTINGS[@]}"; do
  window_log_dir="$MODULE_LOG_ROOT/$setting/$WINDOW"

  if is_complete "$window_log_dir"; then
    echo "[$(date '+%F %T')] skip completed setting=$setting window=$WINDOW"
    continue
  fi

  archive_incomplete_dir "$window_log_dir"
  mkdir -p "$window_log_dir"
  read -r -a overrides <<< "$(setting_overrides "$setting")"

  echo "[$(date '+%F %T')] start train setting=$setting window=$WINDOW"
  "$PYTHON" examples/neurohorizon/train.py \
    --config-name="$CONFIG_NAME" \
    log_dir="$window_log_dir" \
    wandb.run_name="${setting}_${WINDOW}_screening" \
    "${overrides[@]}" \
    > "$window_log_dir/stdout.log"

  run_eval_pair "$window_log_dir" valid best
  run_eval_pair "$window_log_dir" test best
  echo "[$(date '+%F %T')] finished setting=$setting window=$WINDOW"
done

"$PYTHON" "$COLLECT_SCRIPT"
