#!/usr/bin/env bash
set -euo pipefail

MODE="dry-run"
PREFERENCES="data/final/preferences_train.jsonl"
PREFERENCES_VAL="data/final/preferences_val.jsonl"
SFT_PROMPTS_SOURCE="data/final/sft_train.jsonl"
PROMPTS_OUT="data/rlhf_prompts.jsonl"
BASE_MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
SFT_MODEL="results_sft/adapter"
REWARD_OUT="results_reward"
PPO_OUT="results_ppo"
EVAL_OUT="eval_report_ppo.json"
REPO="."
SKIP_REWARD_IF_EXISTS=0
PYTHON_BIN="python"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --train) MODE="train"; shift ;;
    --preferences) PREFERENCES="$2"; shift 2 ;;
    --preferences-val) PREFERENCES_VAL="$2"; shift 2 ;;
    --sft-prompts-source) SFT_PROMPTS_SOURCE="$2"; shift 2 ;;
    --prompts-out) PROMPTS_OUT="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --sft-model) SFT_MODEL="$2"; shift 2 ;;
    --reward-out) REWARD_OUT="$2"; shift 2 ;;
    --ppo-out) PPO_OUT="$2"; shift 2 ;;
    --eval-out) EVAL_OUT="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --skip-reward-if-exists) SKIP_REWARD_IF_EXISTS=1; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "PPO-based RLHF pipeline"
echo "Mode:         $MODE"
echo "Preferences:  $PREFERENCES"
echo "Pref val:     $PREFERENCES_VAL"
echo "SFT prompts:  $SFT_PROMPTS_SOURCE"
echo "Prompts out:  $PROMPTS_OUT"
echo "Base model:   $BASE_MODEL"
echo "SFT model:    $SFT_MODEL"
echo "Reward out:   $REWARD_OUT"
echo "PPO out:      $PPO_OUT"
echo "Eval out:     $EVAL_OUT"
echo "Skip reward:  $SKIP_REWARD_IF_EXISTS"
echo "Python:       $PYTHON_BIN"
echo "========================================"

if [[ ! -f "$PREFERENCES" ]]; then
  echo "Missing preference file: $PREFERENCES"
  exit 1
fi
if [[ ! -f "$PREFERENCES_VAL" ]]; then
  echo "Missing preference validation file: $PREFERENCES_VAL"
  exit 1
fi
if [[ ! -f "$SFT_PROMPTS_SOURCE" ]]; then
  echo "Missing SFT prompt source file: $SFT_PROMPTS_SOURCE"
  exit 1
fi

echo ""
echo "0. Audit final datasets"
"$PYTHON_BIN" scripts/audit_datasets.py \
  --sft "$SFT_PROMPTS_SOURCE" \
  --preferences "$PREFERENCES" "$PREFERENCES_VAL"

echo ""
echo "1. Validate canonical preference data"
"$PYTHON_BIN" -m alignment.validate_preferences "$PREFERENCES"
"$PYTHON_BIN" -m alignment.validate_preferences "$PREFERENCES_VAL"

echo ""
echo "2. Build PPO prompt dataset"
"$PYTHON_BIN" -m alignment.reward_dataset \
  --preferences "$PREFERENCES" \
  --synthetic "$SFT_PROMPTS_SOURCE" \
  --out "$PROMPTS_OUT"

REWARD_CMD=(
  "$PYTHON_BIN" -m alignment.reward_model
  --data "$PREFERENCES"
  --eval-data "$PREFERENCES_VAL"
  --base-model "$BASE_MODEL"
  --out "$REWARD_OUT"
  --max-steps 200
)

PPO_CMD=(
  "$PYTHON_BIN" -m alignment.ppo_trainer
  --prompts "$PROMPTS_OUT"
  --sft-model "$SFT_MODEL"
  --reward-model "$REWARD_OUT"
  --base-model "$BASE_MODEL"
  --out "$PPO_OUT"
  --total-episodes 1000
  --learning-rate 3e-6
  --mini-batch-size 1
  --batch-size 4
  --target-kl 0.1
)

EVAL_CMD=(
  "$PYTHON_BIN" -m eval.eval
  --repo "$REPO"
  --model "$PPO_OUT/adapter"
  --backend hf-peft
  --base-model "$BASE_MODEL"
  --reward-model "$REWARD_OUT"
  --benchmark eval/benchmark_self.json
  --out "$EVAL_OUT"
  --verbose
)

echo ""
echo "3. Reward model command"
printf '  %q' "${REWARD_CMD[@]}"
echo ""

echo ""
echo "4. PPO command"
printf '  %q' "${PPO_CMD[@]}"
echo ""

echo ""
echo "5. Evaluation command"
printf '  %q' "${EVAL_CMD[@]}"
echo ""

if [[ "$MODE" == "dry-run" ]]; then
  echo ""
  echo "Running module dry-runs..."
  "${REWARD_CMD[@]}" --dry-run
  "${PPO_CMD[@]}" --dry-run
  echo ""
  echo "Dry-run completed. No long GPU training was started."
  exit 0
fi

echo ""
if [[ "$SKIP_REWARD_IF_EXISTS" -eq 1 && -d "$REWARD_OUT/adapter" ]]; then
  echo "6. Reusing existing reward adapter at $REWARD_OUT/adapter"
else
  echo "6. Train reward model"
  "${REWARD_CMD[@]}"
fi

echo ""
echo "7. Run PPO training"
"${PPO_CMD[@]}"

echo ""
echo "8. Evaluate PPO adapter"
"${EVAL_CMD[@]}"

echo ""
echo "RLHF PPO pipeline complete."
