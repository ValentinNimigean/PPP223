#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment if present
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

REPO="."
TEACHER_MODEL="gpt-4o"
STUDENT_URL=""
OLLAMA_BASE_MODEL="qwen2.5-coder:1.5b"
SFT_BASE_MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
SFT_TRAIN_DATA="data/final/sft_train.jsonl"
SFT_VAL_DATA="data/final/sft_val.jsonl"
PREF_TRAIN_DATA="data/final/preferences_train.jsonl"
PREF_VAL_DATA="data/final/preferences_val.jsonl"

SKIP_SYNTH=0
SKIP_SFT=0
SKIP_DPO=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --teacher-model) TEACHER_MODEL="$2"; shift 2 ;;
    --student-url) STUDENT_URL="$2"; shift 2 ;;
    --ollama-base-model) OLLAMA_BASE_MODEL="$2"; shift 2 ;;
    --sft-base-model) SFT_BASE_MODEL="$2"; shift 2 ;;
    --skip-synth) SKIP_SYNTH=1; shift ;;
    --skip-sft) SKIP_SFT=1; shift ;;
    --skip-dpo) SKIP_DPO=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

combine_if_exists() {
  local out="$1"
  shift

  rm -f "$out"
  local found=0

  for f in "$@"; do
    if [[ -f "$f" ]]; then
      cat "$f" >> "$out"
      found=1
    else
      echo "Skipping missing file: $f"
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "No input files found for $out"
    return 1
  fi

  echo "Wrote $out with $(wc -l < "$out") rows"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required dataset: $path"
    exit 1
  fi
}

echo "========================================"
echo "PPP223 / Mobtrap training pipeline"
echo "Repo:              $REPO"
echo "Teacher model:     $TEACHER_MODEL"
echo "Ollama eval model: $OLLAMA_BASE_MODEL"
echo "SFT base model:    $SFT_BASE_MODEL"
echo "SFT train data:    $SFT_TRAIN_DATA"
echo "SFT val data:      $SFT_VAL_DATA"
echo "Pref train data:   $PREF_TRAIN_DATA"
echo "Pref val data:     $PREF_VAL_DATA"
echo "========================================"

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 0: Base eval before training"
  echo "========================================"
  python -m eval.eval \
    --repo "$REPO" \
    --model "$OLLAMA_BASE_MODEL" \
    --benchmark eval/benchmark_self.json \
    --out eval_report_base.json \
    --verbose || true
fi

if [[ "$SKIP_SYNTH" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 1: Generate raw synthetic helper data (not used directly for training)"
  echo "========================================"
  python data/synth.py \
    --repo "$REPO" \
    --out synthetic_qa_auto.jsonl \
    --model "$TEACHER_MODEL"
fi

echo ""
echo "========================================"
echo "STEP 2: Verify final cleaned datasets"
echo "========================================"
require_file "$SFT_TRAIN_DATA"
require_file "$SFT_VAL_DATA"
require_file "$PREF_TRAIN_DATA"
require_file "$PREF_VAL_DATA"
python scripts/audit_datasets.py \
  --sft "$SFT_TRAIN_DATA" "$SFT_VAL_DATA" \
  --preferences "$PREF_TRAIN_DATA" "$PREF_VAL_DATA"

if [[ "$SKIP_SFT" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 3: Dry-run SFT formatting"
  echo "========================================"
  python -m model.finetune \
    --data "$SFT_TRAIN_DATA" \
    --val-data "$SFT_VAL_DATA" \
    --model "$SFT_BASE_MODEL" \
    --out results_sft \
    --dry-run \
    --max-seq-length 2048

  echo ""
  echo "========================================"
  echo "STEP 4: SFT fine-tuning"
  echo "========================================"
  python -m model.finetune \
    --data "$SFT_TRAIN_DATA" \
    --val-data "$SFT_VAL_DATA" \
    --model "$SFT_BASE_MODEL" \
    --out results_sft \
    --epochs 3 \
    --learning-rate 1e-4 \
    --lora-r 8 \
    --lora-alpha 16 \
    --eval-split 0
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 5: Eval current Ollama model after code fixes"
  echo "========================================"
  echo "NOTE: This evaluates '$OLLAMA_BASE_MODEL'. It does NOT automatically evaluate results_sft/adapter."
  echo "To evaluate the fine-tuned model, export/serve the adapter first."
  python -m eval.eval \
    --repo "$REPO" \
    --model "$OLLAMA_BASE_MODEL" \
    --benchmark eval/benchmark_self.json \
    --out eval_report_post_codefix_or_base.json \
    --verbose || true
fi

if [[ "$SKIP_DPO" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 6: Final preference dataset audit"
  echo "========================================"
  python -m alignment.validate_preferences "$PREF_TRAIN_DATA"
  python -m alignment.validate_preferences "$PREF_VAL_DATA"

  echo ""
  echo "========================================"
  echo "STEP 7: DPO training baseline on final preference train split"
  echo "========================================"

  if [[ ! -d results_sft/adapter ]]; then
    echo "Missing results_sft/adapter. Run SFT before DPO."
    exit 1
  fi

  python model/dpo.py \
    --dpo-data-path "$PREF_TRAIN_DATA" \
    --out results_dpo \
    --sft-adapter results_sft/adapter \
    --max-steps 200
fi

echo ""
echo "========================================"
echo "DONE"
echo "========================================"

if [[ -f eval_report_base.json ]]; then
  python - <<'PY'
import json
from pathlib import Path

for path in ["eval_report_base.json", "eval_report_post_codefix_or_base.json"]:
    p = Path(path)
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    o = data["overall"]
    print(f"{path}: combined={o['combined_score']:.3f}, penalized={o['penalized_combined_score']:.3f}")
PY
fi

echo ""
echo "Important next step:"
echo "results_sft/adapter and results_dpo/adapter are LoRA adapters."
echo "Ollama eval will not reflect them until you export/merge/serve the adapter as an actual model."
echo "Do not interpret eval_report_post_codefix_or_base.json as fine-tuned-model performance unless '$OLLAMA_BASE_MODEL' points to an exported fine-tuned model."
