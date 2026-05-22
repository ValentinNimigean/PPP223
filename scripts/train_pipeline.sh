#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment if present
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

REPO="."
TEACHER_MODEL="gpt-4o"
STUDENT_URL=""
OLLAMA_BASE_MODEL="${OLLAMA_MODEL:-qwen2.5-coder:1.5b}"
SFT_BASE_MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"

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

echo "========================================"
echo "PPP223 / Mobtrap training pipeline"
echo "Repo:              $REPO"
echo "Teacher model:     $TEACHER_MODEL"
echo "Ollama eval model: $OLLAMA_BASE_MODEL"
echo "SFT base model:    $SFT_BASE_MODEL"
echo "========================================"

mkdir -p training_data/sft training_data/preferences evaluation_reports

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 0: Base eval before training"
  echo "========================================"
  python eval/eval.py \
    --repo "$REPO" \
    --model "$OLLAMA_BASE_MODEL" \
    --benchmark eval/benchmark_self.json \
    --out evaluation_reports/eval_report_base.json \
    --verbose || true
fi

if [[ "$SKIP_SYNTH" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 1: Generate synthetic SFT data"
  echo "========================================"
  python data/synth.py \
    --repo "$REPO" \
    --out training_data/sft/synthetic_qa_auto.jsonl \
    --model "$TEACHER_MODEL"
fi

echo ""
echo "========================================"
echo "STEP 2: Combine SFT data"
echo "========================================"
combine_if_exists training_data/sft/synthetic_qa_combined.jsonl \
  training_data/sft/synthetic_qa_seed.jsonl \
  training_data/sft/synthetic_qa_auto.jsonl

python scripts/audit_training_data.py training_data/sft/synthetic_qa_combined.jsonl

if [[ "$SKIP_SFT" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 3: Dry-run SFT formatting"
  echo "========================================"
  python model/finetune.py \
    --data training_data/sft/synthetic_qa_combined.jsonl \
    --model "$SFT_BASE_MODEL" \
    --out results_sft \
    --dry-run \
    --max-seq-length 2048

  echo ""
  echo "========================================"
  echo "STEP 4: SFT fine-tuning"
  echo "========================================"
  python model/finetune.py \
    --data training_data/sft/synthetic_qa_combined.jsonl \
    --model "$SFT_BASE_MODEL" \
    --out results_sft \
    --epochs 3 \
    --learning-rate 1e-4 \
    --lora-r 8 \
    --lora-alpha 16 \
    --eval-split 0.1
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 5: Eval current Ollama model after code fixes"
  echo "========================================"
  echo "NOTE: This evaluates '$OLLAMA_BASE_MODEL'. It does NOT automatically evaluate results_sft/adapter."
  echo "To evaluate the fine-tuned model, export/serve the adapter first."
  python eval/eval.py \
    --repo "$REPO" \
    --model "$OLLAMA_BASE_MODEL" \
    --benchmark eval/benchmark_self.json \
    --out evaluation_reports/eval_report_post_codefix_or_base.json \
    --verbose || true
fi

if [[ "$SKIP_DPO" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 6: Generate preference data"
  echo "========================================"

  STUDENT_ARG=()
  if [[ -n "$STUDENT_URL" ]]; then
    STUDENT_ARG=(--student-url "$STUDENT_URL" --student-model "$OLLAMA_BASE_MODEL")
    echo "Using student model at $STUDENT_URL for candidate generation."
  fi

  python data/pref_gen.py \
    --input training_data/sft/synthetic_qa_combined.jsonl \
    --output training_data/preferences/preference_data_auto.jsonl \
    --model "$TEACHER_MODEL" \
    --judge "$TEACHER_MODEL" \
    "${STUDENT_ARG[@]}"

  echo ""
  echo "========================================"
  echo "STEP 7: Generate failure-driven DPO rows from eval reports"
  echo "========================================"

  if [[ -f evaluation_reports/eval_report_base.json ]]; then
    python data/failure_dpo_from_eval.py \
      --report evaluation_reports/eval_report_base.json \
      --out training_data/preferences/preference_data_failures_base.jsonl \
      --threshold 0.75
  fi

  if [[ -f evaluation_reports/eval_report_post_codefix_or_base.json ]]; then
    python data/failure_dpo_from_eval.py \
      --report evaluation_reports/eval_report_post_codefix_or_base.json \
      --out training_data/preferences/preference_data_failures_post.jsonl \
      --threshold 0.75
  fi

  echo ""
  echo "========================================"
  echo "STEP 8: Combine preference data"
  echo "========================================"
  combine_if_exists training_data/preferences/preference_data_combined.jsonl \
    training_data/preferences/preference_data_auto.jsonl \
    training_data/preferences/preference_data_failures_base.jsonl \
    training_data/preferences/preference_data_failures_post.jsonl \
    training_data/preferences/preference_data_failures_unseen_v2.jsonl \
    training_data/preferences/preference_data_rlhf.jsonl

  python scripts/audit_training_data.py training_data/preferences/preference_data_combined.jsonl

  echo ""
  echo "========================================"
  echo "STEP 8b: Validate combined DPO data"
  echo "========================================"
  python scripts/validate_dpo.py training_data/preferences/preference_data_combined.jsonl

  echo ""
  echo "========================================"
  echo "STEP 9: DPO training"
  echo "========================================"

  if [[ ! -d results_sft/adapter ]]; then
    echo "Missing results_sft/adapter. Run SFT before DPO."
    exit 1
  fi

  python model/dpo.py \
    --dpo-data-path training_data/preferences/preference_data_combined.jsonl \
    --out results_dpo \
    --sft-adapter results_sft/adapter \
    --max-steps 200
fi

echo ""
echo "========================================"
echo "DONE"
echo "========================================"

if [[ -f evaluation_reports/eval_report_base.json ]]; then
  python - <<'PY'
import json
from pathlib import Path

for path in ["evaluation_reports/eval_report_base.json", "evaluation_reports/eval_report_post_codefix_or_base.json"]:
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
echo "Do not interpret evaluation_reports/eval_report_post_codefix_or_base.json as fine-tuned-model performance unless '$OLLAMA_BASE_MODEL' points to an exported fine-tuned model."
