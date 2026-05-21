#!/usr/bin/env bash
set -euo pipefail

MODE="full"
REPO="."
SEED="3407"
PYTHON_BIN="python"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE="quick"; shift ;;
    --full) MODE="full"; shift ;;
    --repo) REPO="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "Prepare training data"
echo "Mode:       $MODE"
echo "Repo:       $REPO"
echo "Seed:       $SEED"
echo "Python:     $PYTHON_BIN"
echo "========================================"

PROFILE_ARGS=(--profile "$MODE")
if [[ "$MODE" == "quick" ]]; then
  echo "Quick mode targets smaller smoke-test datasets."
fi

if [[ ! -d "$REPO" ]]; then
  echo "Repository path does not exist: $REPO"
  exit 1
fi

echo ""
echo "0. Dependency preflight"
"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = {
    "grimp": "grimp",
    "qdrant_client": "qdrant-client",
    "fastembed": "fastembed",
    "tree_sitter": "tree-sitter",
    "tree_sitter_python": "tree-sitter-python",
    "pydantic": "pydantic",
    "openai": "openai",
}
missing = [package_name for module_name, package_name in required.items() if importlib.util.find_spec(module_name) is None]
if missing:
    print("Missing required packages for dataset preparation: " + ", ".join(missing))
    print("pip install -r requirements.txt")
    sys.exit(1)
PY

echo ""
echo "1. Initial audit of current datasets"
"$PYTHON_BIN" scripts/audit_datasets.py --allow-fail

echo ""
echo "2. Clean raw datasets"
"$PYTHON_BIN" scripts/clean_datasets.py

echo ""
echo "3. Build curated SFT dataset"
"$PYTHON_BIN" data/build_sft_dataset.py \
  --repo "$REPO" \
  --out data/curated/sft_repo_qa.jsonl \
  "${PROFILE_ARGS[@]}"

echo ""
echo "4. Build curated preference dataset"
"$PYTHON_BIN" data/build_preference_dataset.py \
  --repo "$REPO" \
  --sft data/curated/sft_repo_qa.jsonl \
  --out data/curated/preferences_rlhf.jsonl \
  "${PROFILE_ARGS[@]}"

echo ""
echo "5. Split curated datasets into final train/val/holdout splits"
"$PYTHON_BIN" scripts/split_datasets.py \
  --sft data/curated/sft_repo_qa.jsonl \
  --preferences data/curated/preferences_rlhf.jsonl \
  --seed "$SEED"

echo ""
echo "6. Final audit of training splits"
"$PYTHON_BIN" scripts/audit_datasets.py \
  --paths data/final/sft_train.jsonl data/final/preferences_train.jsonl

echo ""
echo "Prepared datasets:"
echo "- data/clean/"
echo "- data/curated/"
echo "- data/final/"
echo "- artifacts/dataset_audit.md"
echo "- artifacts/dataset_cleaning_report.md"
echo "- artifacts/dataset_split_report.md"
