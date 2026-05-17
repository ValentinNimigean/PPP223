#!/usr/bin/env bash
set -e
REPO="${1:-.}"
MODEL="${2:-qwen2.5-coder:3b}"
BENCHMARK="${3:-eval/benchmark_self.json}"
OUT="${4:-eval_report.json}"

echo "Evaluating model=$MODEL on repo=$REPO using benchmark=$BENCHMARK"
python eval/eval.py --repo "$REPO" --model "$MODEL" --benchmark "$BENCHMARK" --out "$OUT" --verbose
