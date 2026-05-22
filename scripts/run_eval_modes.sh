#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-.}"
MODEL="${2:-qwen2.5-coder:3b}"
BENCHMARK="${3:-eval/benchmark_self.json}"

echo "Running demo-mode eval..."
python -m eval.eval \
  --repo "$REPO" \
  --model "$MODEL" \
  --benchmark "$BENCHMARK" \
  --out eval_report_demo_mode.json \
  --verbose

echo "Running honest-mode eval..."
python -m eval.eval \
  --repo "$REPO" \
  --model "$MODEL" \
  --benchmark "$BENCHMARK" \
  --out eval_report_honest_mode.json \
  --disable-deterministic-shortcuts \
  --verbose

python - <<'PY'
import json
from pathlib import Path

for path in ["eval_report_demo_mode.json", "eval_report_honest_mode.json"]:
    data = json.loads(Path(path).read_text())
    overall = data["overall"]
    print()
    print(path)
    print("combined:", overall["combined_score"])
    print("penalized:", overall["penalized_combined_score"])
    print("questions:", overall["questions_evaluated"])
    print("deterministic_shortcuts_enabled:", data.get("deterministic_shortcuts_enabled"))
PY
