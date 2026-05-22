#!/usr/bin/env bash
set -euo pipefail

HTTPX_REPO="${1:-../httpx}"
MODEL="${2:-qwen2.5-coder:3b}"

if [[ ! -d "$HTTPX_REPO" ]]; then
  echo "Missing httpx repo at $HTTPX_REPO"
  echo "Clone it first:"
  echo "git clone https://github.com/encode/httpx.git ../httpx"
  exit 1
fi

python -m eval.eval \
  --repo "$HTTPX_REPO" \
  --model "$MODEL" \
  --benchmark eval/benchmark_httpx.json \
  --out eval_report_httpx_honest.json \
  --disable-deterministic-shortcuts \
  --verbose
