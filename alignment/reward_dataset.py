"""Prepare reward-model and PPO prompt datasets from preference data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alignment.feedback_schema import load_preferences


def format_reward_input(prompt: str, response: str) -> str:
    """Create a reward-model input string from a prompt/response pair."""
    return f"Prompt:\n{prompt}\n\nResponse:\n{response}"


def build_reward_examples(preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create positive and negative reward-model examples from preferences."""
    examples: list[dict[str, Any]] = []
    for row in preferences:
        prompt = row["prompt"]
        metadata = dict(row.get("metadata") or {})
        common = {
            "prompt": prompt,
            "source": row.get("source"),
            "metadata": metadata,
            "safety_labels": dict(row.get("safety_labels") or {}),
        }
        examples.append(
            {
                **common,
                "response": row["chosen"],
                "label": 1,
                "pairwise_target": "chosen",
                "text": format_reward_input(prompt, row["chosen"]),
            }
        )
        examples.append(
            {
                **common,
                "response": row["rejected"],
                "label": 0,
                "pairwise_target": "rejected",
                "text": format_reward_input(prompt, row["rejected"]),
            }
        )
    return examples


def build_pairwise_reward_examples(preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create pairwise ranking rows for chosen/rejected preference pairs."""
    rows: list[dict[str, Any]] = []
    for row in preferences:
        rows.append(
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "source": row.get("source"),
                "metadata": dict(row.get("metadata") or {}),
                "safety_labels": dict(row.get("safety_labels") or {}),
            }
        )
    return rows


def _extract_prompt_from_jsonl_row(row: dict[str, Any]) -> str | None:
    if row.get("prompt"):
        return str(row["prompt"]).strip()
    if row.get("question"):
        return str(row["question"]).strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
    return None


def _load_optional_jsonl_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    if not path.exists():
        return prompts
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            prompt = _extract_prompt_from_jsonl_row(row)
            if prompt:
                prompts.append(
                    {
                        "prompt": prompt,
                        "metadata": {
                            "source_file": str(path),
                            "line_number": line_number,
                            "origin": "synthetic_dataset",
                        },
                    }
                )
    return prompts


def _load_benchmark_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for index, row in enumerate(rows, start=1):
        prompt = str(row.get("question") or "").strip()
        if not prompt:
            continue
        prompts.append(
            {
                "prompt": prompt,
                "metadata": {
                    "source_file": str(path),
                    "row_number": index,
                    "origin": "benchmark",
                },
            }
        )
    return prompts


def build_prompt_dataset(
    preference_path: str,
    synthetic_paths: list[str] | None = None,
    benchmark_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Create a deduplicated PPO prompt dataset."""
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()

    preferences = load_preferences(preference_path, allow_pending=True)
    for row in preferences:
        prompt = row["prompt"].strip()
        if prompt and prompt not in seen:
            prompts.append(
                {
                    "prompt": prompt,
                    "metadata": {
                        "origin": "preferences",
                        "source": row.get("source"),
                        **dict(row.get("metadata") or {}),
                    },
                }
            )
            seen.add(prompt)

    synthetic_candidates = synthetic_paths or [
        "data/final/sft_train.jsonl",
    ]
    for item in synthetic_candidates:
        for row in _load_optional_jsonl_prompts(Path(item)):
            prompt = row["prompt"]
            if prompt not in seen:
                prompts.append(row)
                seen.add(prompt)

    benchmark_candidates = benchmark_paths or []
    for item in benchmark_candidates:
        for row in _load_benchmark_prompts(Path(item)):
            prompt = row["prompt"]
            if prompt not in seen:
                prompts.append(row)
                seen.add(prompt)

    return prompts


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PPO prompt datasets from preference and benchmark data.")
    parser.add_argument("--preferences", required=True, help="Canonical preference JSONL path.")
    parser.add_argument("--out", required=True, help="Output JSONL path for PPO prompts.")
    parser.add_argument("--synthetic", nargs="*", default=None, help="Optional synthetic JSONL paths.")
    parser.add_argument("--benchmarks", nargs="*", default=None, help="Optional benchmark JSON paths.")
    args = parser.parse_args()

    prompts = build_prompt_dataset(
        preference_path=args.preferences,
        synthetic_paths=args.synthetic,
        benchmark_paths=args.benchmarks,
    )
    _write_jsonl(prompts, Path(args.out))
    print(f"Wrote {len(prompts)} PPO prompts to {args.out}")


if __name__ == "__main__":
    main()
