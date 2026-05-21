"""Evaluation report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(report: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_markdown_summary(report: dict[str, Any]) -> str:
    overall = report.get("overall", {})
    lines = [
        "# Evaluation Summary",
        "",
        f"- Model: `{report.get('model')}`",
        f"- Backend: `{report.get('backend', 'ollama')}`",
        f"- Mode: `{report.get('mode', 'self')}`",
        f"- Questions evaluated: `{overall.get('questions_evaluated', 0)}`",
        f"- Combined score: `{overall.get('combined_score')}`",
        f"- Penalized combined score: `{overall.get('penalized_combined_score')}`",
    ]
    if overall.get("average_reward_model_score") is not None:
        lines.append(f"- Average reward model score: `{overall.get('average_reward_model_score')}`")

    lines.extend(["", "## Per Question", ""])
    for index, row in enumerate(report.get("results", []), start=1):
        lines.extend(
            [
                f"### Q{index}",
                f"- Question: {row.get('question')}",
                f"- Combined: `{row.get('combined_score')}`",
                f"- Penalized: `{row.get('penalized_combined_score')}`",
                f"- Hallucination penalty: `{row.get('hallucination_penalty')}`",
                f"- Toxicity risk: `{(row.get('toxicity_flags') or {}).get('risk', 'low')}`",
                f"- Sources: `{', '.join((row.get('sources') or []))}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_markdown_summary(report: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown_summary(report), encoding="utf-8")
