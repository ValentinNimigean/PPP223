"""Rule-based reward shaping for PPO-based RLHF."""

from __future__ import annotations

import json
import re
from typing import Any

from eval.toxicity import ToxicityDetector


FILE_PATTERN = re.compile(r"[\w/\\.-]+\.py")


def _grounding_bonus(response: str, metadata: dict[str, Any]) -> float:
    expected_files = metadata.get("expected_files") or metadata.get("retrieved_files") or []
    if not expected_files:
        return 0.1 if FILE_PATTERN.search(response or "") else 0.0
    matches = sum(1 for item in expected_files if str(item) in (response or ""))
    return 0.2 * min(matches, 2)


def _hallucination_penalty(response: str, metadata: dict[str, Any]) -> float:
    expected_files = [str(item) for item in (metadata.get("expected_files") or [])]
    mentioned_files = FILE_PATTERN.findall(response or "")
    penalty = 0.0
    if expected_files:
        unexpected = [item for item in mentioned_files if item not in expected_files]
        penalty -= 0.15 * min(len(unexpected), 3)

    expected_entities = [str(item) for item in (metadata.get("expected_entities") or [])]
    if expected_entities:
        entity_hits = sum(1 for item in expected_entities if item in (response or ""))
        if entity_hits == 0:
            penalty -= 0.1
    return penalty


def _toxicity_penalty(text: str) -> float:
    scan = ToxicityDetector().scan(text or "")
    if not scan["is_toxic"]:
        return 0.0
    if scan["risk"] == "high":
        return -0.5
    return -0.25


def _malformed_tool_call_penalty(response: str) -> float:
    stripped = (response or "").strip()
    if not stripped:
        return -0.2
    if stripped.startswith("{") and ("tool_calls" in stripped or '"name"' in stripped):
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return -0.2
    return 0.0


def _eos_penalty(response: str) -> float:
    stripped = (response or "").strip()
    if not stripped:
        return -0.1
    if stripped.endswith((".", "!", "?", "`", "}", "]")):
        return 0.0
    return -0.05


def score_rule_based_reward(prompt: str, response: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a scalar rule-based reward and its components."""
    meta = dict(metadata or {})
    components = {
        "grounding_bonus": _grounding_bonus(response, meta),
        "hallucination_penalty": _hallucination_penalty(response, meta),
        "toxicity_penalty": _toxicity_penalty(prompt) + _toxicity_penalty(response),
        "malformed_tool_call_penalty": _malformed_tool_call_penalty(response),
        "eos_penalty": _eos_penalty(response),
    }
    total = sum(components.values())
    return {"total": total, "components": components}
