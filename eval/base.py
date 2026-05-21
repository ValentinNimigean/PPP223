"""Shared evaluation interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MetricRecord:
    """Portable evaluation record for a single prompt/response pair."""

    question: str
    response: str
    entity_score: float
    file_score: float
    combined_score: float
    penalized_combined_score: float
    hallucination_penalty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Interface for benchmark evaluators."""

    def evaluate(self) -> list[MetricRecord]:
        """Run the evaluator and return normalized metric records."""

