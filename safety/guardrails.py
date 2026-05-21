from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from safety.hallucination import HallucinationDetector
from safety.toxicity import ToxicityDetector


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    toxicity_score: float
    hallucination_score: float
    unsupported_entities: List[str] = field(default_factory=list)
    unsupported_files: List[str] = field(default_factory=list)
    toxicity_risk: str = "low"
    hallucination_risk: str = "low"
    verified_entities: List[str] = field(default_factory=list)
    verified_files: List[str] = field(default_factory=list)


def evaluate_guardrails(
    text: str,
    *,
    toxicity_detector: ToxicityDetector,
    hallucination_detector: Optional[HallucinationDetector] = None,
    allowed_entities: Optional[Iterable[str]] = None,
    allowed_files: Optional[Iterable[str]] = None,
) -> GuardrailResult:
    toxicity_scan = toxicity_detector.scan(text or "")
    hallucination_scan = (
        hallucination_detector.scan(
            text or "",
            allowed_entities=allowed_entities,
            allowed_files=allowed_files,
        )
        if hallucination_detector is not None
        else {
            "hallucination_score": 0.0,
            "hallucination_risk": "low",
            "unsupported_entities": [],
            "unsupported_files": [],
            "verified_entities": [],
            "verified_files": [],
        }
    )

    allowed = True
    reason = "ok"
    if toxicity_scan["risk"] in {"medium", "high"}:
        allowed = False
        reason = "toxic_output"
    elif hallucination_scan["hallucination_risk"] == "high":
        allowed = False
        reason = "unsupported_output"

    return GuardrailResult(
        allowed=allowed,
        reason=reason,
        toxicity_score=float(toxicity_scan.get("toxicity_score", 0.0)),
        hallucination_score=float(hallucination_scan.get("hallucination_score", 0.0)),
        unsupported_entities=list(hallucination_scan.get("unsupported_entities", [])),
        unsupported_files=list(hallucination_scan.get("unsupported_files", [])),
        toxicity_risk=str(toxicity_scan.get("risk", "low")),
        hallucination_risk=str(hallucination_scan.get("hallucination_risk", "low")),
        verified_entities=list(hallucination_scan.get("verified_entities", [])),
        verified_files=list(hallucination_scan.get("verified_files", [])),
    )
