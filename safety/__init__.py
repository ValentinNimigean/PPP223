"""Safety package exports."""

__all__ = [
    "GuardrailResult",
    "HallucinationDetector",
    "ToxicityDetector",
    "evaluate_guardrails",
]


def __getattr__(name: str):
    if name == "GuardrailResult" or name == "evaluate_guardrails":
        from safety.guardrails import GuardrailResult, evaluate_guardrails

        return {"GuardrailResult": GuardrailResult, "evaluate_guardrails": evaluate_guardrails}[name]
    if name == "HallucinationDetector":
        from safety.hallucination import HallucinationDetector

        return HallucinationDetector
    if name == "ToxicityDetector":
        from safety.toxicity import ToxicityDetector

        return ToxicityDetector
    raise AttributeError(f"module 'safety' has no attribute {name!r}")
