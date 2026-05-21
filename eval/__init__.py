"""Evaluation package exports."""

from eval.base import EvaluatorProtocol, MetricRecord
from eval.metrics import exact_match, file_score, token_f1
from eval.report import build_markdown_summary, write_json_report, write_markdown_summary

__all__ = [
    "EvaluatorProtocol",
    "HallucinationDetector",
    "MetricRecord",
    "ToxicityDetector",
    "build_markdown_summary",
    "exact_match",
    "file_score",
    "token_f1",
    "write_json_report",
    "write_markdown_summary",
]


def __getattr__(name: str):
    if name == "HallucinationDetector":
        from eval.hallucination import HallucinationDetector

        return HallucinationDetector
    if name == "ToxicityDetector":
        from eval.toxicity import ToxicityDetector

        return ToxicityDetector
    raise AttributeError(f"module 'eval' has no attribute {name!r}")
