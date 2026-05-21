"""Reusable RAG evaluation helpers."""

from __future__ import annotations

from typing import Any

from eval.hallucination import HallucinationDetector
from ingest.metadata import CodeChunk


def retrieval_precision_at_k(results: list[dict[str, Any]], expected_files: list[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = 0
    for row in top_k:
        filepath = str((row.get("metadata") or {}).get("filepath") or "")
        if any(str(expected) == filepath for expected in expected_files):
            hits += 1
    return hits / min(k, len(top_k))


def source_hit_rate(results: list[dict[str, Any]], expected_files: list[str]) -> float:
    if not expected_files:
        return 1.0
    returned = {str((row.get("metadata") or {}).get("filepath") or "") for row in results}
    hits = sum(1 for expected in expected_files if str(expected) in returned)
    return hits / len(expected_files)


def answer_groundedness(response: str, chunks: list[CodeChunk]) -> dict[str, Any]:
    detector = HallucinationDetector(chunks)
    scan = detector.scan(response)
    verified = len(scan["verified_entities"]) + len(scan["verified_files"])
    unverified = len(scan["unverified_entities"]) + len(scan["unverified_files"])
    total = verified + unverified
    groundedness = verified / total if total else 1.0
    return {
        "groundedness_score": groundedness,
        "hallucination_scan": scan,
    }
