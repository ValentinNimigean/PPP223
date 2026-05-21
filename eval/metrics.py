"""Reusable evaluation metrics."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def entity_score(response: str, expected_entities: list[str]) -> float:
    response_lower = (response or "").lower()
    if not expected_entities:
        return 1.0
    found = [entity for entity in expected_entities if str(entity).lower() in response_lower]
    return len(found) / len(expected_entities)


def file_score(response: str, expected_files: list[str]) -> float:
    response_lower = (response or "").lower()
    if not expected_files:
        return 1.0
    found = [path for path in expected_files if str(path).lower() in response_lower]
    return len(found) / len(expected_files)


def combined_score(entity: float, file_: float) -> float:
    return (float(entity) + float(file_)) / 2.0


def hallucination_penalty(response: str, expected_files: list[str]) -> tuple[float, list[str]]:
    mentioned_files = re.findall(r"[\w/\\.-]+\.py", (response or "").lower())
    unexpected = [
        path
        for path in mentioned_files
        if expected_files and not any(str(expected).lower() in path for expected in expected_files)
    ]
    penalty = min(len(unexpected) * 0.1, 0.3)
    return penalty, unexpected


def penalized_combined_score(response: str, expected_entities: list[str], expected_files: list[str]) -> tuple[float, dict[str, Any]]:
    e_score = entity_score(response, expected_entities)
    f_score = file_score(response, expected_files)
    combined = combined_score(e_score, f_score)
    penalty, unexpected = hallucination_penalty(response, expected_files)
    return max(0.0, combined - penalty), {
        "entity_score": e_score,
        "file_score": f_score,
        "combined_score": combined,
        "hallucination_penalty": penalty,
        "unexpected_files_mentioned": unexpected,
        "matched_entities": [e for e in expected_entities if str(e).lower() in (response or "").lower()],
        "matched_files": [f for f in expected_files if str(f).lower() in (response or "").lower()],
        "missed_entities": [e for e in expected_entities if str(e).lower() not in (response or "").lower()],
        "missed_files": [f for f in expected_files if str(f).lower() not in (response or "").lower()],
    }


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if (prediction or "").strip() == (reference or "").strip() else 0.0


def _normalize_tokens(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return [token for token in normalized.split() if token]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_tokens(prediction)
    ref_tokens = _normalize_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def optional_rouge(prediction: str, reference: str) -> dict[str, float] | None:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        return None

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference or "", prediction or "")
    return {
        "rouge1_fmeasure": float(scores["rouge1"].fmeasure),
        "rougeL_fmeasure": float(scores["rougeL"].fmeasure),
    }


def optional_bertscore(prediction: str, reference: str) -> dict[str, float] | None:
    try:
        from bert_score import score as bert_score  # type: ignore
    except ImportError:
        return None

    precision, recall, f1 = bert_score([prediction or ""], [reference or ""], lang="en", verbose=False)
    return {
        "bertscore_precision": float(precision.mean().item()),
        "bertscore_recall": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }
