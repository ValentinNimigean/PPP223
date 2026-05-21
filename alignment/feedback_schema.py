"""Canonical preference schema for PPO-based RLHF and DPO baselines."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from eval.toxicity import ToxicityDetector


PENDING_CORRECTION_TOKEN = "__NEEDS_CORRECTION__"
ALLOWED_SOURCES = {"human", "synthetic", "eval_failure"}
CONTAMINATED_PATH_PATTERNS = [
    r"/home/",
    r"file://",
    r"[A-Za-z]:\\",
    r"Documents/GitHub",
]


def _contains_contaminated_path(text: str) -> bool:
    value = text or ""
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in CONTAMINATED_PATH_PATTERNS)


def _coerce_source(source: Any) -> str:
    normalized = str(source or "synthetic").strip().lower()
    if normalized not in ALLOWED_SOURCES:
        return "synthetic"
    return normalized


def validate_preference_row(row: dict[str, Any], allow_pending: bool = False) -> dict[str, Any]:
    """Validate and normalize a canonical preference row."""
    if not isinstance(row, dict):
        raise ValueError(f"Preference row must be a JSON object, found {type(row).__name__}.")

    prompt = str(row.get("prompt") or "").strip()
    chosen = str(row.get("chosen") or "").strip()
    rejected = str(row.get("rejected") or "").strip()
    source = _coerce_source(row.get("source"))

    if not prompt:
        raise ValueError("prompt must be non-empty.")
    if not chosen:
        raise ValueError("chosen must be non-empty.")
    if not rejected:
        raise ValueError("rejected must be non-empty.")
    if chosen == rejected:
        raise ValueError("chosen and rejected must differ.")

    metadata = dict(row.get("metadata") or {})
    pending = bool(row.get("pending") or metadata.get("pending"))
    if chosen == PENDING_CORRECTION_TOKEN and not pending:
        raise ValueError("rows with __NEEDS_CORRECTION__ are rejected unless explicitly marked pending.")

    combined_fields = [prompt, chosen, rejected]
    if any(_contains_contaminated_path(value) for value in combined_fields):
        raise ValueError("Preference row contains contaminated local paths such as /home/ or file://.")

    safety_labels = dict(row.get("safety_labels") or {})
    toxic_scan = ToxicityDetector().scan(prompt)
    if toxic_scan["is_toxic"]:
        safety_labels.setdefault("toxic_prompt", True)
        safety_labels.setdefault("toxicity_terms", toxic_scan["matched_terms"])
    else:
        safety_labels.setdefault("toxic_prompt", False)

    normalized = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
        "rating": row.get("rating"),
        "metadata": metadata,
        "safety_labels": safety_labels,
    }
    if pending:
        normalized["metadata"]["pending"] = True
    return normalized


def load_preferences(path: str, allow_pending: bool = False) -> list[dict[str, Any]]:
    """Load and validate canonical preference rows from JSONL."""
    pref_path = Path(path)
    if not pref_path.exists():
        raise FileNotFoundError(f"Preference file not found: {path}")

    rows: list[dict[str, Any]] = []
    with pref_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            try:
                normalized = validate_preference_row(parsed, allow_pending=allow_pending)
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            if normalized["metadata"].get("pending") and not allow_pending:
                raise ValueError(
                    f"line {line_number}: pending rows require allow_pending=True."
                )
            rows.append(normalized)

    return rows


def convert_old_feedback_to_preferences(input_path: str, output_path: str) -> dict[str, int]:
    """Convert legacy human feedback JSONL into canonical preference rows."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    converted = 0
    skipped = 0
    pending = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as source, out_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)

            metadata = {
                "timestamp": row.get("timestamp"),
                "hallucination_risk": row.get("hallucination_risk"),
                "correction_provided": row.get("correction_provided"),
            }
            converted_row = {
                "prompt": row.get("prompt", ""),
                "chosen": row.get("chosen", ""),
                "rejected": row.get("rejected", ""),
                "source": row.get("source", "human"),
                "rating": row.get("rating"),
                "metadata": metadata,
                "safety_labels": {},
            }

            if converted_row["chosen"] == PENDING_CORRECTION_TOKEN or not row.get("correction_provided", True):
                converted_row["metadata"]["pending"] = True
                pending += 1

            try:
                normalized = validate_preference_row(converted_row, allow_pending=True)
            except ValueError:
                skipped += 1
                continue

            target.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            converted += 1

    return {"converted": converted, "skipped": skipped, "pending": pending}
