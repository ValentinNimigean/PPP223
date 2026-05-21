"""Dataset loading, validation, and formatting helpers for SFT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from train.tasks import format_record_as_messages, validate_task_record


def compute_dataset_hash(dataset_path: str) -> str:
    """Compute a stable SHA256 for the raw dataset file."""
    return hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()


def validate_jsonl_dataset(task: str, dataset_path: str) -> list[dict[str, Any]]:
    """Load and validate a task dataset from JSONL."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Row {row_number}: invalid JSON: {exc.msg}") from exc
            validate_task_record(task, record, row_number)
            records.append(record)

    if not records:
        raise ValueError(f"Dataset `{dataset_path}` is empty after filtering blank lines.")

    return records


def load_and_prepare_records(task: str, dataset_path: str) -> tuple[list[dict[str, Any]], str]:
    """Validate a JSONL dataset and convert rows to chat-message records."""
    records = validate_jsonl_dataset(task, dataset_path)
    dataset_hash = compute_dataset_hash(dataset_path)
    prepared = []
    for record in records:
        normalized = dict(record)
        normalized["messages"] = format_record_as_messages(task, record)
        prepared.append(normalized)
    return prepared, dataset_hash
