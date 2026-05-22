#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


CONTAMINATION_TERMS = [
    "AWS SDK",
    "Alibaba Cloud",
    "Django framework",
    "TensorFlow",
    "PyTorch",
    "SageMaker",
    "handle_text_tool_call",
    "QWENClient",
    "client_manager",
    "github.com/aws",
    "github.com/alibabacloud",
    "boto3",
    "botocore",
    "django.db",
    "FastAPI framework",
]

TOXIC_TERMS = [
    "idiot",
    "stupid",
    "moron",
    "kill yourself",
    "go die",
    "i hate you",
    "worthless",
    "shut up",
]


def is_toxic_prompt(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return any(term in lowered for term in TOXIC_TERMS)


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    if not path.exists():
        return rows, [f"Missing file: {path}"]

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue

        if not isinstance(row, dict):
            errors.append(f"line {line_no}: row is not a JSON object")
            continue

        row["_line_no"] = line_no
        rows.append(row)

    return rows, errors


def validate_row(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    line_no = row.get("_line_no", "?")

    for field in ["prompt", "chosen", "rejected"]:
        if field not in row:
            errors.append(f"line {line_no}: missing required field `{field}`")

    prompt = str(row.get("prompt") or "")
    chosen = str(row.get("chosen") or "")
    rejected = str(row.get("rejected") or "")

    if not prompt.strip():
        errors.append(f"line {line_no}: empty prompt")

    if not chosen.strip():
        errors.append(f"line {line_no}: empty chosen")

    if not rejected.strip():
        errors.append(f"line {line_no}: empty rejected")

    if chosen.strip() and rejected.strip() and chosen.strip() == rejected.strip():
        errors.append(f"line {line_no}: chosen equals rejected")

    chosen_lower = chosen.lower()
    rejected_lower = rejected.lower()

    for term in CONTAMINATION_TERMS:
        if term.lower() in chosen_lower:
            errors.append(f"line {line_no}: chosen contains contamination term `{term}`")
        if term.lower() in rejected_lower:
            warnings.append(f"line {line_no}: rejected contains contamination term `{term}`")

    if "the answer is" in chosen_lower:
        errors.append(f"line {line_no}: chosen uses vague phrase `The answer is`")

    if "generate_preferences" in chosen:
        errors.append(f"line {line_no}: chosen mentions nonexistent `generate_preferences`")

    if not is_toxic_prompt(prompt):
        if "abusive" in chosen_lower or "threatening" in chosen_lower:
            errors.append(f"line {line_no}: non-toxic prompt got toxicity/refusal chosen answer")

    if "FastAPIRetriever" in prompt and ("abusive" in chosen_lower or "threatening" in chosen_lower):
        errors.append(f"line {line_no}: FastAPIRetriever no-match got toxicity refusal")

    meta = row.get("meta") or {}
    expected_files = meta.get("expected_files") or []
    if isinstance(expected_files, list):
        for file_path in expected_files:
            if file_path and str(file_path) not in chosen:
                errors.append(f"line {line_no}: chosen missing expected file `{file_path}`")

    return errors, warnings


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("training_data/preferences/preference_data_combined.jsonl")

    rows, load_errors = load_jsonl(path)
    errors: List[str] = list(load_errors)
    warnings: List[str] = []

    for row in rows:
        row_errors, row_warnings = validate_row(row)
        errors.extend(row_errors)
        warnings.extend(row_warnings)

    print(f"Validated: {path}")
    print(f"rows_checked: {len(rows)}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("DPO dataset looks clean.")


if __name__ == "__main__":
    main()
