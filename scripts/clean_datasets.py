#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_datasets import load_jsonl


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SFT_FILES = [
    REPO_ROOT / "synthetic_qa_seed.jsonl",
    REPO_ROOT / "synthetic_qa_combined.jsonl",
    REPO_ROOT / "synthetic_qa_combined_2048.jsonl",
]
DEFAULT_PREFERENCE_FILES = [REPO_ROOT / "preference_data_combined.jsonl"]

TOY_PROMPT_PATTERNS = [
    r"\bi could kiss you\b",
    r"^pula$",
    r"\blol\b",
    r"\bhehe\b",
    r"\bjoke\b",
]
SAFETY_TERMS = [
    "toxic",
    "abusive",
    "harassment",
    "sexual",
    "self-harm",
    "violence",
    "hate",
    "swear",
]


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_multiline_text(text: Any) -> str:
    lines = [normalize_text(line) for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def base_row_hash(payload: Any) -> str:
    return stable_hash(payload)[:16]


def canonicalize_tool_calls(tool_calls: Any) -> Any:
    if not isinstance(tool_calls, list):
        return tool_calls
    cleaned = []
    for item in tool_calls:
        if isinstance(item, dict):
            cleaned.append(json.loads(json.dumps(item, sort_keys=True, ensure_ascii=False)))
        else:
            cleaned.append(item)
    return cleaned


def normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {"role": message.get("role")}
    if "content" in message and message.get("content") is not None:
        cleaned["content"] = normalize_multiline_text(message.get("content"))
    elif "content" in message:
        cleaned["content"] = None
    if "tool_calls" in message:
        cleaned["tool_calls"] = canonicalize_tool_calls(message.get("tool_calls"))
    if "tool_call_id" in message:
        cleaned["tool_call_id"] = message.get("tool_call_id")
    if "name" in message:
        cleaned["name"] = message.get("name")
    return cleaned


def normalize_prompt_key(text: str) -> str:
    return normalize_text(text).lower()


def extract_first_user(messages: Sequence[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def extract_final_assistant(messages: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message
    return None


def valid_sft_role_order(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    if not all(isinstance(message, dict) for message in messages):
        return False

    index = 0
    if messages[0].get("role") == "system":
        index = 1
    if index >= len(messages) or messages[index].get("role") != "user":
        return False
    index += 1
    if index >= len(messages):
        return False

    seen_assistant = False
    for position in range(index, len(messages)):
        role = messages[position].get("role")
        if role == "assistant":
            seen_assistant = True
            continue
        if role == "tool":
            if not seen_assistant:
                return False
            continue
        return False
    return messages[-1].get("role") == "assistant"


def answer_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()


def are_answers_meaningfully_different(a: str, b: str) -> bool:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return False
    if left.lower() == right.lower():
        return False
    return answer_similarity(left, right) < 0.90


def build_sft_metadata(source_file: str, source_line: int, cleaned_at: str, row_hash: str) -> Dict[str, Any]:
    return {
        "source_file": source_file,
        "source_line": source_line,
        "cleaned_at": cleaned_at,
        "hash": row_hash,
        "task_type": "sft_code_qa",
    }


def clean_sft_row(row: Dict[str, Any], source_file: str, source_line: int, cleaned_at: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, "invalid_messages"

    normalized_messages = [normalize_message(message) for message in messages if isinstance(message, dict)]
    if len(normalized_messages) != len(messages):
        return None, "invalid_messages"
    if not valid_sft_role_order(normalized_messages):
        return None, "invalid_role_order"

    assistant_messages = [message for message in normalized_messages if message.get("role") == "assistant"]
    for message in assistant_messages:
        if message.get("content") is None:
            return None, "assistant_content_null"
        if not normalize_text(message.get("content")):
            return None, "assistant_content_empty"

    final_assistant = extract_final_assistant(normalized_messages)
    if final_assistant is None:
        return None, "missing_final_assistant"

    final_answer = normalize_text(final_assistant.get("content"))
    if not final_answer:
        return None, "assistant_content_empty"

    normalized_messages[-1]["content"] = final_answer

    cleaned = {"messages": normalized_messages}
    row_hash = base_row_hash(cleaned)
    cleaned["metadata"] = build_sft_metadata(source_file, source_line, cleaned_at, row_hash)
    return cleaned, None


def row_has_safety_label(row: Dict[str, Any]) -> bool:
    labels = row.get("safety_labels")
    if isinstance(labels, list) and labels:
        return True
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        meta_labels = metadata.get("safety_labels")
        if isinstance(meta_labels, list) and meta_labels:
            return True
    meta = row.get("meta")
    if isinstance(meta, dict):
        meta_labels = meta.get("safety_labels")
        if isinstance(meta_labels, list) and meta_labels:
            return True
    text = json.dumps(row, ensure_ascii=False).lower()
    return any(term in text for term in SAFETY_TERMS)


def looks_like_toy_preference(prompt: str, chosen: str, rejected: str, safety_example: bool) -> bool:
    if safety_example:
        return False
    prompt_key = normalize_prompt_key(prompt)
    joined = " ".join([prompt_key, normalize_text(chosen).lower(), normalize_text(rejected).lower()])
    if any(re.search(pattern, prompt_key) for pattern in TOY_PROMPT_PATTERNS):
        return True
    if len(prompt_key) < 4 and "repository" not in joined and "repo" not in joined:
        return True
    if "kiss you" in joined:
        return True
    return False


def normalize_safety_labels(row: Dict[str, Any], prompt: str, chosen: str, rejected: str) -> List[str]:
    labels: List[str] = []
    for key in ("safety_labels",):
        value = row.get(key)
        if isinstance(value, list):
            labels.extend(str(item) for item in value if item)
    for container_key in ("metadata", "meta"):
        container = row.get(container_key)
        if isinstance(container, dict):
            value = container.get("safety_labels")
            if isinstance(value, list):
                labels.extend(str(item) for item in value if item)
    text = " ".join([prompt, chosen, rejected]).lower()
    if any(term in text for term in ("toxic", "abusive", "worthless", "stupid", "swear")) and "safety" not in labels:
        labels.append("safety")
    deduped = []
    seen = set()
    for label in labels:
        norm = normalize_prompt_key(label)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def clean_preference_row(row: Dict[str, Any], source_file: str, source_line: int, cleaned_at: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    prompt = normalize_multiline_text(row.get("prompt"))
    chosen = normalize_multiline_text(row.get("chosen"))
    rejected = normalize_multiline_text(row.get("rejected"))

    if not prompt:
        return None, "empty_prompt"
    if not chosen:
        return None, "empty_chosen"
    if not rejected:
        return None, "empty_rejected"
    if normalize_prompt_key(chosen) == normalize_prompt_key(rejected):
        return None, "chosen_equals_rejected"

    safety_example = row_has_safety_label(row)
    if looks_like_toy_preference(prompt, chosen, rejected, safety_example):
        return None, "toy_or_joke_row"

    safety_labels = normalize_safety_labels(row, prompt, chosen, rejected)
    metadata = {
        "source_file": source_file,
        "source_line": source_line,
        "cleaned_at": cleaned_at,
        "hash": base_row_hash({"prompt": prompt, "chosen": chosen, "rejected": rejected}),
        "original_metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else row.get("meta"),
        "task_type": "preference_pair",
    }
    cleaned = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source_file,
        "rating": row.get("rating"),
        "metadata": metadata,
        "safety_labels": safety_labels,
    }
    return cleaned, None


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_removed(counter: Counter[str]) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def clean_sft_datasets(paths: Sequence[Path], cleaned_at: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    before_count = 0
    removed = Counter()
    candidates: List[Dict[str, Any]] = []

    for path in paths:
        for loaded in load_jsonl(path):
            before_count += 1
            if loaded.parse_error:
                removed["invalid_json"] += 1
                continue
            assert loaded.data is not None
            cleaned_row, reason = clean_sft_row(loaded.data, path.name, loaded.line_no, cleaned_at)
            if cleaned_row is None:
                removed[str(reason)] += 1
                continue
            candidates.append(cleaned_row)

    deduped_rows: List[Dict[str, Any]] = []
    seen_exact = set()
    for row in candidates:
        row_key = stable_hash({"messages": row["messages"]})
        if row_key in seen_exact:
            removed["exact_duplicate_row"] += 1
            continue
        seen_exact.add(row_key)
        deduped_rows.append(row)

    grouped: defaultdict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in deduped_rows:
        grouped[normalize_prompt_key(extract_first_user(row["messages"]))].append(row)

    final_rows: List[Dict[str, Any]] = []
    for rows in grouped.values():
        kept_answers: List[str] = []
        for row in rows:
            final_answer = extract_final_assistant(row["messages"])
            assert final_answer is not None
            answer_text = str(final_answer.get("content") or "")
            if any(not are_answers_meaningfully_different(answer_text, kept) for kept in kept_answers):
                removed["duplicate_user_question_similar_answer"] += 1
                continue
            kept_answers.append(answer_text)
            final_rows.append(row)

    for row in final_rows:
        row["metadata"]["hash"] = base_row_hash({"messages": row["messages"]})

    report = {
        "before": before_count,
        "after": len(final_rows),
        "removed": before_count - len(final_rows),
        "removed_by_reason": summarize_removed(removed),
    }
    return final_rows, report


def clean_preference_datasets(paths: Sequence[Path], cleaned_at: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    before_count = 0
    removed = Counter()
    final_rows: List[Dict[str, Any]] = []
    seen_pairs = set()

    for path in paths:
        for loaded in load_jsonl(path):
            before_count += 1
            if loaded.parse_error:
                removed["invalid_json"] += 1
                continue
            assert loaded.data is not None
            cleaned_row, reason = clean_preference_row(loaded.data, path.name, loaded.line_no, cleaned_at)
            if cleaned_row is None:
                removed[str(reason)] += 1
                continue
            pair_key = stable_hash(
                {
                    "prompt": cleaned_row["prompt"],
                    "chosen": cleaned_row["chosen"],
                    "rejected": cleaned_row["rejected"],
                }
            )
            if pair_key in seen_pairs:
                removed["exact_duplicate_row"] += 1
                continue
            seen_pairs.add(pair_key)
            final_rows.append(cleaned_row)

    report = {
        "before": before_count,
        "after": len(final_rows),
        "removed": before_count - len(final_rows),
        "removed_by_reason": summarize_removed(removed),
    }
    return final_rows, report


def build_cleaning_report(sft_report: Dict[str, Any], preference_report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "sft": sft_report,
            "preferences": preference_report,
        },
        "summary": {
            "sft_before": sft_report["before"],
            "sft_after": sft_report["after"],
            "preferences_before": preference_report["before"],
            "preferences_after": preference_report["after"],
        },
        "notes": [
            "The cleaned datasets are intermediate artifacts only.",
            "Rows are removed when they fail structural checks, duplicate filters, or obvious toy-data heuristics.",
            "This cleaning step does not claim the remaining data is sufficient for final training.",
        ],
    }


def write_cleaning_report(report: Dict[str, Any], artifacts_dir: Path) -> Tuple[Path, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifacts_dir / "dataset_cleaning_report.json"
    md_path = artifacts_dir / "dataset_cleaning_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Dataset Cleaning Report",
        "",
        "This report describes the intermediate cleaning pass. It does not claim the outputs are final training-ready datasets.",
        "",
        "## SFT",
        f"- rows before: {report['datasets']['sft']['before']}",
        f"- rows after: {report['datasets']['sft']['after']}",
        f"- rows removed: {report['datasets']['sft']['removed']}",
        "- rows removed by reason:",
    ]
    for reason, count in report["datasets"]["sft"]["removed_by_reason"].items():
        lines.append(f"  - {reason}: {count}")

    lines.extend(
        [
            "",
            "## Preferences",
            f"- rows before: {report['datasets']['preferences']['before']}",
            f"- rows after: {report['datasets']['preferences']['after']}",
            f"- rows removed: {report['datasets']['preferences']['removed']}",
            "- rows removed by reason:",
        ]
    )
    for reason, count in report["datasets"]["preferences"]["removed_by_reason"].items():
        lines.append(f"  - {reason}: {count}")

    lines.extend(["", "## Notes"])
    for note in report["notes"]:
        lines.append(f"- {note}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_cleaning(
    sft_files: Optional[Sequence[Path]] = None,
    preference_files: Optional[Sequence[Path]] = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    cleaned_at = datetime.now(timezone.utc).isoformat()
    sft_rows, sft_report = clean_sft_datasets(list(sft_files or DEFAULT_SFT_FILES), cleaned_at)
    preference_rows, preference_report = clean_preference_datasets(
        list(preference_files or DEFAULT_PREFERENCE_FILES), cleaned_at
    )

    clean_data_dir = repo_root / "data" / "clean"
    artifacts_dir = repo_root / "artifacts"
    clean_data_dir.mkdir(parents=True, exist_ok=True)
    sft_output = clean_data_dir / "sft_clean.jsonl"
    pref_output = clean_data_dir / "preferences_clean.jsonl"
    write_jsonl(sft_output, sft_rows)
    write_jsonl(pref_output, preference_rows)

    report = build_cleaning_report(sft_report, preference_report)
    report["outputs"] = {
        "sft_clean": sft_output.as_posix(),
        "preferences_clean": pref_output.as_posix(),
    }
    json_report, md_report = write_cleaning_report(report, artifacts_dir)
    report["artifacts"] = {
        "json": json_report.as_posix(),
        "markdown": md_report.as_posix(),
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    report = run_cleaning()
    sft = report["datasets"]["sft"]
    preferences = report["datasets"]["preferences"]

    print(f"SFT rows before: {sft['before']}")
    print(f"SFT rows after: {sft['after']}")
    print(f"SFT rows removed by reason: {json.dumps(sft['removed_by_reason'], ensure_ascii=False)}")
    print(f"Preference rows before: {preferences['before']}")
    print(f"Preference rows after: {preferences['after']}")
    print(f"Preference rows removed by reason: {json.dumps(preferences['removed_by_reason'], ensure_ascii=False)}")
    print(f"Wrote {report['outputs']['sft_clean']}")
    print(f"Wrote {report['outputs']['preferences_clean']}")
    print(f"Wrote {report['artifacts']['markdown']}")
    print(f"Wrote {report['artifacts']['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
