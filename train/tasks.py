"""Task schemas and record formatting for SFT."""

from __future__ import annotations

from typing import Any


TASK_SCHEMAS: dict[str, dict[str, tuple[str, ...]]] = {
    "chat": {"required": ("messages",)},
    "qa": {"required": ("question", "answer"), "optional": ("context",)},
    "classification": {"required": ("text", "label")},
    "summarization": {"required": ("input", "summary")},
    "code_qa": {"required": ("question", "answer", "expected_files"), "optional": ("expected_entities",)},
}


def _missing_required(task: str, record: dict[str, Any]) -> list[str]:
    schema = TASK_SCHEMAS[task]
    return [field for field in schema["required"] if field not in record or record.get(field) in (None, "", [])]


def validate_task_record(task: str, record: dict[str, Any], row_number: int) -> None:
    """Validate a single JSONL row for a given task."""
    if task not in TASK_SCHEMAS:
        raise ValueError(f"Unsupported task `{task}`. Expected one of: {sorted(TASK_SCHEMAS)}")

    if not isinstance(record, dict):
        raise ValueError(f"Row {row_number}: expected a JSON object, found {type(record).__name__}.")

    missing = _missing_required(task, record)
    if missing:
        raise ValueError(
            f"Row {row_number}: task `{task}` is missing required field(s): {', '.join(missing)}."
        )

    if task == "chat":
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Row {row_number}: task `chat` requires a non-empty `messages` list.")

    if task == "code_qa":
        expected_files = record.get("expected_files")
        if not isinstance(expected_files, list) or not expected_files:
            raise ValueError(f"Row {row_number}: task `code_qa` requires non-empty `expected_files`.")


def _qa_user_prompt(question: str, context: str | None = None) -> str:
    if context:
        return f"Context:\n{context}\n\nQuestion:\n{question}"
    return question


def _classification_prompt(text: str) -> str:
    return f"Classify the following text.\n\nText:\n{text}\n\nLabel:"


def _summarization_prompt(text: str) -> str:
    return f"Summarize the following input.\n\nInput:\n{text}\n\nSummary:"


def _code_qa_prompt(
    question: str,
    expected_files: list[Any],
    expected_entities: list[Any] | None = None,
) -> str:
    hints = []
    if expected_files:
        hints.append("Relevant files: " + ", ".join(str(item) for item in expected_files))
    if expected_entities:
        hints.append("Relevant entities: " + ", ".join(str(item) for item in expected_entities))
    if hints:
        return f"{question}\n\nHints:\n" + "\n".join(hints)
    return question


def format_record_as_messages(task: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a task row into chat-style messages."""
    if task == "chat":
        return record["messages"]
    if task == "qa":
        return [
            {"role": "user", "content": _qa_user_prompt(record["question"], record.get("context"))},
            {"role": "assistant", "content": record["answer"]},
        ]
    if task == "classification":
        return [
            {"role": "user", "content": _classification_prompt(record["text"])},
            {"role": "assistant", "content": str(record["label"])},
        ]
    if task == "summarization":
        return [
            {"role": "user", "content": _summarization_prompt(record["input"])},
            {"role": "assistant", "content": record["summary"]},
        ]
    if task == "code_qa":
        return [
            {
                "role": "user",
                "content": _code_qa_prompt(
                    record["question"],
                    record["expected_files"],
                    record.get("expected_entities"),
                ),
            },
            {"role": "assistant", "content": record["answer"]},
        ]
    raise ValueError(f"Unsupported task `{task}`.")
