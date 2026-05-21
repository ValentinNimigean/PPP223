import json

import pytest

from train.datasets import compute_dataset_hash, load_and_prepare_records, validate_jsonl_dataset
from train.tasks import format_record_as_messages, validate_task_record


def test_validate_qa_dataset_and_prepare_messages(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        json.dumps({"question": "What is Python?", "answer": "A programming language.", "context": "General knowledge"}) + "\n",
        encoding="utf-8",
    )

    records = validate_jsonl_dataset("qa", str(path))
    prepared, dataset_hash = load_and_prepare_records("qa", str(path))

    assert len(records) == 1
    assert len(prepared) == 1
    assert prepared[0]["messages"][0]["role"] == "user"
    assert "Context:" in prepared[0]["messages"][0]["content"]
    assert prepared[0]["messages"][1]["content"] == "A programming language."
    assert dataset_hash == compute_dataset_hash(str(path))


def test_validate_classification_and_format_messages():
    row = {"text": "This is positive.", "label": "positive"}
    validate_task_record("classification", row, 1)
    messages = format_record_as_messages("classification", row)
    assert "Classify the following text" in messages[0]["content"]
    assert messages[1]["content"] == "positive"


def test_validate_summarization_and_code_qa_messages():
    summarization = {"input": "Long article", "summary": "Short summary"}
    validate_task_record("summarization", summarization, 1)
    assert format_record_as_messages("summarization", summarization)[1]["content"] == "Short summary"

    code_qa = {
        "question": "Where is HybridRetriever defined?",
        "answer": "In rag/retriever.py.",
        "expected_files": ["rag/retriever.py"],
        "expected_entities": ["HybridRetriever"],
    }
    validate_task_record("code_qa", code_qa, 2)
    messages = format_record_as_messages("code_qa", code_qa)
    assert "Relevant files:" in messages[0]["content"]
    assert "Relevant entities:" in messages[0]["content"]


def test_bad_dataset_fails_with_helpful_error(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"question": "Missing answer"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field"):
        validate_jsonl_dataset("qa", str(path))


def test_invalid_jsonl_reports_row_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"question": "ok", "answer": "yes"}\n{not json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Row 2: invalid JSON"):
        validate_jsonl_dataset("qa", str(path))
