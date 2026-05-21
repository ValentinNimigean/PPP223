import json
from pathlib import Path

from scripts.clean_datasets import run_cleaning


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cleaning_removes_empty_and_duplicate_sft_rows(tmp_path):
    sft_path = tmp_path / "synthetic_qa_seed.jsonl"
    pref_path = tmp_path / "preference_data_combined.jsonl"

    duplicate = {
        "messages": [
            {"role": "system", "content": " sys "},
            {"role": "user", "content": "Where is src.py?"},
            {"role": "assistant", "content": " See src.py for details. "},
        ]
    }
    duplicate_copy = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Where is src.py?"},
            {"role": "assistant", "content": "See src.py for details."},
        ]
    }
    similar_question = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Where is src.py?"},
            {"role": "assistant", "content": "See src.py for more details."},
        ]
    }
    different_answer = {
        "messages": [
            {"role": "user", "content": "Where is src.py?"},
            {
                "role": "assistant",
                "content": "I cannot confirm a single canonical file for this question; the implementation appears split across src.py and docs/reference.py.",
            },
        ]
    }
    empty_answer = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Bad row"},
            {"role": "assistant", "content": ""},
        ]
    }
    null_answer = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Bad row 2"},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "tool"},
            {"role": "assistant", "content": "final"},
        ]
    }
    invalid_order = {
        "messages": [
            {"role": "assistant", "content": "wrong"},
            {"role": "user", "content": "Question"},
        ]
    }
    write_jsonl(sft_path, [duplicate, duplicate_copy, similar_question, different_answer, empty_answer, null_answer, invalid_order])
    write_jsonl(pref_path, [])

    report = run_cleaning([sft_path], [pref_path], repo_root=tmp_path)
    sft_rows = read_jsonl(tmp_path / "data" / "clean" / "sft_clean.jsonl")

    assert len(sft_rows) == 2
    assert all(row["messages"][-1]["content"] for row in sft_rows)
    assert len({json.dumps({"messages": row["messages"]}, sort_keys=True) for row in sft_rows}) == len(sft_rows)
    assert all("metadata" in row for row in sft_rows)
    assert sft_rows[0]["metadata"]["source_file"] == "synthetic_qa_seed.jsonl"
    assert report["datasets"]["sft"]["removed_by_reason"]["exact_duplicate_row"] == 1
    assert report["datasets"]["sft"]["removed_by_reason"]["duplicate_user_question_similar_answer"] == 1
    assert report["datasets"]["sft"]["removed_by_reason"]["assistant_content_empty"] == 1
    assert report["datasets"]["sft"]["removed_by_reason"]["assistant_content_null"] == 1


def test_cleaning_preserves_tool_trace_with_final_answer(tmp_path):
    sft_path = tmp_path / "synthetic_qa_seed.jsonl"
    pref_path = tmp_path / "preference_data_combined.jsonl"
    write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Where is the retriever?"},
                    {"role": "assistant", "content": "Working", "tool_calls": [{"id": "1", "function": {"name": "grep"}}]},
                    {"role": "tool", "content": "rag/retriever.py"},
                    {"role": "assistant", "content": "The retriever is in rag/retriever.py."},
                ]
            }
        ],
    )
    write_jsonl(pref_path, [])

    run_cleaning([sft_path], [pref_path], repo_root=tmp_path)
    sft_rows = read_jsonl(tmp_path / "data" / "clean" / "sft_clean.jsonl")

    assert len(sft_rows) == 1
    assert sft_rows[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "grep"
    assert sft_rows[0]["messages"][-1]["content"] == "The retriever is in rag/retriever.py."


def test_cleaning_canonicalizes_preferences_and_drops_invalid_rows(tmp_path):
    sft_path = tmp_path / "synthetic_qa_seed.jsonl"
    pref_path = tmp_path / "preference_data_combined.jsonl"
    write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Good answer for keeping the pipeline valid."},
                ]
            }
        ],
    )
    write_jsonl(
        pref_path,
        [
            {"prompt": "Explain repo", "chosen": " useful chosen ", "rejected": " rejected answer ", "rating": 0.75},
            {"prompt": "Explain repo", "chosen": "same", "rejected": "same"},
            {"prompt": "Another", "chosen": "valid", "rejected": ""},
        ],
    )

    report = run_cleaning([sft_path], [pref_path], repo_root=tmp_path)
    pref_rows = read_jsonl(tmp_path / "data" / "clean" / "preferences_clean.jsonl")

    assert len(pref_rows) == 1
    row = pref_rows[0]
    assert set(row) == {"prompt", "chosen", "rejected", "source", "rating", "metadata", "safety_labels"}
    assert row["prompt"] == "Explain repo"
    assert row["chosen"] == "useful chosen"
    assert row["rejected"] == "rejected answer"
    assert row["metadata"]["source_line"] == 1
    assert report["datasets"]["preferences"]["removed_by_reason"]["chosen_equals_rejected"] == 1
    assert report["datasets"]["preferences"]["removed_by_reason"]["empty_rejected"] == 1


def test_toy_preference_rows_removed_unless_marked_safety(tmp_path):
    sft_path = tmp_path / "synthetic_qa_seed.jsonl"
    pref_path = tmp_path / "preference_data_combined.jsonl"
    write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Good answer for keeping the pipeline valid."},
                ]
            }
        ],
    )
    write_jsonl(
        pref_path,
        [
            {"prompt": "i could kiss you if could", "chosen": "chosen", "rejected": "rejected"},
            {
                "prompt": "You are stupid and worthless.",
                "chosen": "Refuse abuse and ask for a repo question.",
                "rejected": "Answer normally.",
                "safety_labels": ["abuse"],
            },
        ],
    )

    report = run_cleaning([sft_path], [pref_path], repo_root=tmp_path)
    pref_rows = read_jsonl(tmp_path / "data" / "clean" / "preferences_clean.jsonl")

    assert len(pref_rows) == 1
    assert pref_rows[0]["safety_labels"]
    assert report["datasets"]["preferences"]["removed_by_reason"]["toy_or_joke_row"] == 1


def test_cleaning_writes_reports_with_removal_reasons(tmp_path):
    sft_path = tmp_path / "synthetic_qa_seed.jsonl"
    pref_path = tmp_path / "preference_data_combined.jsonl"
    write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": ""},
                ]
            }
        ],
    )
    write_jsonl(pref_path, [{"prompt": "Explain", "chosen": "ok", "rejected": ""}])

    report = run_cleaning([sft_path], [pref_path], repo_root=tmp_path)
    json_report = json.loads((tmp_path / "artifacts" / "dataset_cleaning_report.json").read_text(encoding="utf-8"))
    md_report = (tmp_path / "artifacts" / "dataset_cleaning_report.md").read_text(encoding="utf-8")

    assert report["datasets"]["sft"]["removed_by_reason"]["assistant_content_empty"] == 1
    assert json_report["datasets"]["preferences"]["removed_by_reason"]["empty_rejected"] == 1
    assert "assistant_content_empty" in md_report
    assert "empty_rejected" in md_report
