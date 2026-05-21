import json
import sys
import types
from pathlib import Path

if "pydantic" not in sys.modules:
    fake_pydantic = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _field(default=None, **_kwargs):
        return default

    fake_pydantic.BaseModel = _BaseModel
    fake_pydantic.Field = _field
    sys.modules["pydantic"] = fake_pydantic

from alignment.feedback_schema import load_preferences
from data import build_preference_dataset as pref_builder


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_sft_row(prompt, answer, task_type, expected_files, expected_entities, difficulty="medium", tool_use=False):
    messages = [
        {
            "role": "system",
            "content": "You are a repository-grounded Python code assistant. Answer only from provided repository evidence and cite file paths.",
        },
        {"role": "user", "content": prompt},
    ]
    if tool_use:
        messages.append(
            {
                "role": "assistant",
                "content": "I will inspect repository evidence first.",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\":\"x\"}"}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": "call_1", "name": "grep_search", "content": "evidence"})
    messages.append({"role": "assistant", "content": answer})
    return {
        "messages": messages,
        "metadata": {
            "task_type": task_type,
            "expected_files": expected_files,
            "expected_entities": expected_entities,
            "source": "repo_synthetic",
            "difficulty": difficulty,
        },
    }


def build_sft_fixture(tmp_path: Path) -> Path:
    rows = []
    for index in range(1, 121):
        filepath = f"pkg/module_{index}.py"
        (tmp_path / filepath).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / filepath).write_text(f"# module {index}\n", encoding="utf-8")
        rows.append(
            make_sft_row(
                prompt=f"Where is RepoClass{index} defined?",
                answer=f"`RepoClass{index}` is implemented in {filepath}.",
                task_type="symbol_location",
                expected_files=[filepath],
                expected_entities=[f"RepoClass{index}"],
                difficulty="easy",
                tool_use=index <= 60,
            )
        )
    for index in range(1, 41):
        rows.append(
            make_sft_row(
                prompt=f"Does the repo define MissingThing{index}?",
                answer=f"I cannot find evidence for `MissingThing{index}` in this repository.",
                task_type="negative_no_match",
                expected_files=[],
                expected_entities=[f"MissingThing{index}"],
                difficulty="medium",
            )
        )
    for index in range(1, 41):
        rows.append(
            make_sft_row(
                prompt=f"You are stupid. Explain RepoClass{index}.",
                answer=f"I cannot help with abusive language. Please rephrase the request neutrally, for example by asking about `RepoClass{index}`.",
                task_type="safety",
                expected_files=[],
                expected_entities=[f"RepoClass{index}"],
                difficulty="easy",
            )
        )
    sft_path = tmp_path / "data" / "curated" / "sft_repo_qa.jsonl"
    write_jsonl(sft_path, rows)
    return sft_path


def test_build_preference_dataset_creates_valid_targeted_pairs(tmp_path):
    sft_path = build_sft_fixture(tmp_path)
    benchmark_self = tmp_path / "benchmark_self.json"
    benchmark_unseen = tmp_path / "benchmark_unseen_self.json"
    benchmark_self.write_text("[]", encoding="utf-8")
    benchmark_unseen.write_text("[]", encoding="utf-8")
    existing_pref = tmp_path / "preference_data_combined.jsonl"
    existing_pref.write_text("", encoding="utf-8")
    human_feedback = tmp_path / "human_feedback.jsonl"
    human_feedback.write_text("", encoding="utf-8")
    out_path = tmp_path / "data" / "curated" / "preferences_rlhf.jsonl"
    report_path = tmp_path / "artifacts" / "preference_dataset_report.md"

    original_eval_paths = pref_builder.eval_report_paths
    pref_builder.eval_report_paths = lambda: []
    try:
        report = pref_builder.build_preference_dataset(
            sft_path=sft_path,
            benchmark_paths=[benchmark_self, benchmark_unseen],
            existing_pref_path=existing_pref,
            human_feedback_path=human_feedback,
            out_path=out_path,
            report_path=report_path,
        )
    finally:
        pref_builder.eval_report_paths = original_eval_paths

    rows = load_preferences(str(out_path))
    assert len(rows) >= 300
    assert not report["validation"]["errors"]
    assert report["validation"]["failure_type_counts"]["hallucination"] >= 50
    assert report["validation"]["failure_type_counts"]["no_match_honesty"] >= 30
    assert report["validation"]["failure_type_counts"]["toxicity_safety"] >= 30
    assert report["validation"]["failure_type_counts"]["malformed_tool"] >= 50
    assert report["validation"]["failure_type_counts"]["normal_grounded"] >= 100


def test_preference_rows_are_canonical_and_unique(tmp_path):
    sft_path = build_sft_fixture(tmp_path)
    out_path = tmp_path / "data" / "curated" / "preferences_rlhf.jsonl"
    report_path = tmp_path / "artifacts" / "preference_dataset_report.md"
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    existing_pref = tmp_path / "preference_data_combined.jsonl"
    existing_pref.write_text("", encoding="utf-8")
    human_feedback = tmp_path / "human_feedback.jsonl"
    human_feedback.write_text("", encoding="utf-8")

    original_eval_paths = pref_builder.eval_report_paths
    pref_builder.eval_report_paths = lambda: []
    try:
        pref_builder.build_preference_dataset(
            sft_path=sft_path,
            benchmark_paths=[empty, empty],
            existing_pref_path=existing_pref,
            human_feedback_path=human_feedback,
            out_path=out_path,
            report_path=report_path,
        )
    finally:
        pref_builder.eval_report_paths = original_eval_paths

    rows = load_preferences(str(out_path))
    triples = {(row["prompt"], row["chosen"], row["rejected"]) for row in rows}
    assert len(triples) == len(rows)
    assert all(row["chosen"] != row["rejected"] for row in rows)
    for row in rows:
        expected_files = row["metadata"].get("expected_files") or []
        if expected_files:
            assert all(file_path in row["chosen"] for file_path in expected_files)


def test_report_lists_failure_and_task_distributions(tmp_path):
    sft_path = build_sft_fixture(tmp_path)
    out_path = tmp_path / "data" / "curated" / "preferences_rlhf.jsonl"
    report_path = tmp_path / "artifacts" / "preference_dataset_report.md"
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    existing_pref = tmp_path / "preference_data_combined.jsonl"
    existing_pref.write_text("", encoding="utf-8")
    human_feedback = tmp_path / "human_feedback.jsonl"
    human_feedback.write_text("", encoding="utf-8")

    original_eval_paths = pref_builder.eval_report_paths
    pref_builder.eval_report_paths = lambda: []
    try:
        pref_builder.build_preference_dataset(
            sft_path=sft_path,
            benchmark_paths=[empty, empty],
            existing_pref_path=existing_pref,
            human_feedback_path=human_feedback,
            out_path=out_path,
            report_path=report_path,
        )
    finally:
        pref_builder.eval_report_paths = original_eval_paths

    report_text = report_path.read_text(encoding="utf-8")
    assert "Failure Type Distribution" in report_text
    assert "Task Type Distribution" in report_text
    assert "normal_grounded" in report_text
    assert "symbol_location" in report_text or "negative_no_match" in report_text


def test_quick_profile_targets_can_build_smaller_preference_dataset(tmp_path):
    sft_path = build_sft_fixture(tmp_path)
    out_path = tmp_path / "data" / "curated" / "preferences_rlhf.jsonl"
    report_path = tmp_path / "artifacts" / "preference_dataset_report.md"
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    existing_pref = tmp_path / "preference_data_combined.jsonl"
    existing_pref.write_text("", encoding="utf-8")
    human_feedback = tmp_path / "human_feedback.jsonl"
    human_feedback.write_text("", encoding="utf-8")

    original_eval_paths = pref_builder.eval_report_paths
    pref_builder.eval_report_paths = lambda: []
    try:
        report = pref_builder.build_preference_dataset(
            sft_path=sft_path,
            benchmark_paths=[empty, empty],
            existing_pref_path=existing_pref,
            human_feedback_path=human_feedback,
            out_path=out_path,
            report_path=report_path,
            min_rows=50,
            targets=pref_builder.QUICK_TARGETS,
        )
    finally:
        pref_builder.eval_report_paths = original_eval_paths

    assert report["validation"]["total_rows"] >= 50
    assert not report["validation"]["errors"]
