import json
from pathlib import Path

from scripts.audit_datasets import infer_dataset_kind, main as audit_main, run_audit


def write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_eval_json(path: Path, questions):
    payload = [{"question": question, "expected_files": []} for question in questions]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_duplicate_detection(tmp_path):
    sft_path = tmp_path / "sft.jsonl"
    pref_path = tmp_path / "pref.jsonl"
    eval_path = tmp_path / "eval.json"
    (tmp_path / "src.py").write_text("print('ok')\n", encoding="utf-8")

    duplicate_row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Where is src.py?"},
            {"role": "assistant", "content": "See src.py for the implementation details."},
        ]
    }
    write_jsonl(sft_path, [duplicate_row, duplicate_row])
    write_jsonl(
        pref_path,
        [{"prompt": f"prompt {i}", "chosen": "chosen answer is long enough", "rejected": "rejected answer is long enough"} for i in range(100)],
    )
    write_eval_json(eval_path, [])

    audit = run_audit([sft_path], [pref_path], [eval_path], repo_root=tmp_path)

    assert audit["sft_datasets"][0]["duplicate_exact_rows"] == 1
    assert audit["sft_datasets"][0]["duplicate_user_questions"] == 1
    assert any("duplicate exact rows exceed 5%" in failure for failure in audit["failures"])


def test_empty_assistant_detection(tmp_path):
    sft_path = tmp_path / "sft.jsonl"
    pref_path = tmp_path / "pref.jsonl"
    eval_path = tmp_path / "eval.json"
    (tmp_path / "src.py").write_text("print('ok')\n", encoding="utf-8")

    write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": None},
                ]
            }
        ],
    )
    write_jsonl(
        pref_path,
        [{"prompt": f"prompt {i}", "chosen": "chosen answer is long enough", "rejected": "rejected answer is long enough"} for i in range(100)],
    )
    write_eval_json(eval_path, [])

    audit = run_audit([sft_path], [pref_path], [eval_path], repo_root=tmp_path)

    report = audit["sft_datasets"][0]
    assert report["empty_assistant_answers"] == 1
    assert report["assistant_answers_with_none"] == 1
    assert any("empty assistant answers" in failure for failure in audit["failures"])


def test_invalid_preference_pair_detection(tmp_path):
    sft_path = tmp_path / "sft.jsonl"
    pref_path = tmp_path / "pref.jsonl"
    eval_path = tmp_path / "eval.json"
    (tmp_path / "src.py").write_text("print('ok')\n", encoding="utf-8")

    write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "This answer is definitely long enough."},
                ]
            }
        ],
    )
    write_jsonl(pref_path, [{"prompt": "Explain", "chosen": "", "rejected": ""}])
    write_eval_json(eval_path, [])

    audit = run_audit([sft_path], [pref_path], [eval_path], repo_root=tmp_path)

    report = audit["preference_datasets"][0]
    assert report["empty_chosen"] == 1
    assert report["empty_rejected"] == 1
    assert report["valid_rows"] == 0
    assert any("empty pairs" in failure for failure in audit["failures"])
    assert any("fewer than 100 valid pairs" in failure for failure in audit["failures"])


def test_contamination_and_hallucinated_path_detection(tmp_path):
    sft_path = tmp_path / "sft.jsonl"
    pref_path = tmp_path / "pref.jsonl"
    eval_path = tmp_path / "eval.json"
    (tmp_path / "src.py").write_text("print('ok')\n", encoding="utf-8")

    write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "Use /home/user/Documents/GitHub/PPP223/fake.py and also inspect missing/module.py for details.",
                    },
                ]
            }
        ],
    )
    write_jsonl(
        pref_path,
        [{"prompt": f"prompt {i}", "chosen": "chosen answer is long enough", "rejected": "rejected answer is long enough"} for i in range(100)],
    )
    write_eval_json(eval_path, ["Question"])

    audit = run_audit([sft_path], [pref_path], [eval_path], repo_root=tmp_path)

    report = audit["sft_datasets"][0]
    assert report["contamination_rows"] == 1
    assert report["hallucinated_file_path_rows"] == 1
    assert audit["train_eval_leakage"]["total_exact_overlaps"] == 1


def test_mixed_paths_and_allow_fail_cli(tmp_path):
    sft_path = tmp_path / "sft.jsonl"
    pref_path = tmp_path / "pref.jsonl"
    (tmp_path / "src.py").write_text("print('ok')\n", encoding="utf-8")

    write_jsonl(
        sft_path,
        [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": None},
                ]
            }
        ],
    )
    write_jsonl(
        pref_path,
        [{"prompt": "Prompt", "chosen": "chosen answer is long enough", "rejected": "rejected answer is long enough"}],
    )

    assert infer_dataset_kind(sft_path) == "sft"
    assert infer_dataset_kind(pref_path) == "preference"
    assert audit_main(["--paths", str(sft_path), str(pref_path), "--allow-fail"]) == 0
