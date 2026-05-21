import json
from pathlib import Path

from scripts.split_datasets import run_split


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_sft_row(prompt: str, answer: str, task_type: str, source: str):
    return {
        "messages": [
            {"role": "system", "content": "You are a repository-grounded Python code assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "task_type": task_type,
            "expected_files": ["pkg/core.py"] if "cannot find" not in answer.lower() else [],
            "expected_entities": ["CoreThing"] if "cannot find" not in answer.lower() else ["MissingThing"],
            "source": source,
            "difficulty": "medium",
        },
    }


def make_pref_row(prompt: str, chosen: str, rejected: str, task_type: str, source: str):
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
        "rating": "positive",
        "metadata": {
            "expected_files": ["pkg/core.py"] if "cannot find" not in chosen.lower() else [],
            "expected_entities": ["CoreThing"] if "cannot find" not in chosen.lower() else ["MissingThing"],
            "failure_type": "normal_grounded",
            "task_type": task_type,
        },
        "safety_labels": {"toxic_prompt": False},
    }


def build_inputs(tmp_path: Path):
    (tmp_path / "pkg").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pkg" / "core.py").write_text("class CoreThing:\n    pass\n", encoding="utf-8")

    sft_rows = [
        make_sft_row("What class handles hybrid vector search?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("What class handles hybrid vector search in this repo", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("Where is RepoMapGenerator defined?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("Explain the loader fallback path.", "`CoreThing` is implemented in pkg/core.py.", "behavior", "repo_synthetic"),
        make_sft_row("Does the repo define MissingThing?", "I cannot find evidence for `MissingThing` in this repository.", "negative_no_match", "repo_synthetic"),
        make_sft_row("Which file defines HelperOne?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("Which file defines HelperTwo?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("Which file defines HelperThree?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("Which file defines HelperFour?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("Which file defines HelperFive?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("Which file defines HelperSix?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
        make_sft_row("Which file defines HelperSeven?", "`CoreThing` is implemented in pkg/core.py.", "symbol_location", "repo_synthetic"),
    ]
    pref_rows = [
        make_pref_row("What class handles hybrid vector search?", "`CoreThing` is implemented in pkg/core.py.", "Generic unsupported answer.", "symbol_location", "synthetic"),
        make_pref_row("What class handles hybrid vector search in this repo", "`CoreThing` is implemented in pkg/core.py.", "Generic unsupported answer.", "symbol_location", "synthetic"),
        make_pref_row("Where is RepoMapGenerator defined?", "`CoreThing` is implemented in pkg/core.py.", "Wrong file.", "symbol_location", "synthetic"),
        make_pref_row("Explain the loader fallback path.", "`CoreThing` is implemented in pkg/core.py.", "Wrong file.", "behavior", "synthetic"),
        make_pref_row("Does the repo define MissingThing?", "I cannot find evidence for `MissingThing` in this repository.", "`MissingThing` definitely exists.", "negative_no_match", "synthetic"),
        make_pref_row("Which file defines HelperOne?", "`CoreThing` is implemented in pkg/core.py.", "Unsupported answer.", "symbol_location", "human"),
        make_pref_row("Which file defines HelperTwo?", "`CoreThing` is implemented in pkg/core.py.", "Unsupported answer.", "symbol_location", "eval_failure"),
        make_pref_row("Which file defines HelperThree?", "`CoreThing` is implemented in pkg/core.py.", "Unsupported answer.", "symbol_location", "synthetic"),
        make_pref_row("Which file defines HelperFour?", "`CoreThing` is implemented in pkg/core.py.", "Unsupported answer.", "symbol_location", "synthetic"),
        make_pref_row("Which file defines HelperFive?", "`CoreThing` is implemented in pkg/core.py.", "Unsupported answer.", "symbol_location", "synthetic"),
        make_pref_row("Which file defines HelperSix?", "`CoreThing` is implemented in pkg/core.py.", "Unsupported answer.", "symbol_location", "synthetic"),
        make_pref_row("Which file defines HelperSeven?", "`CoreThing` is implemented in pkg/core.py.", "Unsupported answer.", "symbol_location", "synthetic"),
    ]

    sft_path = tmp_path / "data" / "curated" / "sft_repo_qa.jsonl"
    pref_path = tmp_path / "data" / "curated" / "preferences_rlhf.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(pref_path, pref_rows)

    benchmark_self = tmp_path / "eval" / "benchmark_self.json"
    benchmark_unseen = tmp_path / "eval" / "benchmark_unseen_self.json"
    benchmark_self.parent.mkdir(parents=True, exist_ok=True)
    benchmark_self.write_text(json.dumps([{"question": "What class handles hybrid vector search?"}]), encoding="utf-8")
    benchmark_unseen.write_text(json.dumps([{"question": "Where is SomethingElse defined?"}]), encoding="utf-8")
    return sft_path, pref_path, [benchmark_self, benchmark_unseen]


def test_split_prevents_prompt_overlap_and_benchmark_leakage(tmp_path):
    sft_path, pref_path, benchmarks = build_inputs(tmp_path)
    final_dir = tmp_path / "data" / "final"
    report_path = tmp_path / "artifacts" / "dataset_split_report.md"

    report = run_split(
        sft_path=sft_path,
        pref_path=pref_path,
        benchmark_paths=benchmarks,
        final_dir=final_dir,
        report_path=report_path,
        seed=17,
    )

    sft_train = read_jsonl(final_dir / "sft_train.jsonl")
    sft_val = read_jsonl(final_dir / "sft_val.jsonl")
    sft_holdout = read_jsonl(final_dir / "sft_test_holdout.jsonl")

    train_prompts = {row["metadata"]["normalized_prompt"] for row in sft_train}
    val_prompts = {row["metadata"]["normalized_prompt"] for row in sft_val}
    holdout_prompts = {row["metadata"]["normalized_prompt"] for row in sft_holdout}

    assert not (train_prompts & val_prompts)
    assert not (train_prompts & holdout_prompts)
    assert not (val_prompts & holdout_prompts)
    assert "what class handles hybrid vector search" not in train_prompts
    assert "what class handles hybrid vector search in this repo" not in train_prompts
    assert not report["errors"]


def test_duplicate_prompt_groups_stay_in_single_split(tmp_path):
    sft_path, pref_path, benchmarks = build_inputs(tmp_path)
    duplicated_pref = read_jsonl(pref_path)
    duplicated_pref.append(duplicated_pref[0])
    write_jsonl(pref_path, duplicated_pref)

    final_dir = tmp_path / "data" / "final"
    report_path = tmp_path / "artifacts" / "dataset_split_report.md"
    run_split(
        sft_path=sft_path,
        pref_path=pref_path,
        benchmark_paths=benchmarks,
        final_dir=final_dir,
        report_path=report_path,
        seed=17,
    )

    all_pref = {
        "train": read_jsonl(final_dir / "preferences_train.jsonl"),
        "val": read_jsonl(final_dir / "preferences_val.jsonl"),
        "holdout": read_jsonl(final_dir / "preferences_test_holdout.jsonl"),
    }
    prompt_locations = {}
    for split_name, rows in all_pref.items():
        for row in rows:
            prompt_locations.setdefault(row["metadata"]["normalized_prompt"], set()).add(split_name)

    assert all(len(splits) == 1 for splits in prompt_locations.values())


def test_split_report_lists_task_type_and_source_counts(tmp_path):
    sft_path, pref_path, benchmarks = build_inputs(tmp_path)
    final_dir = tmp_path / "data" / "final"
    report_path = tmp_path / "artifacts" / "dataset_split_report.md"

    run_split(
        sft_path=sft_path,
        pref_path=pref_path,
        benchmark_paths=benchmarks,
        final_dir=final_dir,
        report_path=report_path,
        seed=17,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "task_type counts" in report_text
    assert "source counts" in report_text
    assert "symbol_location" in report_text
    assert "synthetic" in report_text or "repo_synthetic" in report_text
