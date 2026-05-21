import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from alignment.reward_model import _refuse_raw_preference_dataset
from scripts.clean_datasets import run_cleaning
from scripts.split_datasets import run_split
from scripts.audit_datasets import run_audit
from train.config import SFTTrainConfig


ROOT = Path(__file__).resolve().parent.parent


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_sft_row(index: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a repository-grounded Python code assistant."},
            {"role": "user", "content": f"Where is RepoThing{index} defined?"},
            {"role": "assistant", "content": f"`RepoThing{index}` is defined in pkg/module_{index}.py."},
        ],
        "metadata": {
            "task_type": "symbol_location",
            "expected_files": [f"pkg/module_{index}.py"],
            "expected_entities": [f"RepoThing{index}"],
            "source": "repo_synthetic",
        },
    }


def make_pref_row(index: int) -> dict:
    return {
        "prompt": f"Where is RepoThing{index} defined?",
        "chosen": f"`RepoThing{index}` is defined in pkg/module_{index}.py.",
        "rejected": f"It is probably defined somewhere else, but I am not sure about RepoThing{index}.",
        "source": "synthetic",
        "rating": "positive",
        "metadata": {
            "task_type": "symbol_location",
            "failure_type": "hallucination" if index % 2 else "normal_grounded",
            "expected_files": [f"pkg/module_{index}.py"],
            "expected_entities": [f"RepoThing{index}"],
        },
        "safety_labels": {"toxic_prompt": False},
    }


def test_prepare_training_data_quick_runs(tmp_path):
    repo = tmp_path
    (repo / "scripts").mkdir()
    (repo / "data").mkdir()
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/usr/bin/env bash\nexec python \"$@\"\n", encoding="utf-8")
    (repo / ".venv" / "bin" / "python").chmod(0o755)
    (repo / "scripts" / "prepare_training_data.sh").write_text(
        (ROOT / "scripts" / "prepare_training_data.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "scripts" / "prepare_training_data.sh").chmod(0o755)
    for module_name in ("grimp", "qdrant_client", "fastembed", "tree_sitter", "tree_sitter_python", "pydantic", "openai"):
        (repo / f"{module_name}.py").write_text("# stub\n", encoding="utf-8")

    (repo / "scripts" / "audit_datasets.py").write_text(
        """
import json
import sys
from pathlib import Path

Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/dataset_audit.json").write_text("{}\\n", encoding="utf-8")
Path("artifacts/dataset_audit.md").write_text("# audit\\n", encoding="utf-8")
if "--paths" in sys.argv:
    for item in sys.argv[sys.argv.index("--paths") + 1:]:
        if item.startswith("--"):
            break
        if not Path(item).exists():
            raise SystemExit(f"missing final dataset: {item}")
print("ok")
""",
        encoding="utf-8",
    )
    (repo / "scripts" / "clean_datasets.py").write_text(
        """
from pathlib import Path
Path("data/clean").mkdir(parents=True, exist_ok=True)
Path("artifacts").mkdir(exist_ok=True)
Path("data/clean/sft_clean.jsonl").write_text("{}\\n", encoding="utf-8")
Path("data/clean/preferences_clean.jsonl").write_text("{}\\n", encoding="utf-8")
Path("artifacts/dataset_cleaning_report.md").write_text("# clean\\n", encoding="utf-8")
Path("artifacts/dataset_cleaning_report.json").write_text("{}\\n", encoding="utf-8")
print("cleaned")
""",
        encoding="utf-8",
    )
    (repo / "data" / "build_sft_dataset.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--report", default="artifacts/sft_dataset_report.md")
parser.add_argument("--profile", default="full")
parser.add_argument("--repo", default=".")
args = parser.parse_args()
Path("pkg").mkdir(exist_ok=True)
rows = []
for i in range(50):
    Path(f"pkg/module_{i}.py").write_text("print('ok')\\n", encoding="utf-8")
    rows.append({"messages":[{"role":"system","content":"sys"},{"role":"user","content":f"q{i}"},{"role":"assistant","content":f"a{i} in pkg/module_{i}.py"}],"metadata":{"task_type":"symbol_location","expected_files":[f"pkg/module_{i}.py"],"expected_entities":[f"Thing{i}"],"source":"repo_synthetic"}})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with Path(args.out).open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\\n")
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text("# sft\\n", encoding="utf-8")
print(args.out)
""",
        encoding="utf-8",
    )
    (repo / "data" / "build_preference_dataset.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--report", default="artifacts/preference_dataset_report.md")
parser.add_argument("--profile", default="full")
parser.add_argument("--repo", default=".")
parser.add_argument("--sft", default="")
args = parser.parse_args()
rows = []
for i in range(50):
    rows.append({"prompt":f"q{i}","chosen":f"a{i} in pkg/module_{i}.py","rejected":f"bad{i}","source":"synthetic","rating":"positive","metadata":{"task_type":"symbol_location","failure_type":"normal_grounded","expected_files":[f"pkg/module_{i}.py"],"expected_entities":[f"Thing{i}"]},"safety_labels":{"toxic_prompt":False}})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with Path(args.out).open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\\n")
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text("# pref\\n", encoding="utf-8")
print(args.out)
""",
        encoding="utf-8",
    )
    (repo / "scripts" / "split_datasets.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--final-dir", default="data/final")
parser.add_argument("--report", default="artifacts/dataset_split_report.md")
parser.add_argument("--sft", required=True)
parser.add_argument("--preferences", required=True)
parser.add_argument("--seed", default="3407")
args = parser.parse_args()
final_dir = Path(args.final_dir)
final_dir.mkdir(parents=True, exist_ok=True)
sft_rows = [json.loads(line) for line in Path(args.sft).read_text(encoding="utf-8").splitlines() if line.strip()]
pref_rows = [json.loads(line) for line in Path(args.preferences).read_text(encoding="utf-8").splitlines() if line.strip()]
for name, rows in {
    "sft_train.jsonl": sft_rows[:40],
    "sft_val.jsonl": sft_rows[40:45],
    "sft_test_holdout.jsonl": sft_rows[45:50],
    "preferences_train.jsonl": pref_rows[:40],
    "preferences_val.jsonl": pref_rows[40:45],
    "preferences_test_holdout.jsonl": pref_rows[45:50],
}.items():
    with (final_dir / name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\\n")
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text("# split\\n", encoding="utf-8")
print("split")
""",
        encoding="utf-8",
    )

    subprocess.run(["bash", "scripts/prepare_training_data.sh", "--quick"], cwd=repo, check=True)

    assert (repo / "data/curated/sft_repo_qa.jsonl").exists()
    assert (repo / "data/curated/preferences_rlhf.jsonl").exists()
    assert (repo / "data/final/sft_train.jsonl").exists()
    assert (repo / "data/final/preferences_train.jsonl").exists()
    assert (repo / "artifacts/dataset_split_report.md").exists()


def test_final_dataset_paths_are_required_for_training(monkeypatch):
    assert SFTTrainConfig().dataset_path == "data/final/sft_train.jsonl"
    assert SFTTrainConfig().val_dataset_path == "data/final/sft_val.jsonl"

    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda _seed: None
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sys.modules.pop("train.sft", None)
    train_sft = importlib.import_module("train.sft")

    with pytest.raises(ValueError):
        train_sft._refuse_raw_dataset("synthetic_qa_combined_2048.jsonl", False)
    train_sft._refuse_raw_dataset("data/final/sft_train.jsonl", False)

    with pytest.raises(ValueError):
        _refuse_raw_preference_dataset("preference_data_combined.jsonl", False)
    _refuse_raw_preference_dataset("data/final/preferences_train.jsonl", False)


def test_artifacts_are_not_required_source_files(tmp_path):
    sft_raw = tmp_path / "synthetic_qa_seed.jsonl"
    pref_raw = tmp_path / "preference_data_combined.jsonl"
    write_jsonl(
        sft_raw,
        [{"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}]}],
    )
    write_jsonl(pref_raw, [{"prompt": "q", "chosen": "chosen", "rejected": "rejected"}])

    report = run_cleaning(sft_files=[sft_raw], preference_files=[pref_raw], repo_root=tmp_path)

    assert (tmp_path / "artifacts").exists()
    assert Path(report["artifacts"]["markdown"]).exists()
    assert Path(report["outputs"]["sft_clean"]).exists()
    assert Path(report["outputs"]["preferences_clean"]).exists()


def test_docs_do_not_present_dpo_as_main_rlhf():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    how_to = (ROOT / "HOW_TO_USE.md").read_text(encoding="utf-8")
    rlhf = (ROOT / "docs" / "RLHF_PPO.md").read_text(encoding="utf-8")
    grading = (ROOT / "docs" / "GRADING_CHECKLIST.md").read_text(encoding="utf-8")

    assert "PPO-based RLHF is the main" in how_to
    assert "optional baseline" in rlhf
    assert "PPO-based RLHF is the main RLHF implementation" in grading
    assert "DPO / RLHF-style preference optimization" not in how_to
    assert "Raw helper files such as `synthetic_qa_combined_2048.jsonl` and `preference_data_combined.jsonl` are not valid training inputs" in readme


def test_no_absolute_paths_in_generated_final_datasets(tmp_path):
    for index in range(12):
        module = tmp_path / f"pkg/module_{index}.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("print('ok')\n", encoding="utf-8")

    sft_curated = tmp_path / "data/curated/sft_repo_qa.jsonl"
    pref_curated = tmp_path / "data/curated/preferences_rlhf.jsonl"
    benchmark_self = tmp_path / "eval/benchmark_self.json"
    benchmark_unseen = tmp_path / "eval/benchmark_unseen_self.json"
    benchmark_self.parent.mkdir(parents=True, exist_ok=True)
    benchmark_self.write_text("[]", encoding="utf-8")
    benchmark_unseen.write_text("[]", encoding="utf-8")
    write_jsonl(sft_curated, [make_sft_row(index) for index in range(12)])
    write_jsonl(pref_curated, [make_pref_row(index) for index in range(12)])

    run_split(
        sft_path=sft_curated,
        pref_path=pref_curated,
        benchmark_paths=[benchmark_self, benchmark_unseen],
        final_dir=tmp_path / "data/final",
        report_path=tmp_path / "artifacts/dataset_split_report.md",
        seed=3407,
    )
    audit = run_audit(
        sft_files=[tmp_path / "data/final/sft_train.jsonl"],
        preference_files=[tmp_path / "data/final/preferences_train.jsonl"],
        eval_files=[benchmark_self, benchmark_unseen],
        repo_root=tmp_path,
    )

    assert audit["sft_datasets"][0]["contamination_rows"] == 0
    assert audit["sft_datasets"][0]["hallucinated_file_path_rows"] == 0
    assert audit["preference_datasets"][0]["contamination_rows"] == 0
