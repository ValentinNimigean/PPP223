import json
from pathlib import Path
from types import SimpleNamespace

from data.build_sft_dataset import QUICK_CATEGORY_TARGETS, generate_dataset_from_inputs
from scripts.audit_datasets import audit_sft_dataset


class FakeDepGraph:
    def __init__(self, intra_file_deps):
        self.intra_file_deps = intra_file_deps


def make_chunk(filepath: str, chunk_type: str, name: str, start_line: int, end_line: int, text: str, **kwargs):
    payload = {
        "filepath": filepath,
        "chunk_type": chunk_type,
        "name": name,
        "qualified_name": kwargs.get("qualified_name", name),
        "parent_class": kwargs.get("parent_class"),
        "start_line": start_line,
        "end_line": end_line,
        "text": text,
        "decorators": kwargs.get("decorators", []),
        "signature": kwargs.get("signature", ""),
        "bases": kwargs.get("bases", []),
    }
    return SimpleNamespace(**payload)


def build_fake_repo(tmp_path: Path):
    chunks = []
    intra = {}
    for index in range(1, 31):
        module = f"pkg/module_{index}.py"
        file_path = tmp_path / module
        file_path.parent.mkdir(parents=True, exist_ok=True)
        source = f"""import os
from collections import Counter

class RepoClass{index}(BaseWorker):
    def method_{index}(self):
        try:
            return helper_{index}()
        except ValueError:
            return None

def helper_{index}():
    if {index} < 0:
        raise ValueError("bad")
    return "ok"
"""
        file_path.write_text(source, encoding="utf-8")
        chunks.append(
            make_chunk(
                module,
                "class",
                f"RepoClass{index}",
                4,
                9,
                """class RepoClass:
    def method(self):
        try:
            return helper()
        except ValueError:
            return None
""",
                bases=["BaseWorker"],
            )
        )
        chunks.append(
            make_chunk(
                module,
                "method",
                f"method_{index}",
                5,
                8,
                """def method(self):
    try:
        return helper()
    except ValueError:
        return None
""",
                qualified_name=f"RepoClass{index}.method_{index}",
                parent_class=f"RepoClass{index}",
            )
        )
        chunks.append(
            make_chunk(
                module,
                "function",
                f"helper_{index}",
                10,
                13,
                """def helper():
    if flag:
        raise ValueError("bad")
    return "ok"
""",
            )
        )
        intra[str(file_path.resolve())] = {
            "calls": [(f"RepoClass{index}.method_{index}", f"helper_{index}")],
            "inherits": [(f"RepoClass{index}", "BaseWorker")],
            "defines": [],
        }

    repo_map = "\n".join(["Repository Map:", "================="] + [f"[-] pkg/module_{i}.py" for i in range(1, 31)])
    benchmarks = [
        {
            "question": "What happens when the loader hits syntax errors?",
            "expected_entities": ["helper_1"],
            "expected_files": ["pkg/module_1.py"],
        },
        {
            "question": "How does the retriever combine dense and sparse results?",
            "expected_entities": ["RepoClass2"],
            "expected_files": ["pkg/module_2.py"],
        },
    ]
    return chunks, repo_map, FakeDepGraph(intra), benchmarks


def test_generate_dataset_from_inputs_produces_large_unique_grounded_dataset(tmp_path):
    chunks, repo_map, dep_graph, benchmarks = build_fake_repo(tmp_path)

    rows, report = generate_dataset_from_inputs(
        repo_root=tmp_path,
        chunks=chunks,
        repo_map=repo_map,
        dep_graph=dep_graph,
        benchmarks=benchmarks,
    )

    assert len(rows) >= 500
    assert not report["validation"]["errors"]

    questions = [row["messages"][1]["content"] for row in rows]
    assert len(questions) == len(set(questions))
    assert all(row["messages"][-1]["content"].strip() for row in rows)
    for row in rows:
        for filepath in row["metadata"]["expected_files"]:
            assert (tmp_path / filepath).exists()


def test_generated_dataset_audits_cleanly_for_sft(tmp_path):
    chunks, repo_map, dep_graph, benchmarks = build_fake_repo(tmp_path)
    rows, _report = generate_dataset_from_inputs(
        repo_root=tmp_path,
        chunks=chunks,
        repo_map=repo_map,
        dep_graph=dep_graph,
        benchmarks=benchmarks,
    )
    output = tmp_path / "sft_repo_qa.jsonl"
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    repo_files = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()}
    audit = audit_sft_dataset(output, repo_files)

    assert audit["invalid_rows"] == 0
    assert audit["duplicate_exact_rows"] == 0
    assert audit["duplicate_user_questions"] == 0
    assert audit["empty_assistant_answers"] == 0


def test_category_balance_stays_under_cap(tmp_path):
    chunks, repo_map, dep_graph, benchmarks = build_fake_repo(tmp_path)
    rows, report = generate_dataset_from_inputs(
        repo_root=tmp_path,
        chunks=chunks,
        repo_map=repo_map,
        dep_graph=dep_graph,
        benchmarks=benchmarks,
    )

    total = len(rows)
    for category, count in report["validation"]["category_counts"].items():
        assert (count / total) <= 0.35


def test_quick_profile_can_generate_smaller_valid_dataset(tmp_path):
    chunks, repo_map, dep_graph, benchmarks = build_fake_repo(tmp_path)
    rows, report = generate_dataset_from_inputs(
        repo_root=tmp_path,
        chunks=chunks,
        repo_map=repo_map,
        dep_graph=dep_graph,
        benchmarks=benchmarks,
        category_targets=QUICK_CATEGORY_TARGETS,
        min_total=50,
    )

    assert len(rows) >= 50
    assert not report["validation"]["errors"]
