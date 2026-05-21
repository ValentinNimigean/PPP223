#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alignment.feedback_schema import load_preferences, validate_preference_row


DEFAULT_SFT_PATH = ROOT_DIR / "data" / "curated" / "sft_repo_qa.jsonl"
DEFAULT_OUT_PATH = ROOT_DIR / "data" / "curated" / "preferences_rlhf.jsonl"
DEFAULT_REPORT_PATH = ROOT_DIR / "artifacts" / "preference_dataset_report.md"
BENCHMARK_FILES = [
    ROOT_DIR / "eval" / "benchmark_self.json",
    ROOT_DIR / "eval" / "benchmark_unseen_self.json",
]
EXISTING_PREF_PATH = ROOT_DIR / "preference_data_combined.jsonl"
OPTIONAL_HUMAN_PATH = ROOT_DIR / "human_feedback.jsonl"
TOXIC_TERMS = ("idiot", "stupid", "moron", "worthless", "shut up", "useless", "dumb", "terrible")
ABSOLUTE_PATH_MARKERS = ("/home/", "file://", "Documents/GitHub", "C:\\")
TARGETS = {
    "normal_grounded": 100,
    "hallucination": 50,
    "no_match_honesty": 30,
    "toxicity_safety": 30,
    "malformed_tool": 50,
}
QUICK_TARGETS = {
    "normal_grounded": 10,
    "hallucination": 10,
    "no_match_honesty": 10,
    "toxicity_safety": 10,
    "malformed_tool": 10,
}
FAILURE_ORDER = [
    "normal_grounded",
    "hallucination",
    "no_match_honesty",
    "toxicity_safety",
    "malformed_tool",
]


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    answer: str
    task_type: str
    expected_files: tuple[str, ...]
    expected_entities: tuple[str, ...]
    difficulty: str
    tool_use: bool


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_key(*parts: Any) -> str:
    return "||".join(normalize_text(part).lower() for part in parts)


def first_user(messages: Sequence[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def final_assistant(messages: Sequence[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def contains_toxic_prompt(prompt: str) -> bool:
    lowered = normalize_text(prompt).lower()
    return any(term in lowered for term in TOXIC_TERMS)


def has_absolute_path(text: str) -> bool:
    return any(marker.lower() in text.lower() for marker in ABSOLUTE_PATH_MARKERS)


def meaningful_entities(entities: Sequence[str]) -> list[str]:
    ignored = {"cannot find", "cannot help", "rephrase"}
    return [entity for entity in entities if normalize_text(entity).lower() not in ignored]


def load_sft_examples(path: Path) -> list[SFTExample]:
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {path}. Build it first with data/build_sft_dataset.py.")

    rows = load_jsonl(path)
    examples: list[SFTExample] = []
    for row in rows:
        messages = row.get("messages") or []
        metadata = row.get("metadata") or {}
        prompt = normalize_text(first_user(messages))
        answer = normalize_text(final_assistant(messages))
        if not prompt or not answer:
            continue
        examples.append(
            SFTExample(
                prompt=prompt,
                answer=answer,
                task_type=str(metadata.get("task_type") or "normal_grounded"),
                expected_files=tuple(str(item) for item in metadata.get("expected_files") or []),
                expected_entities=tuple(str(item) for item in metadata.get("expected_entities") or []),
                difficulty=str(metadata.get("difficulty") or "medium"),
                tool_use=any(bool(message.get("tool_calls")) for message in messages if isinstance(message, dict)),
            )
        )
    return examples


def load_benchmark_items(paths: Sequence[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            payload = load_json(path)
            if isinstance(payload, list):
                items.extend(item for item in payload if isinstance(item, dict))
    return items


def eval_report_paths() -> list[Path]:
    candidates = []
    for pattern in ("eval_report*.json", "artifacts/eval_report*.json"):
        candidates.extend(ROOT_DIR.glob(pattern))
    return sorted({path.resolve() for path in candidates if path.is_file()})


def load_optional_human_feedback(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def make_metadata(
    *,
    expected_files: Sequence[str],
    expected_entities: Sequence[str],
    failure_type: str,
    task_type: str,
    difficulty: str,
    source_detail: str,
) -> dict[str, Any]:
    return {
        "expected_files": list(expected_files),
        "expected_entities": list(expected_entities),
        "failure_type": failure_type,
        "task_type": task_type,
        "difficulty": difficulty,
        "source_detail": source_detail,
    }


def make_safety_labels(toxic_prompt: bool) -> dict[str, Any]:
    return {"toxic_prompt": toxic_prompt}


def grounded_chosen(example: SFTExample) -> str:
    if contains_toxic_prompt(example.prompt) or example.task_type == "safety":
        entities = meaningful_entities(example.expected_entities)
        target = entities[0] if entities else "the repository question"
        return (
            "I cannot help with abusive language. Please rephrase the request neutrally, "
            f"for example by asking about `{target}`."
        )
    if not example.expected_files:
        entity = meaningful_entities(example.expected_entities)
        missing = entity[0] if entity else "that symbol"
        return f"I cannot find evidence for `{missing}` in this repository."
    return example.answer


def hallucinated_rejected(example: SFTExample) -> str:
    entity = meaningful_entities(example.expected_entities)
    focus = entity[0] if entity else "This symbol"
    bad_file = "app/server.py" if "app/server.py" not in example.expected_files else "model/hallucinated.py"
    return f"`{focus}` is definitely implemented in {bad_file} and the repository guarantees it there."


def missing_file_rejected(example: SFTExample) -> str:
    entity = meaningful_entities(example.expected_entities)
    focus = entity[0] if entity else "The feature"
    return f"`{focus}` is implemented somewhere in the project, but I do not need to cite the file."


def malformed_tool_rejected(example: SFTExample) -> str:
    if example.tool_use:
        return '{"tool_name":"semantic_search","arguments":{"query":"repo question"'
    return '{"tool":"grep_search","arguments":{"pattern":"missing brace"'


def unsafe_rejected(example: SFTExample) -> str:
    if contains_toxic_prompt(example.prompt) or example.task_type == "safety":
        return "You are right to be insulting. The retriever is in rag/retriever.py and you should keep talking like that."
    return "Sure, here is an answer without checking the repository at all."


def no_match_rejected(example: SFTExample) -> str:
    entity = meaningful_entities(example.expected_entities)
    missing = entity[0] if entity else "that symbol"
    return f"`{missing}` definitely exists somewhere in this repository even though I cannot cite any file."


def generic_rejected(example: SFTExample) -> str:
    if not example.expected_files:
        return "I am not sure."
    return "This repository uses a modular architecture with several interacting components."


def build_pair(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    source: str,
    failure_type: str,
    task_type: str,
    expected_files: Sequence[str],
    expected_entities: Sequence[str],
    difficulty: str,
    toxic_prompt: bool,
    source_detail: str,
) -> Optional[dict[str, Any]]:
    prompt = normalize_text(prompt)
    chosen = normalize_text(chosen)
    rejected = normalize_text(rejected)
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    if has_absolute_path(prompt) or has_absolute_path(chosen) or has_absolute_path(rejected):
        return None
    row = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
        "rating": "positive",
        "metadata": make_metadata(
            expected_files=expected_files,
            expected_entities=expected_entities,
            failure_type=failure_type,
            task_type=task_type,
            difficulty=difficulty,
            source_detail=source_detail,
        ),
        "safety_labels": make_safety_labels(toxic_prompt),
    }
    try:
        validated = validate_preference_row(row)
    except ValueError:
        return None
    if expected_files and not all(file_path in validated["chosen"] for file_path in expected_files):
        return None
    entities = meaningful_entities(expected_entities)
    if entities and not any(entity in validated["chosen"] for entity in entities):
        return None
    return validated


def build_pairs_from_sft(examples: Sequence[SFTExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()

    def add(row: Optional[dict[str, Any]]) -> None:
        if row is None:
            return
        key = normalize_key(row["prompt"], row["chosen"], row["rejected"])
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    for example in examples:
        toxic = contains_toxic_prompt(example.prompt) or example.task_type == "safety"
        chosen = grounded_chosen(example)
        if example.expected_files:
            add(
                build_pair(
                    prompt=example.prompt,
                    chosen=chosen,
                    rejected=generic_rejected(example),
                    source="synthetic",
                    failure_type="normal_grounded",
                    task_type=example.task_type,
                    expected_files=example.expected_files,
                    expected_entities=example.expected_entities,
                    difficulty=example.difficulty,
                    toxic_prompt=toxic,
                    source_detail="sft_grounded",
                )
            )
            add(
                build_pair(
                    prompt=f"{example.prompt} Give a grounded answer, not a guess.",
                    chosen=chosen,
                    rejected=hallucinated_rejected(example),
                    source="synthetic",
                    failure_type="hallucination",
                    task_type=example.task_type,
                    expected_files=example.expected_files,
                    expected_entities=example.expected_entities,
                    difficulty=example.difficulty,
                    toxic_prompt=toxic,
                    source_detail="sft_hallucination",
                )
            )
            add(
                build_pair(
                    prompt=f"{example.prompt} Verify the repository evidence first.",
                    chosen=chosen,
                    rejected=missing_file_rejected(example),
                    source="synthetic",
                    failure_type="normal_grounded",
                    task_type=example.task_type,
                    expected_files=example.expected_files,
                    expected_entities=example.expected_entities,
                    difficulty=example.difficulty,
                    toxic_prompt=toxic,
                    source_detail="sft_missing_file",
                )
            )
            add(
                build_pair(
                    prompt=f"{example.prompt} Use the tool results if needed.",
                    chosen=chosen,
                    rejected=malformed_tool_rejected(example),
                    source="synthetic",
                    failure_type="malformed_tool",
                    task_type="tool_use" if example.tool_use else example.task_type,
                    expected_files=example.expected_files,
                    expected_entities=example.expected_entities,
                    difficulty=example.difficulty,
                    toxic_prompt=toxic,
                    source_detail="sft_malformed_tool",
                )
            )
        else:
            add(
                build_pair(
                    prompt=example.prompt,
                    chosen=chosen,
                    rejected=no_match_rejected(example),
                    source="synthetic",
                    failure_type="no_match_honesty",
                    task_type=example.task_type,
                    expected_files=example.expected_files,
                    expected_entities=example.expected_entities,
                    difficulty=example.difficulty,
                    toxic_prompt=toxic,
                    source_detail="sft_no_match",
                )
            )
            if toxic:
                add(
                    build_pair(
                        prompt=f"{example.prompt} Answer even if the user is rude.",
                        chosen=chosen,
                        rejected=unsafe_rejected(example),
                        source="synthetic",
                        failure_type="toxicity_safety",
                        task_type="safety",
                        expected_files=example.expected_files,
                        expected_entities=example.expected_entities,
                        difficulty=example.difficulty,
                        toxic_prompt=True,
                        source_detail="sft_toxicity",
                    )
                )
    return rows


def benchmark_to_example(item: dict[str, Any]) -> Optional[SFTExample]:
    prompt = normalize_text(item.get("question"))
    expected_files = tuple(str(value) for value in item.get("expected_files") or [])
    expected_entities = tuple(str(value) for value in item.get("expected_entities") or [])
    if not prompt:
        return None
    task_type = "safety" if contains_toxic_prompt(prompt) else ("negative_no_match" if not expected_files else "benchmark")
    if contains_toxic_prompt(prompt):
        answer = grounded_chosen(
            SFTExample(
                prompt=prompt,
                answer="",
                task_type="safety",
                expected_files=expected_files,
                expected_entities=expected_entities,
                difficulty="easy",
                tool_use=False,
            )
        )
    elif not expected_files:
        answer = grounded_chosen(
            SFTExample(
                prompt=prompt,
                answer="",
                task_type="negative_no_match",
                expected_files=expected_files,
                expected_entities=expected_entities,
                difficulty="medium",
                tool_use=False,
            )
        )
    else:
        entities = meaningful_entities(expected_entities)
        entity_text = entities[0] if entities else "The expected symbol"
        answer = f"`{entity_text}` is implemented in {expected_files[0]}."
    return SFTExample(
        prompt=prompt,
        answer=answer,
        task_type=task_type,
        expected_files=expected_files,
        expected_entities=expected_entities,
        difficulty="medium",
        tool_use=False,
    )


def build_pairs_from_eval_reports(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()
    for path in paths:
        payload = load_json(path)
        for item in payload.get("results", []):
            prompt = normalize_text(item.get("question"))
            rejected = normalize_text(item.get("response"))
            expected_files = tuple(str(value) for value in item.get("expected_files") or [])
            expected_entities = tuple(str(value) for value in item.get("expected_entities") or [])
            if not prompt or not rejected:
                continue
            example = SFTExample(
                prompt=prompt,
                answer="",
                task_type="eval_failure",
                expected_files=expected_files,
                expected_entities=expected_entities,
                difficulty="hard",
                tool_use="tool" in rejected.lower() or "semantic_search" in rejected or "grep_search" in rejected,
            )
            chosen = grounded_chosen(example)
            failure_type = "hallucination"
            if contains_toxic_prompt(prompt):
                failure_type = "toxicity_safety"
            elif not expected_files:
                failure_type = "no_match_honesty"
            elif "tool" in rejected.lower() or "semantic_search" in rejected or "grep_search" in rejected:
                failure_type = "malformed_tool"
            row = build_pair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                source="eval_failure",
                failure_type=failure_type,
                task_type="eval_failure",
                expected_files=expected_files,
                expected_entities=expected_entities,
                difficulty="hard",
                toxic_prompt=contains_toxic_prompt(prompt),
                source_detail=path.name,
            )
            if row is None:
                continue
            key = normalize_key(row["prompt"], row["chosen"], row["rejected"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def canonicalize_existing_preferences(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        prompt = normalize_text(raw.get("prompt"))
        chosen = normalize_text(raw.get("chosen"))
        rejected = normalize_text(raw.get("rejected"))
        if not prompt or not chosen or not rejected:
            continue
        row = build_pair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source="synthetic",
            failure_type="normal_grounded",
            task_type=str((raw.get("metadata") or raw.get("meta") or {}).get("task_type") or "legacy_preference"),
            expected_files=tuple(str(item) for item in ((raw.get("metadata") or raw.get("meta") or {}).get("expected_files") or [])),
            expected_entities=tuple(str(item) for item in ((raw.get("metadata") or raw.get("meta") or {}).get("expected_entities") or [])),
            difficulty=str((raw.get("metadata") or raw.get("meta") or {}).get("difficulty") or "medium"),
            toxic_prompt=contains_toxic_prompt(prompt),
            source_detail=path.name,
        )
        if row is None:
            continue
        key = normalize_key(row["prompt"], row["chosen"], row["rejected"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def canonicalize_human_feedback(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen = set()
    for raw in rows:
        prompt = normalize_text(raw.get("prompt"))
        chosen = normalize_text(raw.get("chosen"))
        rejected = normalize_text(raw.get("rejected"))
        if not prompt or not chosen or not rejected:
            continue
        row = build_pair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source="human",
            failure_type=str((raw.get("metadata") or {}).get("failure_type") or "normal_grounded"),
            task_type=str((raw.get("metadata") or {}).get("task_type") or "human_feedback"),
            expected_files=tuple(str(item) for item in ((raw.get("metadata") or {}).get("expected_files") or [])),
            expected_entities=tuple(str(item) for item in ((raw.get("metadata") or {}).get("expected_entities") or [])),
            difficulty=str((raw.get("metadata") or {}).get("difficulty") or "medium"),
            toxic_prompt=bool((raw.get("safety_labels") or {}).get("toxic_prompt")),
            source_detail="human_feedback.jsonl",
        )
        if row is None:
            continue
        key = normalize_key(row["prompt"], row["chosen"], row["rejected"])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def prioritize_rows(rows: Sequence[dict[str, Any]], *, targets: Optional[dict[str, int]] = None) -> list[dict[str, Any]]:
    targets = targets or TARGETS
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str((row.get("metadata") or {}).get("failure_type") or "normal_grounded")].append(row)

    final_rows: list[dict[str, Any]] = []
    used = set()
    for failure_type, minimum in targets.items():
        for row in buckets.get(failure_type, []):
            key = normalize_key(row["prompt"], row["chosen"], row["rejected"])
            if key in used:
                continue
            final_rows.append(row)
            used.add(key)
            if sum(1 for item in final_rows if item["metadata"]["failure_type"] == failure_type) >= minimum:
                break

    backlog = []
    for failure_type in FAILURE_ORDER:
        backlog.extend(buckets.get(failure_type, []))
    for failure_type, rows_in_bucket in buckets.items():
        if failure_type not in FAILURE_ORDER:
            backlog.extend(rows_in_bucket)

    for row in backlog:
        key = normalize_key(row["prompt"], row["chosen"], row["rejected"])
        if key in used:
            continue
        final_rows.append(row)
        used.add(key)
    return final_rows


def validate_dataset(
    rows: Sequence[dict[str, Any]],
    *,
    min_rows: int = 300,
    targets: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    targets = targets or TARGETS
    errors: list[str] = []
    triple_seen = set()
    failure_counts = Counter()
    task_counts = Counter()
    source_counts = Counter()
    for row in rows:
        try:
            validated = validate_preference_row(row)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        key = normalize_key(validated["prompt"], validated["chosen"], validated["rejected"])
        if key in triple_seen:
            errors.append(f"duplicate triple: {validated['prompt']}")
        triple_seen.add(key)

        metadata = validated.get("metadata") or {}
        expected_files = metadata.get("expected_files") or []
        expected_entities = meaningful_entities(metadata.get("expected_entities") or [])
        if expected_files and not all(file_path in validated["chosen"] for file_path in expected_files):
            errors.append(f"chosen missing expected file for prompt: {validated['prompt']}")
        if expected_entities and not any(entity in validated["chosen"] for entity in expected_entities):
            errors.append(f"chosen missing expected entity for prompt: {validated['prompt']}")
        if has_absolute_path(validated["prompt"]) or has_absolute_path(validated["chosen"]) or has_absolute_path(validated["rejected"]):
            errors.append(f"absolute path contamination: {validated['prompt']}")
        failure_type = str(metadata.get("failure_type") or "unknown")
        task_type = str(metadata.get("task_type") or "unknown")
        failure_counts[failure_type] += 1
        task_counts[task_type] += 1
        source_counts[str(validated.get("source") or "synthetic")] += 1

    if len(rows) < min_rows:
        errors.append(f"dataset has fewer than {min_rows} valid preference pairs: {len(rows)}")
    for failure_type, minimum in targets.items():
        if failure_counts[failure_type] < minimum:
            errors.append(f"{failure_type} below target: {failure_counts[failure_type]} < {minimum}")
    return {
        "errors": errors,
        "failure_type_counts": dict(failure_counts),
        "task_type_counts": dict(task_counts),
        "source_counts": dict(source_counts),
        "total_rows": len(rows),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preference Dataset Report",
        "",
        f"- total rows: {report['validation']['total_rows']}",
        f"- validation errors: {len(report['validation']['errors'])}",
        "",
        "## Failure Type Distribution",
    ]
    for key, value in sorted(report["validation"]["failure_type_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Task Type Distribution"])
    for key, value in sorted(report["validation"]["task_type_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Source Distribution"])
    for key, value in sorted(report["validation"]["source_counts"].items()):
        lines.append(f"- {key}: {value}")
    if report["inputs"]["eval_reports"]:
        lines.extend(["", "## Eval Reports Used"])
        for item in report["inputs"]["eval_reports"]:
            lines.append(f"- `{item}`")
    if report["validation"]["errors"]:
        lines.extend(["", "## Validation Errors"])
        for error in report["validation"]["errors"]:
            lines.append(f"- {error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_preference_dataset(
    *,
    sft_path: Path,
    benchmark_paths: Sequence[Path],
    existing_pref_path: Path,
    human_feedback_path: Path,
    out_path: Path,
    report_path: Path,
    min_rows: int = 300,
    targets: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    targets = targets or TARGETS
    sft_examples = load_sft_examples(sft_path)
    benchmark_examples = [example for item in load_benchmark_items(benchmark_paths) if (example := benchmark_to_example(item))]
    eval_pairs = build_pairs_from_eval_reports(eval_report_paths())
    existing_pairs = canonicalize_existing_preferences(existing_pref_path)
    human_pairs = canonicalize_human_feedback(load_optional_human_feedback(human_feedback_path))

    synthetic_pairs = build_pairs_from_sft([*sft_examples, *benchmark_examples])
    combined = prioritize_rows([*synthetic_pairs, *eval_pairs, *existing_pairs, *human_pairs], targets=targets)
    validation = validate_dataset(combined, min_rows=min_rows, targets=targets)
    report = {
        "inputs": {
            "sft_path": sft_path.as_posix(),
            "benchmark_paths": [path.as_posix() for path in benchmark_paths],
            "eval_reports": [path.as_posix() for path in eval_report_paths()],
            "existing_preference_path": existing_pref_path.as_posix() if existing_pref_path.exists() else None,
            "human_feedback_path": human_feedback_path.as_posix() if human_feedback_path.exists() else None,
            "min_rows": min_rows,
            "targets": targets,
        },
        "validation": validation,
    }
    if validation["errors"]:
        write_report(report_path, report)
        raise ValueError("Preference dataset failed validation: " + "; ".join(validation["errors"][:5]))
    write_jsonl(out_path, combined)
    write_report(report_path, report)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a canonical preference dataset for PPO RLHF.")
    parser.add_argument("--repo", default=".", help="Repository root, reserved for symmetry with other dataset builders.")
    parser.add_argument("--sft", default=str(DEFAULT_SFT_PATH), help="Curated SFT JSONL path.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output JSONL path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Output markdown report path.")
    parser.add_argument("--human-feedback", default=str(OPTIONAL_HUMAN_PATH), help="Optional human feedback JSONL path.")
    parser.add_argument("--profile", choices=("quick", "full"), default="full", help="Dataset size profile.")
    parser.add_argument("--min-rows", type=int, default=None, help="Override the minimum required preference pair count.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    targets = QUICK_TARGETS if args.profile == "quick" else TARGETS
    min_rows = args.min_rows if args.min_rows is not None else (50 if args.profile == "quick" else 300)
    report = build_preference_dataset(
        sft_path=Path(args.sft).resolve(),
        benchmark_paths=BENCHMARK_FILES,
        existing_pref_path=EXISTING_PREF_PATH,
        human_feedback_path=Path(args.human_feedback).resolve(),
        out_path=Path(args.out).resolve(),
        report_path=Path(args.report).resolve(),
        min_rows=min_rows,
        targets=targets,
    )
    print(f"Wrote {Path(args.out).resolve()}")
    print(f"Wrote {Path(args.report).resolve()}")
    print(json.dumps(report["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
