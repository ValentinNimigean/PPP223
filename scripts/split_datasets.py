#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_SFT_PATH = ROOT_DIR / "data" / "curated" / "sft_repo_qa.jsonl"
DEFAULT_PREF_PATH = ROOT_DIR / "data" / "curated" / "preferences_rlhf.jsonl"
DEFAULT_FINAL_DIR = ROOT_DIR / "data" / "final"
DEFAULT_REPORT = ROOT_DIR / "artifacts" / "dataset_split_report.md"
BENCHMARK_PATHS = [
    ROOT_DIR / "eval" / "benchmark_self.json",
    ROOT_DIR / "eval" / "benchmark_unseen_self.json",
]
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "holdout": 0.1}
DEFAULT_SEED = 3407
NEAR_DUP_THRESHOLD = 0.85


@dataclass
class RowGroup:
    prompt: str
    normalized_prompt: str
    prompt_hash: str
    task_type: str
    source: str
    rows: list[dict[str, Any]]
    benchmark_like: bool


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return rows


def normalize_prompt(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"`+", "", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def prompt_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def load_benchmark_prompts(paths: Sequence[Path]) -> list[str]:
    prompts: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        payload = load_json(path)
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict) and item.get("question"):
                prompts.append(normalize_prompt(item["question"]))
    return prompts


def sft_prompt(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def row_task_type(row: dict[str, Any], dataset_kind: str) -> str:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("task_type"):
        return str(metadata["task_type"])
    return "chat" if dataset_kind == "sft" else "preference"


def row_source(row: dict[str, Any], dataset_kind: str) -> str:
    if dataset_kind == "preference" and row.get("source"):
        return str(row["source"])
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict):
        if metadata.get("source"):
            return str(metadata["source"])
        if metadata.get("source_file"):
            return str(metadata["source_file"])
    return "unknown"


def is_benchmark_like(prompt_norm: str, benchmark_norms: Sequence[str]) -> bool:
    return any(prompt_norm == benchmark or prompt_similarity(prompt_norm, benchmark) >= NEAR_DUP_THRESHOLD for benchmark in benchmark_norms)


def build_groups(rows: Sequence[dict[str, Any]], dataset_kind: str, benchmark_norms: Sequence[str]) -> list[RowGroup]:
    grouped: dict[str, RowGroup] = {}
    for row in rows:
        prompt = sft_prompt(row) if dataset_kind == "sft" else str(row.get("prompt") or "")
        prompt = prompt.strip()
        if not prompt:
            continue
        normalized = normalize_prompt(prompt)
        digest = prompt_hash(normalized)
        if digest not in grouped:
            grouped[digest] = RowGroup(
                prompt=prompt,
                normalized_prompt=normalized,
                prompt_hash=digest,
                task_type=row_task_type(row, dataset_kind),
                source=row_source(row, dataset_kind),
                rows=[],
                benchmark_like=is_benchmark_like(normalized, benchmark_norms),
            )
        grouped[digest].rows.append(row)
    return list(grouped.values())


def target_count(total: int, split_name: str) -> int:
    if split_name == "train":
        return int(total * SPLIT_RATIOS["train"])
    if split_name == "val":
        return int(total * SPLIT_RATIOS["val"])
    return total - int(total * SPLIT_RATIOS["train"]) - int(total * SPLIT_RATIOS["val"])


def assign_groups(groups: Sequence[RowGroup], seed: int) -> dict[str, list[RowGroup]]:
    rng = random.Random(seed)
    by_task: dict[str, list[RowGroup]] = defaultdict(list)
    for group in groups:
        by_task[group.task_type].append(group)

    splits: dict[str, list[RowGroup]] = {"train": [], "val": [], "holdout": []}

    for task_type, task_groups in sorted(by_task.items()):
        benchmark_groups = [group for group in task_groups if group.benchmark_like]
        non_benchmark_groups = [group for group in task_groups if not group.benchmark_like]
        rng.shuffle(benchmark_groups)
        rng.shuffle(non_benchmark_groups)

        total = len(task_groups)
        val_target = target_count(total, "val")
        holdout_target = target_count(total, "holdout")

        val_groups = benchmark_groups[: min(len(benchmark_groups), val_target)]
        remaining_benchmark = benchmark_groups[len(val_groups) :]
        holdout_groups = list(remaining_benchmark[: min(len(remaining_benchmark), holdout_target)])
        leftover = non_benchmark_groups

        if len(remaining_benchmark) > len(holdout_groups):
            holdout_groups.extend(remaining_benchmark[len(holdout_groups) :])

        while len(val_groups) < val_target and leftover:
            val_groups.append(leftover.pop(0))
        while len(holdout_groups) < holdout_target and leftover:
            holdout_groups.append(leftover.pop(0))

        train_groups = leftover
        if any(group.benchmark_like for group in train_groups):
            benchmark_leaks = [group.prompt for group in train_groups if group.benchmark_like]
            raise ValueError(f"Benchmark-like prompts leaked into train for task `{task_type}`: {benchmark_leaks[:3]}")

        splits["train"].extend(train_groups)
        splits["val"].extend(val_groups)
        splits["holdout"].extend(holdout_groups)

    return splits


def augment_row(row: dict[str, Any], split_name: str, dataset_kind: str, group: RowGroup, seed: int) -> dict[str, Any]:
    cloned = json.loads(json.dumps(row))
    metadata = dict(cloned.get("metadata") or {})
    metadata["split"] = split_name
    metadata["prompt_hash"] = group.prompt_hash
    metadata["normalized_prompt"] = group.normalized_prompt
    metadata["split_seed"] = seed
    metadata["benchmark_like"] = group.benchmark_like
    cloned["metadata"] = metadata
    if dataset_kind == "preference" and "source" not in cloned:
        cloned["source"] = group.source
    return cloned


def materialize_split_rows(splits: dict[str, list[RowGroup]], dataset_kind: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    materialized: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "holdout": []}
    for split_name, groups in splits.items():
        for group in groups:
            for row in group.rows:
                materialized[split_name].append(augment_row(row, split_name, dataset_kind, group, seed))
    return materialized


def count_by_key(rows: Sequence[dict[str, Any]], key_fn) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[key_fn(row)] += 1
    return dict(sorted(counts.items()))


def prompt_set(rows: Sequence[dict[str, Any]], dataset_kind: str) -> set[str]:
    prompts = set()
    for row in rows:
        prompt = sft_prompt(row) if dataset_kind == "sft" else str(row.get("prompt") or "")
        prompts.add(normalize_prompt(prompt))
    return prompts


def validate_splits(materialized: dict[str, list[dict[str, Any]]], dataset_kind: str, benchmark_norms: Sequence[str]) -> list[str]:
    errors: list[str] = []
    split_prompts = {name: prompt_set(rows, dataset_kind) for name, rows in materialized.items()}
    names = list(split_prompts)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_prompts[left] & split_prompts[right]
            if overlap:
                errors.append(f"prompt overlap between {left} and {right}: {sorted(list(overlap))[:3]}")
    if split_prompts["train"] & set(benchmark_norms):
        errors.append("exact benchmark prompt leaked into train")
    for prompt in split_prompts["train"]:
        if is_benchmark_like(prompt, benchmark_norms):
            errors.append(f"benchmark-like prompt leaked into train: {prompt}")
            break
    return errors


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Split Report",
        "",
        f"- seed: {report['seed']}",
        f"- benchmark prompts: {report['benchmark_prompt_count']}",
        "",
    ]
    for dataset_name in ("sft", "preferences"):
        dataset_report = report[dataset_name]
        lines.append(f"## {dataset_name.upper()}")
        for split_name in ("train", "val", "holdout"):
            split = dataset_report["splits"][split_name]
            lines.append(f"- {split_name}: rows={split['rows']} groups={split['groups']}")
        lines.append("- task_type counts:")
        for split_name in ("train", "val", "holdout"):
            for key, value in dataset_report["task_type_counts"][split_name].items():
                lines.append(f"  - {split_name}:{key}={value}")
        lines.append("- source counts:")
        for split_name in ("train", "val", "holdout"):
            for key, value in dataset_report["source_counts"][split_name].items():
                lines.append(f"  - {split_name}:{key}={value}")
        lines.append("")
    if report["errors"]:
        lines.append("## Errors")
        for error in report["errors"]:
            lines.append(f"- {error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_dataset(rows: Sequence[dict[str, Any]], dataset_kind: str, benchmark_norms: Sequence[str], seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups = build_groups(rows, dataset_kind, benchmark_norms)
    split_groups = assign_groups(groups, seed)
    materialized = materialize_split_rows(split_groups, dataset_kind, seed)
    report = {
        "splits": {
            name: {"rows": len(materialized[name]), "groups": len(split_groups[name])}
            for name in ("train", "val", "holdout")
        },
        "task_type_counts": {
            name: count_by_key(materialized[name], lambda row: row_task_type(row, dataset_kind))
            for name in ("train", "val", "holdout")
        },
        "source_counts": {
            name: count_by_key(materialized[name], lambda row: row_source(row, dataset_kind))
            for name in ("train", "val", "holdout")
        },
    }
    return materialized, report


def run_split(
    *,
    sft_path: Path,
    pref_path: Path,
    benchmark_paths: Sequence[Path],
    final_dir: Path,
    report_path: Path,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if not sft_path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {sft_path}")
    if not pref_path.exists():
        raise FileNotFoundError(f"Preference dataset not found: {pref_path}")

    benchmark_norms = load_benchmark_prompts(benchmark_paths)
    sft_rows = load_jsonl(sft_path)
    pref_rows = load_jsonl(pref_path)

    sft_splits, sft_report = split_dataset(sft_rows, "sft", benchmark_norms, seed)
    pref_splits, pref_report = split_dataset(pref_rows, "preference", benchmark_norms, seed)

    outputs = {
        "sft_train": final_dir / "sft_train.jsonl",
        "sft_val": final_dir / "sft_val.jsonl",
        "sft_test_holdout": final_dir / "sft_test_holdout.jsonl",
        "preferences_train": final_dir / "preferences_train.jsonl",
        "preferences_val": final_dir / "preferences_val.jsonl",
        "preferences_test_holdout": final_dir / "preferences_test_holdout.jsonl",
    }

    write_jsonl(outputs["sft_train"], sft_splits["train"])
    write_jsonl(outputs["sft_val"], sft_splits["val"])
    write_jsonl(outputs["sft_test_holdout"], sft_splits["holdout"])
    write_jsonl(outputs["preferences_train"], pref_splits["train"])
    write_jsonl(outputs["preferences_val"], pref_splits["val"])
    write_jsonl(outputs["preferences_test_holdout"], pref_splits["holdout"])

    errors = [
        *validate_splits(sft_splits, "sft", benchmark_norms),
        *validate_splits(pref_splits, "preference", benchmark_norms),
    ]

    report = {
        "seed": seed,
        "benchmark_prompt_count": len(benchmark_norms),
        "sft": sft_report,
        "preferences": pref_report,
        "errors": errors,
        "outputs": {name: path.as_posix() for name, path in outputs.items()},
    }
    write_report(report_path, report)
    if errors:
        raise ValueError("Dataset split validation failed: " + "; ".join(errors[:5]))
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split curated SFT and RLHF datasets deterministically.")
    parser.add_argument("--sft", default=str(DEFAULT_SFT_PATH), help="Curated SFT dataset path.")
    parser.add_argument("--preferences", default=str(DEFAULT_PREF_PATH), help="Curated preference dataset path.")
    parser.add_argument("--final-dir", default=str(DEFAULT_FINAL_DIR), help="Output directory for final split datasets.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Markdown split report path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic split seed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_split(
        sft_path=Path(args.sft).resolve(),
        pref_path=Path(args.preferences).resolve(),
        benchmark_paths=BENCHMARK_PATHS,
        final_dir=Path(args.final_dir).resolve(),
        report_path=Path(args.report).resolve(),
        seed=args.seed,
    )
    print(f"Wrote {report['outputs']['sft_train']}")
    print(f"Wrote {report['outputs']['preferences_train']}")
    print(f"Wrote {Path(args.report).resolve()}")
    print(json.dumps({"sft": report["sft"]["splits"], "preferences": report["preferences"]["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
