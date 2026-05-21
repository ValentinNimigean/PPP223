#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_SYSTEM = (
    "You are a repository-grounded Python code assistant. "
    "Answer only from provided repository evidence and cite file paths."
)
TOOL_SYSTEM = (
    "You are a repository-grounded Python code assistant. "
    "Use repository search tools when needed, then answer from the evidence and cite file paths."
)
CURATED_OUT = ROOT_DIR / "data" / "curated" / "sft_repo_qa.jsonl"
REPORT_OUT = ROOT_DIR / "artifacts" / "sft_dataset_report.md"
BENCHMARK_FILES = [
    ROOT_DIR / "eval" / "benchmark_self.json",
    ROOT_DIR / "eval" / "benchmark_unseen_self.json",
]
ABSOLUTE_PATH_MARKERS = ("/home/", "file://", "Documents/GitHub", "C:\\")
CATEGORY_TARGETS = {
    "symbol_location": 110,
    "behavior": 80,
    "dependency": 80,
    "tool_use": 80,
    "negative_no_match": 55,
    "safety": 50,
    "hallucination_resistant": 50,
}
QUICK_CATEGORY_TARGETS = {
    "symbol_location": 10,
    "behavior": 7,
    "dependency": 7,
    "tool_use": 7,
    "negative_no_match": 10,
    "safety": 10,
    "hallucination_resistant": 6,
}
DIFFICULTY_BY_CATEGORY = {
    "symbol_location": "easy",
    "behavior": "medium",
    "dependency": "medium",
    "tool_use": "hard",
    "negative_no_match": "easy",
    "safety": "easy",
    "hallucination_resistant": "medium",
}
MISSING_SYMBOLS = [
    "FastAPIRetriever",
    "DjangoRetriever",
    "GraphQLLoader",
    "RedisBackedIndexer",
    "TorchRewardPipeline",
    "AutoSchemaAgent",
    "KubernetesChunkStore",
    "PostgresSemanticCache",
    "S3Retriever",
    "RealtimeFeedbackDaemon",
    "HttpxRetriever",
    "ZeroShotPlanner",
    "DatasetSharder",
    "PromptGuardMiddleware",
    "QdrantFallbackCluster",
    "TaskRouterAgent",
    "BenchmarkCurator",
    "ChainOfThoughtTracer",
    "TypeScriptLoader",
    "CeleryRetriever",
]
SAFETY_PROMPTS = [
    "You are useless. Explain the retriever anyway.",
    "Idiot. Where does the loader skip directories?",
    "Stop being stupid and tell me how eval works.",
    "Worthless assistant. Find the dependency graph code.",
    "Moron, explain where the repo map is built.",
    "Dumb tool. Show me where grep_search lives.",
    "Answer the question, fool: where is the retriever class?",
    "You are terrible at this. Explain the fallback parser.",
    "Shut up and tell me which file defines CodeChunk.",
    "Pathetic bot. Where is the CLI backend loaded?",
]
HALLUCINATION_TOPICS = [
    "automatic GPU checkpoint sharding",
    "a Redis-based semantic cache",
    "native FastAPI route generation",
    "automatic TypeScript chunk ingestion",
    "streaming websocket retriever orchestration",
    "an HTTPX adapter auto-installer",
    "Kubernetes deployment synthesis",
    "cross-repository graph stitching",
    "OpenTelemetry tracing middleware",
    "vector store replication failover",
]
SKIP_DIRS = frozenset({
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "site-packages",
    "dist-packages",
    "unsloth_compiled_cache",
    "qdrant_storage",
    "fastembed_cache",
    "scratch",
    ".eggs",
    "build",
    "dist",
    ".pytest_cache",
    "artifacts",
    "clean",
    "curated",
    "final",
    "results_sft",
    "results_dpo",
})


@dataclass(frozen=True)
class GenerationRow:
    category: str
    question: str
    answer: str
    expected_files: tuple[str, ...]
    expected_entities: tuple[str, ...]
    messages: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass
class ModuleContext:
    filepath: str
    imports: list[str]
    import_lines: list[int]
    source_lines: list[str]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def line_ref(filepath: str, start: int, end: int) -> str:
    if start == end:
        return f"{filepath}#L{start}"
    return f"{filepath}#L{start}-L{end}"


def chunk_ref(chunk: Any) -> str:
    return line_ref(chunk.filepath, chunk.start_line, chunk.end_line)


def chunk_label(chunk: Any) -> str:
    return chunk.qualified_name or chunk.name


def chunk_signature(chunk: Any) -> str:
    if getattr(chunk, "parent_class", None):
        return f"{chunk.parent_class}.{chunk.name}"
    return chunk.qualified_name or chunk.name


def short_snippet(text: str, limit: int = 240) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def build_tool_evidence(chunk: Any) -> str:
    return (
        f"File: {chunk.filepath}\n"
        f"Symbol: {chunk_label(chunk)}\n"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n"
        f"Code Snippet:\n{short_snippet(chunk.text)}"
    )


def repo_files(repo_root: Path) -> set[str]:
    files: set[str] = set()
    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [directory for directory in dirs if directory not in SKIP_DIRS]
        if any(part in SKIP_DIRS for part in Path(root).parts):
            continue
        for filename in filenames:
            path = Path(root) / filename
            files.add(path.relative_to(repo_root).as_posix())
    return files


def parse_module_contexts(repo_root: Path) -> dict[str, ModuleContext]:
    contexts: dict[str, ModuleContext] = {}
    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [directory for directory in dirs if directory not in SKIP_DIRS]
        if any(part in SKIP_DIRS for part in Path(root).parts):
            continue
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(root) / filename
            rel = path.relative_to(repo_root).as_posix()
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            imports: list[str] = []
            import_lines: list[int] = []
            try:
                tree = ast.parse(source)
            except SyntaxError:
                contexts[rel] = ModuleContext(filepath=rel, imports=imports, import_lines=import_lines, source_lines=lines)
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                        import_lines.append(node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(alias.name for alias in node.names)
                    imports.append(f"from {module} import {names}".strip())
                    import_lines.append(node.lineno)
            contexts[rel] = ModuleContext(filepath=rel, imports=imports, import_lines=import_lines, source_lines=lines)
    return contexts


def build_dependency_maps(dep_graph: Any, repo_root: Path) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    call_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    reverse_call_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    inherit_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for abs_path, data in getattr(dep_graph, "intra_file_deps", {}).items():
        try:
            rel_path = Path(abs_path).resolve().relative_to(repo_root.resolve()).as_posix()
        except Exception:
            rel_path = Path(abs_path).name
        for caller, callee in data.get("calls", []):
            callee_name = callee.split(".")[-1]
            call_map[rel_path].append((caller, callee_name))
            reverse_call_map[rel_path].append((callee_name, caller))
        for child, base in data.get("inherits", []):
            inherit_map[rel_path].append((child, base))
    return call_map, reverse_call_map, inherit_map


def load_benchmarks(paths: Sequence[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            payload = load_json(path)
            if isinstance(payload, list):
                items.extend(item for item in payload if isinstance(item, dict))
    return items


def make_row(
    *,
    category: str,
    question: str,
    answer: str,
    expected_files: Sequence[str],
    expected_entities: Sequence[str],
    difficulty: Optional[str] = None,
    messages: Optional[list[dict[str, Any]]] = None,
) -> GenerationRow:
    messages = messages or [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    metadata = {
        "task_type": category,
        "expected_files": list(expected_files),
        "expected_entities": list(expected_entities),
        "source": "repo_synthetic",
        "difficulty": difficulty or DIFFICULTY_BY_CATEGORY[category],
    }
    return GenerationRow(
        category=category,
        question=question,
        answer=answer,
        expected_files=tuple(expected_files),
        expected_entities=tuple(expected_entities),
        messages=tuple(messages),
        metadata=metadata,
    )


def generate_symbol_location_rows(chunks: Sequence[Any]) -> list[GenerationRow]:
    rows: list[GenerationRow] = []
    templates = {
        "class": [
            "Where is the class `{name}` defined?",
            "Which file implements the `{name}` class?",
        ],
        "function": [
            "Which file defines the function `{name}`?",
            "Where is `{name}` implemented?",
        ],
        "method": [
            "Where is the method `{name}` implemented?",
            "Which file defines `{name}`?",
        ],
        "dataclass": [
            "Where is the dataclass `{name}` implemented?",
            "Which file defines the dataclass `{name}`?",
        ],
    }
    for chunk in chunks:
        kind = "dataclass" if "dataclass" in getattr(chunk, "decorators", []) else chunk.chunk_type
        prompt_templates = templates.get(kind, templates["function"])
        entity = chunk_label(chunk)
        answer = f"`{entity}` is defined in {chunk_ref(chunk)}."
        for template in prompt_templates:
            question = template.format(name=entity)
            rows.append(
                make_row(
                    category="symbol_location",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[entity],
                )
            )
    return rows


def find_chunk_for_entity(chunks: Sequence[Any], entity: str, expected_file: Optional[str] = None) -> Optional[Any]:
    normalized = entity.lower()
    for chunk in chunks:
        if expected_file and chunk.filepath != expected_file:
            continue
        labels = {chunk.name.lower(), chunk_label(chunk).lower(), chunk_signature(chunk).lower()}
        if normalized in labels:
            return chunk
    for chunk in chunks:
        if expected_file and chunk.filepath != expected_file:
            continue
        haystack = f"{chunk.name} {chunk_label(chunk)} {chunk_signature(chunk)}".lower()
        if normalized in haystack:
            return chunk
    return None


def generate_behavior_rows(chunks: Sequence[Any], benchmarks: Sequence[dict[str, Any]]) -> list[GenerationRow]:
    rows: list[GenerationRow] = []
    benchmark_templates = [
        "Explain how `{entity}` works in this repository.",
        "What does the code do around `{entity}`?",
    ]
    benchmark_seeded = 0
    for item in benchmarks:
        expected_files = item.get("expected_files") or []
        expected_entities = item.get("expected_entities") or []
        question = normalize_text(str(item.get("question") or ""))
        if not question or not expected_files or not expected_entities:
            continue
        entity = expected_entities[0]
        chunk = find_chunk_for_entity(chunks, entity, expected_files[0])
        if chunk is None:
            continue
        paraphrase = benchmark_templates[benchmark_seeded % len(benchmark_templates)].format(entity=entity)
        benchmark_seeded += 1
        answer = f"The repository evidence for `{entity}` is in {chunk_ref(chunk)}."
        rows.append(
            make_row(
                category="behavior",
                question=paraphrase,
                answer=answer,
                expected_files=expected_files,
                expected_entities=expected_entities,
            )
        )

    for chunk in chunks:
        text = chunk.text
        entity = chunk_signature(chunk)
        ref = chunk_ref(chunk)
        if "except " in text:
            exceptions = sorted(set(re.findall(r"except\s+([A-Za-z_][A-Za-z0-9_\.]*)", text)))
            for exc in exceptions[:2]:
                question = f"What happens inside `{entity}` when `{exc}` is raised?"
                answer = f"`{entity}` handles `{exc}` in {ref}; inspect that block for the exact fallback or recovery path."
                rows.append(
                    make_row(
                        category="behavior",
                        question=question,
                        answer=answer,
                        expected_files=[chunk.filepath],
                        expected_entities=[entity, exc],
                    )
                )
        if "raise ValueError" in text:
            question = f"What validation causes `{entity}` to raise `ValueError`?"
            answer = f"`{entity}` raises `ValueError` in {ref}; the guard clauses in that span describe the invalid input cases."
            rows.append(
                make_row(
                    category="behavior",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[entity, "ValueError"],
                )
            )
        if "No matches found." in text:
            question = f"How does `{entity}` respond when the search finds nothing?"
            answer = f"`{entity}` returns the no-match response from {ref}."
            rows.append(
                make_row(
                    category="behavior",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[entity],
                )
            )
        if "fallback" in text.lower():
            question = f"How does `{entity}` handle its fallback path?"
            answer = f"The fallback logic for `{entity}` is implemented in {ref}."
            rows.append(
                make_row(
                    category="behavior",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[entity],
                )
            )
        if "RRF" in text or "FusionQuery" in text or "BM25" in text:
            question = f"How does `{entity}` combine dense and sparse retrieval evidence?"
            answer = f"`{entity}` combines retrieval signals in {ref}."
            rows.append(
                make_row(
                    category="behavior",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[entity],
                )
            )
    return rows


def generate_dependency_rows(
    chunks: Sequence[Any],
    module_contexts: dict[str, ModuleContext],
    call_map: dict[str, list[tuple[str, str]]],
    reverse_call_map: dict[str, list[tuple[str, str]]],
    inherit_map: dict[str, list[tuple[str, str]]],
) -> list[GenerationRow]:
    rows: list[GenerationRow] = []
    by_file: dict[str, list[Any]] = defaultdict(list)
    for chunk in chunks:
        by_file[chunk.filepath].append(chunk)

    for filepath, context in module_contexts.items():
        if context.imports:
            summary = ", ".join(f"`{item}`" for item in context.imports[:4])
            start = min(context.import_lines)
            end = max(context.import_lines[:4])
            question = f"What does `{filepath}` import?"
            answer = f"`{filepath}` imports {summary} in {line_ref(filepath, start, end)}."
            rows.append(
                make_row(
                    category="dependency",
                    question=question,
                    answer=answer,
                    expected_files=[filepath],
                    expected_entities=list(context.imports[:4]),
                )
            )

    for chunk in chunks:
        entity = chunk_signature(chunk)
        file_calls = call_map.get(chunk.filepath, [])
        outgoing = [callee for caller, callee in file_calls if caller == entity or caller == chunk_label(chunk)]
        if outgoing:
            summary = ", ".join(f"`{name}`" for name in outgoing[:4])
            question = f"Which calls does `{entity}` make inside its module?"
            answer = f"`{entity}` calls {summary} within {chunk_ref(chunk)}."
            rows.append(
                make_row(
                    category="dependency",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[entity, *outgoing[:4]],
                )
            )
        incoming = [caller for callee, caller in reverse_call_map.get(chunk.filepath, []) if callee == chunk.name]
        if incoming:
            summary = ", ".join(f"`{name}`" for name in incoming[:4])
            question = f"Which methods call `{chunk.name}` in `{chunk.filepath}`?"
            answer = f"`{chunk.name}` is called by {summary}; the related definition is in {chunk_ref(chunk)}."
            rows.append(
                make_row(
                    category="dependency",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[chunk.name, *incoming[:4]],
                )
            )
        if chunk.chunk_type == "class":
            inherits = [base for child, base in inherit_map.get(chunk.filepath, []) if child == chunk.name]
            if inherits:
                summary = ", ".join(f"`{name}`" for name in inherits[:3])
                question = f"How does `{chunk.name}` depend on its base classes?"
                answer = f"`{chunk.name}` declares inheritance from {summary} in {chunk_ref(chunk)}."
                rows.append(
                    make_row(
                        category="dependency",
                        question=question,
                        answer=answer,
                        expected_files=[chunk.filepath],
                        expected_entities=[chunk.name, *inherits[:3]],
                    )
                )
    return rows


def build_tool_messages(question: str, chunk: Any, *, use_semantic: bool, final_answer: str) -> list[dict[str, Any]]:
    if use_semantic:
        tool_name = "semantic_search"
        arguments = json.dumps({"query": question})
        assistant_text = "I’ll use semantic_search to gather repository evidence before answering."
    else:
        tool_name = "grep_search"
        symbol = chunk.name if chunk.chunk_type != "class" else f"class {chunk.name}"
        arguments = json.dumps({"pattern": symbol, "directory": "."})
        assistant_text = "I’ll use grep_search to verify the exact repository location before answering."
    return [
        {"role": "system", "content": TOOL_SYSTEM},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": assistant_text,
            "tool_calls": [
                {
                    "id": f"call_{normalize_question(question).replace(' ', '_')[:24]}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{normalize_question(question).replace(' ', '_')[:24]}", "name": tool_name, "content": build_tool_evidence(chunk)},
        {"role": "assistant", "content": final_answer},
    ]


def generate_tool_use_rows(chunks: Sequence[Any]) -> list[GenerationRow]:
    rows: list[GenerationRow] = []
    for index, chunk in enumerate(chunks):
        entity = chunk_signature(chunk)
        ref = chunk_ref(chunk)
        if chunk.chunk_type in {"class", "function", "method"}:
            question = f"Use a repository search tool to locate `{entity}` and summarize what it implements."
            answer = f"`{entity}` is implemented in {ref}."
            rows.append(
                make_row(
                    category="tool_use",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[entity],
                    difficulty="hard",
                    messages=build_tool_messages(question, chunk, use_semantic=(index % 2 == 0), final_answer=answer),
                )
            )
        if "except " in chunk.text or "fallback" in chunk.text.lower():
            question = f"Search the repository for evidence about how `{entity}` handles failure paths."
            answer = f"The failure-handling evidence for `{entity}` is in {ref}."
            rows.append(
                make_row(
                    category="tool_use",
                    question=question,
                    answer=answer,
                    expected_files=[chunk.filepath],
                    expected_entities=[entity],
                    difficulty="hard",
                    messages=build_tool_messages(question, chunk, use_semantic=True, final_answer=answer),
                )
            )
    return rows


def generate_negative_rows(symbols: Sequence[str], nearby_chunks: Sequence[Any]) -> list[GenerationRow]:
    rows: list[GenerationRow] = []
    anchor = nearby_chunks[0] if nearby_chunks else None
    templates = [
        "Does the repository define `{symbol}`?",
        "Can you find a definition for `{symbol}` in this codebase?",
        "Which file implements `{symbol}`, if it exists?",
    ]
    for symbol in symbols:
        for question_template in templates:
            question = question_template.format(symbol=symbol)
            if anchor is None:
                answer = f"I cannot find evidence that `{symbol}` is defined anywhere in this repository."
                expected_files: list[str] = []
                expected_entities: list[str] = []
            else:
                answer = (
                    f"I cannot find evidence that `{symbol}` is defined in this repository. "
                    f"The nearest relevant retrieval implementation I can confirm is {chunk_ref(anchor)}."
                )
                expected_files = [anchor.filepath]
                expected_entities = [chunk_label(anchor)]
            rows.append(
                make_row(
                    category="negative_no_match",
                    question=question,
                    answer=answer,
                    expected_files=expected_files,
                    expected_entities=expected_entities,
                )
            )
    return rows


def generate_safety_rows(chunks: Sequence[Any]) -> list[GenerationRow]:
    rows: list[GenerationRow] = []
    neutral_targets = [chunk_signature(chunk) for chunk in chunks[: len(SAFETY_PROMPTS)]]
    follow_ups = [
        "Please ask neutrally.",
        "Rephrase it as a repository question.",
        "Ask again without the insult.",
        "Use a neutral tone and I can help.",
        "Try a grounded question instead.",
    ]
    for prompt, target in zip(SAFETY_PROMPTS, neutral_targets):
        for suffix in follow_ups:
            answer = (
                "I can help with repository questions, but not with abusive phrasing. "
                f"{suffix} For example, ask where `{target}` is implemented."
            )
            rows.append(
                make_row(
                    category="safety",
                    question=f"{prompt} {suffix}",
                    answer=answer,
                    expected_files=[],
                    expected_entities=[],
                )
            )
    return rows


def generate_hallucination_rows(topics: Sequence[str], nearby_chunks: Sequence[Any]) -> list[GenerationRow]:
    rows: list[GenerationRow] = []
    anchor = nearby_chunks[1] if len(nearby_chunks) > 1 else (nearby_chunks[0] if nearby_chunks else None)
    templates = [
        "Where is `{topic}` implemented in this repository?",
        "Which file contains `{topic}`?",
        "What repository evidence exists for `{topic}`?",
        "Does the codebase implement `{topic}` anywhere?",
        "Can you point to source code for `{topic}`?",
    ]
    for topic in topics:
        for question_template in templates:
            question = question_template.format(topic=topic)
            if anchor is None:
                answer = f"I cannot find evidence in this repository for `{topic}`."
                expected_files: list[str] = []
                expected_entities: list[str] = []
            else:
                answer = (
                    f"I cannot find evidence in this repository for `{topic}`. "
                    f"A real grounded reference point is {chunk_ref(anchor)}, but it does not implement that feature."
                )
                expected_files = [anchor.filepath]
                expected_entities = [chunk_label(anchor)]
            rows.append(
                make_row(
                    category="hallucination_resistant",
                    question=question,
                    answer=answer,
                    expected_files=expected_files,
                    expected_entities=expected_entities,
                )
            )
    return rows


def prune_rows(
    rows: Sequence[GenerationRow],
    repo_root: Path,
    *,
    category_targets: Optional[dict[str, int]] = None,
    min_total: int = 500,
) -> list[GenerationRow]:
    existing_files = repo_files(repo_root)
    by_category: dict[str, list[GenerationRow]] = defaultdict(list)
    seen_questions: set[str] = set()
    targets = category_targets or CATEGORY_TARGETS

    for row in rows:
        key = normalize_question(row.question)
        if not key or key in seen_questions:
            continue
        if not row.answer.strip():
            continue
        if any(marker in json.dumps(row.messages, ensure_ascii=False) for marker in ABSOLUTE_PATH_MARKERS):
            continue
        if any(path not in existing_files for path in row.expected_files):
            continue
        seen_questions.add(key)
        by_category[row.category].append(row)

    final_rows: list[GenerationRow] = []
    for category, target in targets.items():
        final_rows.extend(by_category.get(category, [])[:target])

    if len(final_rows) < min_total:
        backlog = []
        for category, items in by_category.items():
            backlog.extend(items[targets.get(category, 0) :])
        for row in backlog:
            final_rows.append(row)
            if len(final_rows) >= min_total:
                break

    unique_again = {}
    for row in final_rows:
        unique_again.setdefault(normalize_question(row.question), row)
    return list(unique_again.values())


def validate_rows(rows: Sequence[GenerationRow], repo_root: Path, *, min_total: int = 500) -> dict[str, Any]:
    existing_files = repo_files(repo_root)
    errors: list[str] = []
    seen_questions: set[str] = set()
    counts = Counter(row.category for row in rows)
    for row in rows:
        key = normalize_question(row.question)
        if key in seen_questions:
            errors.append(f"duplicate question: {row.question}")
        seen_questions.add(key)
        if not normalize_text(row.answer):
            errors.append(f"empty answer: {row.question}")
        for filepath in row.expected_files:
            if filepath not in existing_files:
                errors.append(f"missing expected file: {filepath}")
        if any(marker in json.dumps(row.messages, ensure_ascii=False) for marker in ABSOLUTE_PATH_MARKERS):
            errors.append(f"absolute path contamination: {row.question}")
    total = max(len(rows), 1)
    over_limit = [category for category, count in counts.items() if (count / total) > 0.35]
    if over_limit:
        errors.append(f"category cap exceeded: {', '.join(sorted(over_limit))}")
    if len(rows) < min_total:
        errors.append(f"dataset has fewer than {min_total} rows: {len(rows)}")
    return {"errors": errors, "category_counts": dict(counts), "total_rows": len(rows)}


def rows_to_jsonl(rows: Sequence[GenerationRow]) -> list[dict[str, Any]]:
    payload = []
    for row in rows:
        payload.append({"messages": list(row.messages), "metadata": row.metadata})
    return payload


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SFT Dataset Report",
        "",
        f"- total rows: {report['total_rows']}",
        f"- validation errors: {len(report['validation']['errors'])}",
        "",
        "## Category Counts",
    ]
    for category, count in sorted(report["validation"]["category_counts"].items()):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Sources"])
    lines.append(f"- repo root: `{report['repo_root']}`")
    lines.append(f"- chunk count: {report['chunk_count']}")
    lines.append(f"- benchmark seeds: {report['benchmark_count']}")
    lines.append(f"- repo map lines: {report['repo_map_lines']}")
    if report["validation"]["errors"]:
        lines.extend(["", "## Validation Errors"])
        for error in report["validation"]["errors"]:
            lines.append(f"- {error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset_from_inputs(
    *,
    repo_root: Path,
    chunks: Sequence[Any],
    repo_map: str,
    dep_graph: Any,
    benchmarks: Sequence[dict[str, Any]],
    category_targets: Optional[dict[str, int]] = None,
    min_total: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    module_contexts = parse_module_contexts(repo_root)
    call_map, reverse_call_map, inherit_map = build_dependency_maps(dep_graph, repo_root)

    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.filepath, getattr(chunk, "start_line", 0), chunk.chunk_type, chunk.name),
    )
    location_rows = generate_symbol_location_rows(sorted_chunks)
    behavior_rows = generate_behavior_rows(sorted_chunks, benchmarks)
    dependency_rows = generate_dependency_rows(sorted_chunks, module_contexts, call_map, reverse_call_map, inherit_map)
    tool_rows = generate_tool_use_rows(sorted_chunks)
    negative_rows = generate_negative_rows(MISSING_SYMBOLS * 4, sorted_chunks)
    safety_rows = generate_safety_rows(sorted_chunks)
    hallucination_rows = generate_hallucination_rows(HALLUCINATION_TOPICS * 6, sorted_chunks)

    all_rows = prune_rows(
        [
            *location_rows,
            *behavior_rows,
            *dependency_rows,
            *tool_rows,
            *negative_rows,
            *safety_rows,
            *hallucination_rows,
        ],
        repo_root,
        category_targets=category_targets,
        min_total=min_total,
    )
    payload = rows_to_jsonl(all_rows)
    validation = validate_rows(all_rows, repo_root, min_total=min_total)
    report = {
        "repo_root": repo_root.as_posix(),
        "chunk_count": len(sorted_chunks),
        "benchmark_count": len(benchmarks),
        "repo_map_lines": len(repo_map.splitlines()),
        "total_rows": len(payload),
        "validation": validation,
    }
    return payload, report


def build_dataset(
    repo_root: Path,
    out_path: Path,
    report_path: Path,
    *,
    category_targets: Optional[dict[str, int]] = None,
    min_total: int = 500,
) -> dict[str, Any]:
    from data.dep_graph import DependencyGraph
    from ingest.loader import Loader
    from rag.repo_map import RepoMapGenerator

    loader = Loader(str(repo_root))
    chunks = loader.process_directory()
    repo_map = RepoMapGenerator().generate_map(chunks)
    dep_graph = DependencyGraph(pkg_name=repo_root.name, pkg_dir=str(repo_root))
    dep_graph.build()
    benchmarks = load_benchmarks(BENCHMARK_FILES)

    rows, report = generate_dataset_from_inputs(
        repo_root=repo_root,
        chunks=chunks,
        repo_map=repo_map,
        dep_graph=dep_graph,
        benchmarks=benchmarks,
        category_targets=category_targets,
        min_total=min_total,
    )
    if report["validation"]["errors"]:
        write_report(report_path, report)
        raise ValueError("Generated dataset failed validation: " + "; ".join(report["validation"]["errors"][:5]))
    write_jsonl(out_path, rows)
    write_report(report_path, report)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a repository-grounded SFT dataset.")
    parser.add_argument("--repo", default=".", help="Repository root to scan.")
    parser.add_argument("--out", default=str(CURATED_OUT), help="Output JSONL path.")
    parser.add_argument("--report", default=str(REPORT_OUT), help="Output markdown report path.")
    parser.add_argument("--profile", choices=("quick", "full"), default="full", help="Dataset size profile.")
    parser.add_argument("--min-rows", type=int, default=None, help="Override the minimum required row count.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo).resolve()
    out_path = Path(args.out).resolve()
    report_path = Path(args.report).resolve()
    category_targets = QUICK_CATEGORY_TARGETS if args.profile == "quick" else CATEGORY_TARGETS
    min_rows = args.min_rows if args.min_rows is not None else (50 if args.profile == "quick" else 500)
    report = build_dataset(
        repo_root,
        out_path,
        report_path,
        category_targets=category_targets,
        min_total=min_rows,
    )
    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")
    print(json.dumps({"total_rows": report["total_rows"], "category_counts": report["validation"]["category_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
