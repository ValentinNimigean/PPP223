#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

DEFAULT_SFT_FILES = [
    REPO_ROOT / "synthetic_qa_seed.jsonl",
    REPO_ROOT / "synthetic_qa_combined.jsonl",
    REPO_ROOT / "synthetic_qa_combined_2048.jsonl",
]
DEFAULT_PREFERENCE_FILES = [REPO_ROOT / "preference_data_combined.jsonl"]
DEFAULT_EVAL_FILES = [
    REPO_ROOT / "eval" / "benchmark_self.json",
    REPO_ROOT / "eval" / "benchmark_unseen_self.json",
    REPO_ROOT / "eval" / "benchmark_httpx.json",
]

LOCAL_PATH_PATTERNS = [
    "/home/",
    "file://",
    "Documents/GitHub",
    "C:\\",
]
PENDING_CORRECTION_PATTERNS = [
    "pending correction",
    "todo",
    "fixme",
    "tbd",
    "placeholder",
]
CONTAMINATION_TERMS = [
    "AWS SDK",
    "Alibaba Cloud",
    "Django framework",
    "TensorFlow",
    "PyTorch",
    "SageMaker",
    "handle_text_tool_call",
    "QWENClient",
    "client_manager",
    "github.com/aws",
    "github.com/alibabacloud",
    "boto3",
    "botocore",
    "django.db",
    "FastAPI framework",
]
TOXIC_TERMS = [
    "idiot",
    "stupid",
    "moron",
    "kill yourself",
    "go die",
    "i hate you",
    "worthless",
    "shut up",
]
PATH_RE = re.compile(r"(?<![\w/.-])(?:[A-Za-z]:\\|/?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)\.(?:py|sh|jsonl|json|md|txt|yaml|yml|toml)")


@dataclass
class LoadedRow:
    data: Optional[Dict[str, Any]]
    line_no: int
    raw_line: str
    parse_error: Optional[str] = None


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_prompt(text: Any) -> str:
    return normalize_text(text).lower()


def is_toxic_prompt(text: str) -> bool:
    lowered = normalize_prompt(text)
    return any(term in lowered for term in TOXIC_TERMS)


def load_jsonl(path: Path) -> List[LoadedRow]:
    if not path.exists():
        return [LoadedRow(data=None, line_no=0, raw_line="", parse_error=f"missing file: {path}")]

    rows: List[LoadedRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append(LoadedRow(data=None, line_no=line_no, raw_line=line, parse_error=f"invalid JSON: {exc}"))
            continue
        if not isinstance(data, dict):
            rows.append(LoadedRow(data=None, line_no=line_no, raw_line=line, parse_error="row is not a JSON object"))
            continue
        rows.append(LoadedRow(data=data, line_no=line_no, raw_line=line))
    return rows


def infer_dataset_kind(path: Path) -> Optional[str]:
    for loaded in load_jsonl(path):
        if loaded.parse_error or loaded.data is None:
            continue
        if "messages" in loaded.data:
            return "sft"
        if "prompt" in loaded.data or "chosen" in loaded.data or "rejected" in loaded.data:
            return "preference"
    return None


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def iter_message_texts(messages: Sequence[Dict[str, Any]]) -> Iterable[str]:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is None:
            continue
        yield str(content)


def detect_contamination(text: str) -> List[str]:
    hits: List[str] = []
    lowered = text.lower()
    for marker in LOCAL_PATH_PATTERNS:
        if marker.lower() in lowered:
            hits.append(marker)
    for term in CONTAMINATION_TERMS:
        if term.lower() in lowered:
            hits.append(term)
    return hits


def extract_paths(text: str) -> List[str]:
    return PATH_RE.findall(text)


def normalize_path_reference(path_text: str) -> str:
    normalized = path_text.replace("\\", "/").strip()
    if normalized.startswith(str(REPO_ROOT).replace("\\", "/")):
        normalized = normalized[len(str(REPO_ROOT).replace("\\", "/")) :].lstrip("/")
    return normalized.lstrip("./")


def find_hallucinated_paths(text: str, repo_files: set[str]) -> List[str]:
    hallucinated: List[str] = []
    for raw_path in extract_paths(text):
        normalized = normalize_path_reference(raw_path)
        if normalized.startswith("/"):
            continue
        if normalized not in repo_files:
            hallucinated.append(raw_path)
    return hallucinated


def looks_like_raw_tool_json(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    candidate = stripped
    if stripped.startswith("```"):
        candidate = stripped.strip("`")
        candidate = re.sub(r"^json\s*", "", candidate, flags=re.IGNORECASE)
    if not candidate.startswith("{"):
        return False
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    keys = set(payload)
    return bool({"tool", "tool_name", "tool_calls", "function", "arguments"} & keys)


def get_repo_file_set(repo_root: Path) -> set[str]:
    files = set()
    for path in repo_root.rglob("*"):
        if path.is_file():
            try:
                files.add(path.relative_to(repo_root).as_posix())
            except ValueError:
                continue
    return files


def extract_final_assistant(messages: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    return None


def extract_first_user_content(messages: Sequence[Dict[str, Any]]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def has_malformed_role_order(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 3:
        return True
    roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
    if len(roles) != len(messages):
        return True
    if roles[0] != "system" or roles[1] != "user" or roles[-1] != "assistant":
        return True
    seen_user = False
    seen_assistant = False
    for index, role in enumerate(roles):
        if role not in {"system", "user", "assistant", "tool"}:
            return True
        if role == "system" and index != 0:
            return True
        if role == "user":
            seen_user = True
            if seen_assistant:
                return True
        if role == "assistant":
            seen_assistant = True
        if role == "tool" and not seen_assistant:
            return True
    return not seen_user


def summarize_counter(counter: Counter[str], limit: int = 10) -> List[Dict[str, Any]]:
    items = []
    for value, count in counter.most_common(limit):
        items.append({"value": value, "count": count})
    return items


def maybe_build_tokenizer() -> Tuple[Optional[Any], Optional[str]]:
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base"), "tiktoken:cl100k_base"
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        return None, None

    candidates = []
    env_candidate = os.environ.get("DATASET_AUDIT_TOKENIZER")
    if env_candidate:
        candidates.append(env_candidate)
    candidates.extend(["gpt2", "distilbert-base-uncased"])

    for name in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            return tokenizer, f"transformers:{name}"
        except Exception:
            continue
    return None, None


def tokenize_text(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        return 0
    if hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text))
        except TypeError:
            return len(tokenizer.encode(text, add_special_tokens=False))
    return 0


def token_stats(lengths: List[int], tokenizer_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not lengths or not tokenizer_name:
        return None
    lengths = sorted(lengths)
    mid = len(lengths) // 2
    median = lengths[mid] if len(lengths) % 2 else int((lengths[mid - 1] + lengths[mid]) / 2)
    return {
        "tokenizer": tokenizer_name,
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(mean(lengths), 2),
        "median": median,
    }


def audit_sft_dataset(path: Path, repo_files: set[str], tokenizer: Any = None, tokenizer_name: Optional[str] = None) -> Dict[str, Any]:
    rows = load_jsonl(path)
    total_rows = len(rows)
    valid_rows = 0
    invalid_rows = 0

    exact_rows: Counter[str] = Counter()
    user_questions: Counter[str] = Counter()
    duplicate_exact_rows = 0
    duplicate_user_questions = 0
    empty_assistant_answers = 0
    short_assistant_answers = 0
    assistant_answers_with_none = 0
    malformed_role_order = 0
    raw_tool_json_final_answers = 0
    contamination_rows = 0
    hallucinated_path_rows = 0
    token_lengths: List[int] = []
    errors: List[str] = []
    contamination_examples: List[Dict[str, Any]] = []
    hallucinated_examples: List[Dict[str, Any]] = []

    for row in rows:
        if row.parse_error:
            invalid_rows += 1
            errors.append(f"line {row.line_no}: {row.parse_error}")
            continue

        assert row.data is not None
        exact_rows[json.dumps(row.data, sort_keys=True, ensure_ascii=False)] += 1

        messages = row.data.get("messages")
        row_errors: List[str] = []
        if has_malformed_role_order(messages):
            malformed_role_order += 1
            row_errors.append("malformed role order")

        if not isinstance(messages, list):
            row_errors.append("messages is not a list")
            invalid_rows += 1
            errors.append(f"line {row.line_no}: {'; '.join(row_errors)}")
            continue

        question = extract_first_user_content(messages)
        if question:
            user_questions[normalize_prompt(question)] += 1

        assistant_messages = [
            message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        has_empty_assistant = False
        has_none_assistant = False
        for message in assistant_messages:
            content = message.get("content")
            if content is None:
                has_none_assistant = True
                has_empty_assistant = True
                continue
            if not str(content).strip():
                has_empty_assistant = True
        if has_none_assistant:
            assistant_answers_with_none += 1
            row_errors.append("assistant answer is None")
        if has_empty_assistant:
            empty_assistant_answers += 1
            if "assistant answer is None" not in row_errors:
                row_errors.append("empty assistant answer")

        final_assistant = extract_final_assistant(messages)
        final_content = None if final_assistant is None else final_assistant.get("content")
        if final_assistant is None:
            row_errors.append("missing final assistant message")
        else:
            final_text = str(final_content).strip()
            if not final_text:
                if "empty assistant answer" not in row_errors:
                    row_errors.append("empty assistant answer")
            else:
                if len(final_text) < 20:
                    short_assistant_answers += 1
                if looks_like_raw_tool_json(final_text):
                    raw_tool_json_final_answers += 1
                    row_errors.append("raw tool-call JSON final answer")
                if tokenizer is not None:
                    token_lengths.append(tokenize_text(tokenizer, final_text))

        joined_text = "\n".join(iter_message_texts(messages))
        contamination_hits = detect_contamination(joined_text)
        if contamination_hits:
            contamination_rows += 1
            row_errors.append("contamination detected")
            if len(contamination_examples) < 10:
                contamination_examples.append(
                    {"line": row.line_no, "hits": contamination_hits[:5], "question": normalize_text(question)}
                )
        hallucinated_paths = find_hallucinated_paths(joined_text, repo_files)
        if hallucinated_paths:
            hallucinated_path_rows += 1
            row_errors.append("hallucinated file path detected")
            if len(hallucinated_examples) < 10:
                hallucinated_examples.append(
                    {"line": row.line_no, "paths": hallucinated_paths[:5], "question": normalize_text(question)}
                )

        if row_errors:
            invalid_rows += 1
            errors.append(f"line {row.line_no}: {'; '.join(row_errors)}")
        else:
            valid_rows += 1

    duplicate_exact_rows = sum(count - 1 for count in exact_rows.values() if count > 1)
    duplicate_user_questions = sum(count - 1 for count in user_questions.values() if count > 1)

    return {
        "path": path.as_posix(),
        "dataset_type": "sft",
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "duplicate_exact_rows": duplicate_exact_rows,
        "duplicate_user_questions": duplicate_user_questions,
        "empty_assistant_answers": empty_assistant_answers,
        "assistant_answers_under_20_chars": short_assistant_answers,
        "assistant_answers_with_none": assistant_answers_with_none,
        "malformed_role_order": malformed_role_order,
        "raw_tool_call_json_final_answers": raw_tool_json_final_answers,
        "contamination_rows": contamination_rows,
        "hallucinated_file_path_rows": hallucinated_path_rows,
        "token_length_stats": token_stats(token_lengths, tokenizer_name),
        "top_duplicate_questions": summarize_counter(Counter({k: v for k, v in user_questions.items() if v > 1})),
        "top_duplicate_rows": [
            {"count": count, "row": json.loads(row_text)}
            for row_text, count in exact_rows.most_common(5)
            if count > 1
        ],
        "contamination_examples": contamination_examples,
        "hallucinated_path_examples": hallucinated_examples,
        "errors": errors[:200],
    }


def has_toxic_label(row: Dict[str, Any]) -> bool:
    if any(key in row for key in ("is_toxic", "prompt_is_toxic", "toxicity_label")):
        return True
    meta = row.get("meta")
    return isinstance(meta, dict) and any(key in meta for key in ("is_toxic", "prompt_is_toxic", "toxicity_label"))


def audit_preference_dataset(path: Path, repo_files: set[str], tokenizer: Any = None, tokenizer_name: Optional[str] = None) -> Dict[str, Any]:
    rows = load_jsonl(path)
    total_rows = len(rows)
    valid_rows = 0
    invalid_rows = 0

    empty_chosen = 0
    empty_rejected = 0
    chosen_equals_rejected = 0
    rejected_missing = 0
    contamination_rows = 0
    toxic_prompt_labels_missing = 0
    pending_correction_markers = 0
    prompt_counter: Counter[str] = Counter()
    token_lengths: List[int] = []
    errors: List[str] = []
    contamination_examples: List[Dict[str, Any]] = []

    for row in rows:
        if row.parse_error:
            invalid_rows += 1
            errors.append(f"line {row.line_no}: {row.parse_error}")
            continue

        assert row.data is not None
        data = row.data
        row_errors: List[str] = []

        prompt = normalize_text(data.get("prompt"))
        chosen = data.get("chosen")
        rejected_value = data.get("rejected") if "rejected" in data else None
        rejected = normalize_text(rejected_value)
        chosen_text = normalize_text(chosen)

        if prompt:
            prompt_counter[normalize_prompt(prompt)] += 1
        else:
            row_errors.append("empty prompt")

        if chosen is None or not chosen_text:
            empty_chosen += 1
            row_errors.append("empty chosen")

        if "rejected" not in data:
            rejected_missing += 1
            empty_rejected += 1
            row_errors.append("rejected missing")
        elif rejected_value is None or not rejected:
            empty_rejected += 1
            row_errors.append("empty rejected")

        if chosen_text and rejected and chosen_text == rejected:
            chosen_equals_rejected += 1
            row_errors.append("chosen equals rejected")

        combined_text = "\n".join(
            [
                prompt,
                chosen_text,
                rejected,
                json.dumps(data.get("meta"), ensure_ascii=False) if "meta" in data else "",
            ]
        )
        contamination_hits = detect_contamination(combined_text)
        hallucinated_paths = find_hallucinated_paths(combined_text, repo_files)
        if contamination_hits or hallucinated_paths:
            contamination_rows += 1
            row_errors.append("contamination detected")
            if len(contamination_examples) < 10:
                contamination_examples.append(
                    {
                        "line": row.line_no,
                        "hits": contamination_hits[:5],
                        "hallucinated_paths": hallucinated_paths[:5],
                        "prompt": prompt,
                    }
                )

        if is_toxic_prompt(prompt) and not has_toxic_label(data):
            toxic_prompt_labels_missing += 1
            row_errors.append("toxic prompt label missing")

        lowered = combined_text.lower()
        if any(marker in lowered for marker in PENDING_CORRECTION_PATTERNS):
            pending_correction_markers += 1
            row_errors.append("pending correction marker present")

        if tokenizer is not None and chosen_text:
            token_lengths.append(tokenize_text(tokenizer, chosen_text))

        if row_errors:
            invalid_rows += 1
            errors.append(f"line {row.line_no}: {'; '.join(row_errors)}")
        else:
            valid_rows += 1

    prompt_duplicates = sum(count - 1 for count in prompt_counter.values() if count > 1)

    return {
        "path": path.as_posix(),
        "dataset_type": "preference",
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "empty_chosen": empty_chosen,
        "empty_rejected": empty_rejected,
        "chosen_equals_rejected": chosen_equals_rejected,
        "rejected_missing": rejected_missing,
        "prompt_duplicates": prompt_duplicates,
        "toxic_prompt_labels_missing": toxic_prompt_labels_missing,
        "contamination_rows": contamination_rows,
        "pending_correction_markers": pending_correction_markers,
        "token_length_stats": token_stats(token_lengths, tokenizer_name),
        "top_duplicate_prompts": summarize_counter(Counter({k: v for k, v in prompt_counter.items() if v > 1})),
        "contamination_examples": contamination_examples,
        "errors": errors[:200],
    }


def load_eval_questions(paths: Sequence[Path]) -> Dict[str, List[str]]:
    questions: Dict[str, List[str]] = {}
    for path in paths:
        dataset_questions = []
        for row in load_json_array(path):
            question = normalize_text(row.get("question") or row.get("prompt"))
            if question:
                dataset_questions.append(question)
        questions[path.as_posix()] = dataset_questions
    return questions


def compute_leakage(sft_reports: Sequence[Dict[str, Any]], preference_reports: Sequence[Dict[str, Any]], eval_questions: Dict[str, List[str]]) -> Dict[str, Any]:
    train_prompts: Dict[str, set[str]] = {}
    for report in list(sft_reports) + list(preference_reports):
        prompts = set()
        path = Path(report["path"])
        if report["dataset_type"] == "sft":
            for row in load_jsonl(path):
                if row.data and isinstance(row.data.get("messages"), list):
                    prompt = extract_first_user_content(row.data["messages"])
                    if prompt:
                        prompts.add(normalize_prompt(prompt))
        else:
            for row in load_jsonl(path):
                if row.data and row.data.get("prompt"):
                    prompts.add(normalize_prompt(row.data["prompt"]))
        train_prompts[report["path"]] = prompts

    train_union = set().union(*train_prompts.values()) if train_prompts else set()
    leakage_by_eval: Dict[str, Dict[str, Any]] = {}
    total_overlaps = 0
    for eval_path, questions in eval_questions.items():
        normalized = [normalize_prompt(q) for q in questions]
        overlaps = sorted({question for question in normalized if question in train_union})
        total_overlaps += len(overlaps)
        leakage_by_eval[eval_path] = {
            "eval_questions": len(normalized),
            "exact_overlaps": len(overlaps),
            "overlap_examples": overlaps[:20],
        }

    return {
        "train_prompt_count": len(train_union),
        "eval_files": leakage_by_eval,
        "total_exact_overlaps": total_overlaps,
    }


def build_summary(audit: Dict[str, Any]) -> Dict[str, Any]:
    sft = audit["sft_datasets"]
    pref = audit["preference_datasets"]
    return {
        "sft_total_rows": sum(item["total_rows"] for item in sft),
        "sft_invalid_rows": sum(item["invalid_rows"] for item in sft),
        "sft_duplicate_exact_rows": sum(item["duplicate_exact_rows"] for item in sft),
        "sft_empty_assistant_answers": sum(item["empty_assistant_answers"] for item in sft),
        "preference_total_rows": sum(item["total_rows"] for item in pref),
        "preference_invalid_rows": sum(item["invalid_rows"] for item in pref),
        "preference_empty_chosen": sum(item["empty_chosen"] for item in pref),
        "preference_empty_rejected": sum(item["empty_rejected"] for item in pref),
        "train_eval_exact_overlaps": audit["train_eval_leakage"]["total_exact_overlaps"],
        "audit_passed": audit["passed"],
    }


def determine_failures(audit: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    sft_total_rows = sum(item["total_rows"] for item in audit["sft_datasets"])
    sft_duplicates = sum(item["duplicate_exact_rows"] for item in audit["sft_datasets"])
    sft_empty = sum(item["empty_assistant_answers"] for item in audit["sft_datasets"])
    pref_empty_chosen = sum(item["empty_chosen"] for item in audit["preference_datasets"])
    pref_empty_rejected = sum(item["empty_rejected"] for item in audit["preference_datasets"])
    pref_valid = sum(item["valid_rows"] for item in audit["preference_datasets"])

    if sft_empty > 0:
        failures.append(f"SFT datasets contain {sft_empty} empty assistant answers.")
    if sft_total_rows and (sft_duplicates / sft_total_rows) > 0.05:
        failures.append(f"SFT duplicate exact rows exceed 5% ({sft_duplicates}/{sft_total_rows}).")
    if pref_empty_chosen > 0 or pref_empty_rejected > 0:
        failures.append(
            f"Preference datasets contain empty pairs (chosen={pref_empty_chosen}, rejected={pref_empty_rejected})."
        )
    if pref_valid < 100:
        failures.append(f"Preference datasets have fewer than 100 valid pairs ({pref_valid}).")
    return failures


def write_markdown_report(path: Path, audit: Dict[str, Any]) -> None:
    lines = [
        "# Dataset Audit",
        "",
        f"Passed: `{'yes' if audit['passed'] else 'no'}`",
        "",
        "## Failures",
    ]
    if audit["failures"]:
        for failure in audit["failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")

    lines.extend(["", "## SFT Datasets"])
    for item in audit["sft_datasets"]:
        lines.extend(
            [
                f"### `{Path(item['path']).name}`",
                f"- total rows: {item['total_rows']}",
                f"- valid rows: {item['valid_rows']}",
                f"- invalid rows: {item['invalid_rows']}",
                f"- duplicate exact rows: {item['duplicate_exact_rows']}",
                f"- duplicate user questions: {item['duplicate_user_questions']}",
                f"- empty assistant answers: {item['empty_assistant_answers']}",
                f"- assistant answers under 20 chars: {item['assistant_answers_under_20_chars']}",
                f"- assistant answers with None: {item['assistant_answers_with_none']}",
                f"- malformed role order: {item['malformed_role_order']}",
                f"- raw tool-call JSON final answers: {item['raw_tool_call_json_final_answers']}",
                f"- contamination rows: {item['contamination_rows']}",
                f"- hallucinated file path rows: {item['hallucinated_file_path_rows']}",
            ]
        )
        if item["token_length_stats"]:
            stats = item["token_length_stats"]
            lines.append(
                f"- token stats ({stats['tokenizer']}): min={stats['min']} median={stats['median']} mean={stats['mean']} max={stats['max']}"
            )

    lines.extend(["", "## Preference Datasets"])
    for item in audit["preference_datasets"]:
        lines.extend(
            [
                f"### `{Path(item['path']).name}`",
                f"- total rows: {item['total_rows']}",
                f"- valid rows: {item['valid_rows']}",
                f"- invalid rows: {item['invalid_rows']}",
                f"- empty chosen: {item['empty_chosen']}",
                f"- empty rejected: {item['empty_rejected']}",
                f"- chosen == rejected: {item['chosen_equals_rejected']}",
                f"- rejected missing: {item['rejected_missing']}",
                f"- prompt duplicates: {item['prompt_duplicates']}",
                f"- toxic prompt labels missing: {item['toxic_prompt_labels_missing']}",
                f"- contamination rows: {item['contamination_rows']}",
                f"- pending correction markers: {item['pending_correction_markers']}",
            ]
        )
        if item["token_length_stats"]:
            stats = item["token_length_stats"]
            lines.append(
                f"- token stats ({stats['tokenizer']}): min={stats['min']} median={stats['median']} mean={stats['mean']} max={stats['max']}"
            )

    lines.extend(["", "## Train/Eval Leakage"])
    lines.append(f"- total exact overlaps: {audit['train_eval_leakage']['total_exact_overlaps']}")
    for eval_path, leakage in audit["train_eval_leakage"]["eval_files"].items():
        lines.append(
            f"- `{Path(eval_path).name}`: overlaps={leakage['exact_overlaps']} / questions={leakage['eval_questions']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    sft_files: Optional[Sequence[Path]] = None,
    preference_files: Optional[Sequence[Path]] = None,
    eval_files: Optional[Sequence[Path]] = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    repo_files = get_repo_file_set(repo_root)
    tokenizer, tokenizer_name = maybe_build_tokenizer()

    sft_paths = list(sft_files or DEFAULT_SFT_FILES)
    preference_paths = list(preference_files or DEFAULT_PREFERENCE_FILES)
    eval_paths = list(eval_files or DEFAULT_EVAL_FILES)

    sft_reports = [audit_sft_dataset(path, repo_files, tokenizer, tokenizer_name) for path in sft_paths]
    preference_reports = [
        audit_preference_dataset(path, repo_files, tokenizer, tokenizer_name) for path in preference_paths
    ]
    eval_questions = load_eval_questions(eval_paths)
    leakage = compute_leakage(sft_reports, preference_reports, eval_questions)

    audit: Dict[str, Any] = {
        "repo_root": repo_root.as_posix(),
        "tokenizer": tokenizer_name,
        "sft_datasets": sft_reports,
        "preference_datasets": preference_reports,
        "train_eval_leakage": leakage,
    }
    audit["failures"] = determine_failures(audit)
    audit["passed"] = not audit["failures"]
    audit["summary"] = build_summary(audit)
    return audit


def write_artifacts(audit: Dict[str, Any], output_dir: Path = ARTIFACTS_DIR) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_audit.json"
    md_path = output_dir / "dataset_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_report(md_path, audit)
    return json_path, md_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit SFT and preference datasets.")
    parser.add_argument("--sft", nargs="*", default=None, help="Optional explicit SFT JSONL paths.")
    parser.add_argument("--preferences", nargs="*", default=None, help="Optional explicit preference JSONL paths.")
    parser.add_argument("--paths", nargs="*", default=None, help="Optional mixed JSONL paths to auto-classify as SFT or preference datasets.")
    parser.add_argument("--allow-fail", action="store_true", help="Always exit 0 after writing the audit report.")
    args = parser.parse_args(list(argv or sys.argv[1:]))
    sft_files = [Path(path).resolve() for path in args.sft] if args.sft else []
    preference_files = [Path(path).resolve() for path in args.preferences] if args.preferences else []
    for raw_path in args.paths or []:
        path = Path(raw_path).resolve()
        kind = infer_dataset_kind(path)
        if kind == "sft":
            sft_files.append(path)
        elif kind == "preference":
            preference_files.append(path)
        else:
            raise ValueError(f"Could not infer dataset type for path: {path}")
    sft_arg = sft_files or None
    preference_arg = preference_files or None
    audit = run_audit(sft_files=sft_arg, preference_files=preference_arg)
    json_path, md_path = write_artifacts(audit)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(audit["summary"], indent=2))

    if audit["failures"] and not args.allow_fail:
        print("\nAudit failures:")
        for failure in audit["failures"]:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
