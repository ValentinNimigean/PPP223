from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


TOXIC_TERMS = {
    "idiot",
    "stupid",
    "moron",
    "kill yourself",
    "go die",
    "i hate you",
    "worthless",
    "shut up",
}


REFUSAL_TEXT = (
    "I cannot help with abusive, threatening, or harmful language. "
    "Please rephrase your request in a neutral way."
)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_toxic_question(question: str, expected_entities: List[str]) -> bool:
    q = (question or "").lower()
    expected_lower = {str(e).lower() for e in expected_entities}

    if {"cannot help", "rephrase"}.issubset(expected_lower):
        return True

    return any(term in q for term in TOXIC_TERMS)


def meaningful_entities(expected_entities: List[str]) -> List[str]:
    ignored = {"cannot find", "cannot help", "rephrase"}
    return [
        str(e).strip()
        for e in expected_entities
        if str(e).strip() and str(e).strip().lower() not in ignored
    ]


def no_match_entity(expected_entities: List[str], question: str) -> str:
    entities = meaningful_entities(expected_entities)
    if entities:
        return entities[0]

    # Try backtick-enclosed symbols first.
    backtick = re.findall(r"`([^`]+)`", question or "")
    if backtick:
        return backtick[0]

    # Then try PascalCase / symbol-looking names.
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", question or "")
    ignored = {"Does", "Where", "What", "How", "Which", "Can"}
    for candidate in candidates:
        if candidate not in ignored:
            return candidate

    return "that symbol"


def fmt_entities(entities: List[str]) -> str:
    clean = [str(e).strip() for e in entities if str(e).strip()]
    if not clean:
        return "`the expected entity`"
    if len(clean) == 1:
        return f"`{clean[0]}`"
    return ", ".join(f"`{e}`" for e in clean[:-1]) + f", and `{clean[-1]}`"


def fmt_files(files: List[str]) -> str:
    clean = [str(f).strip() for f in files if str(f).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return f"`{clean[0]}`"
    return ", ".join(f"`{f}`" for f in clean[:-1]) + f", and `{clean[-1]}`"


def make_chosen_answer(question: str, expected_entities: List[str], expected_files: List[str]) -> str:
    if is_toxic_question(question, expected_entities):
        return REFUSAL_TEXT

    if not expected_files:
        entity = no_match_entity(expected_entities, question)
        return f"I cannot find `{entity}` in this repository."

    entities = meaningful_entities(expected_entities)
    entity_text = fmt_entities(entities or expected_entities)
    file_text = fmt_files(expected_files)
    return f"{entity_text} is implemented in {file_text}."


def validate_chosen(
    question: str,
    chosen: str,
    rejected: str,
    expected_entities: List[str],
    expected_files: List[str],
) -> Tuple[bool, str]:
    chosen = chosen or ""
    rejected = rejected or ""

    if not chosen.strip():
        return False, "empty chosen"

    if chosen.strip() == rejected.strip():
        return False, "chosen equals rejected"

    if "abusive" in chosen.lower() and not is_toxic_question(question, expected_entities):
        return False, "non-toxic question got toxicity refusal"

    for file_path in expected_files:
        if file_path not in chosen:
            return False, f"missing expected file {file_path}"

    entities = meaningful_entities(expected_entities)
    if entities and not any(entity in chosen for entity in entities):
        return False, "missing expected entity"

    return True, "ok"


def rows_from_report(
    report: Dict[str, Any],
    source_report: str,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "rows_read": 0,
        "rows_below_threshold": 0,
        "rows_written": 0,
        "rows_skipped": 0,
    }
    skipped_reasons: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []

    for item in report.get("results", []):
        stats["rows_read"] += 1

        score = float(item.get("penalized_combined_score", item.get("combined_score", 0.0)) or 0.0)
        if score >= threshold:
            continue

        stats["rows_below_threshold"] += 1

        question = str(item.get("question") or "").strip()
        rejected = str(item.get("response") or "").strip()
        expected_entities = item.get("expected_entities") or []
        expected_files = item.get("expected_files") or []

        if not question or not rejected:
            stats["rows_skipped"] += 1
            skipped_reasons["missing question or rejected"] = skipped_reasons.get("missing question or rejected", 0) + 1
            continue

        chosen = make_chosen_answer(question, expected_entities, expected_files)

        ok, reason = validate_chosen(question, chosen, rejected, expected_entities, expected_files)
        if not ok:
            stats["rows_skipped"] += 1
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        rows.append(
            {
                "prompt": question,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "original_score": score,
                    "expected_entities": expected_entities,
                    "expected_files": expected_files,
                    "source_report": source_report,
                },
            }
        )
        stats["rows_written"] += 1

    for reason, count in skipped_reasons.items():
        print(f"Skipped {count}: {reason}")

    return rows, stats


def main():
    parser = argparse.ArgumentParser(description="Create failure-driven DPO rows from eval report.")
    parser.add_argument("--report", required=True, help="Path to eval_report*.json")
    parser.add_argument("--out", default="training_data/preferences/preference_data_failures.jsonl")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--repo-map", default=None, help="Reserved for future richer chosen-answer generation.")
    parser.add_argument("--benchmark", default=None, help="Reserved for future benchmark-aware generation.")
    args = parser.parse_args()

    report = load_json(args.report)
    rows, stats = rows_from_report(report, source_report=args.report, threshold=args.threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"Wrote {len(rows)} DPO failure rows to {out_path}")


if __name__ == "__main__":
    main()
