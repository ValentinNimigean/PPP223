import re
from typing import Dict, Iterable, List, Optional

from ingest.metadata import CodeChunk


def _is_code_like_identifier(identifier: str) -> bool:
    if "_" in identifier:
        return True
    if "." in identifier:
        return True

    has_upper = any(c.isupper() for c in identifier)
    has_lower = any(c.islower() for c in identifier)
    if has_upper and has_lower and not identifier.istitle():
        return True
    if identifier.isupper() and len(identifier) > 2:
        return True
    return False


class HallucinationDetector:
    IGNORE = {
        "def", "class", "return", "import", "from", "self", "true",
        "false", "none", "and", "or", "not", "in", "is", "if", "else",
        "for", "while", "with", "as", "try", "except", "pass", "print",
        "str", "int", "float", "list", "dict", "set", "bool", "len",
        "range", "type", "the", "app", "tests", "repository", "answer",
        "function", "method", "file", "code", "flow", "chat", "driven", "uses",
    }

    def __init__(self, chunks: List[CodeChunk]):
        self.known_names = set()
        self.known_qualified = set()
        self.known_files = set()

        for chunk in chunks:
            if chunk.name:
                self.known_names.add(chunk.name.lower())
            if chunk.qualified_name:
                self.known_qualified.add(chunk.qualified_name.lower())
            if chunk.filepath:
                self.known_files.add(chunk.filepath.lower())

        self.known_all = self.known_names.union(self.known_qualified).union(self.known_files)

    def scan(
        self,
        response: str,
        allowed_entities: Optional[Iterable[str]] = None,
        allowed_files: Optional[Iterable[str]] = None,
    ) -> Dict:
        identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\b", response or ""))

        allowed_entities_set = {str(item).lower() for item in (allowed_entities or []) if str(item).strip()}
        allowed_files_set = {str(item).lower() for item in (allowed_files or []) if str(item).strip()}
        verified_entities = []
        unverified_entities = []
        unsupported_entities = []

        for ident in identifiers:
            ident_lower = ident.lower()
            if ident_lower.endswith(".py"):
                continue
            if ident_lower in self.IGNORE:
                continue
            if not _is_code_like_identifier(ident):
                continue

            if ident_lower in self.known_all:
                verified_entities.append(ident)
            else:
                unverified_entities.append(ident)

            if allowed_entities_set and ident_lower not in allowed_entities_set:
                unsupported_entities.append(ident)

        file_candidates = set(re.findall(r"[\w/\\.-]+\.py", response or ""))
        verified_files = []
        unverified_files = []
        unsupported_files = []

        for path in file_candidates:
            path_lower = path.lower()
            if path_lower in self.known_files:
                verified_files.append(path)
            else:
                unverified_files.append(path)

            if allowed_files_set and path_lower not in allowed_files_set:
                unsupported_files.append(path)

        total_extracted = (
            len(unverified_entities)
            + len(verified_entities)
            + len(unverified_files)
            + len(verified_files)
            + len(unsupported_entities)
            + len(unsupported_files)
        )
        unsupported_count = (
            len(unverified_entities)
            + len(unverified_files)
            + len(set(unsupported_entities))
            + len(set(unsupported_files))
        )
        score = unsupported_count / (total_extracted + 1e-9)

        if score < 0.2:
            risk = "low"
        elif score < 0.5:
            risk = "medium"
        else:
            risk = "high"

        return {
            "verified_entities": sorted(set(verified_entities)),
            "unverified_entities": sorted(set(unverified_entities)),
            "verified_files": sorted(set(verified_files)),
            "unverified_files": sorted(set(unverified_files)),
            "unsupported_entities": sorted(set(unsupported_entities)),
            "unsupported_files": sorted(set(unsupported_files)),
            "hallucination_risk": risk,
            "hallucination_score": score,
        }

    def format_report(self, scan_result: Dict) -> str:
        lines = [
            f"Hallucination Risk: {scan_result['hallucination_risk'].upper()} (score: {scan_result['hallucination_score']:.2f})",
            f"Verified entities: {', '.join(scan_result['verified_entities']) if scan_result['verified_entities'] else 'None'}",
            f"Unverified entities: {', '.join(scan_result['unverified_entities']) if scan_result['unverified_entities'] else 'None'}",
            f"Verified files: {', '.join(scan_result['verified_files']) if scan_result['verified_files'] else 'None'}",
            f"Unverified files: {', '.join(scan_result['unverified_files']) if scan_result['unverified_files'] else 'None'}",
        ]
        return "\n".join(lines)
