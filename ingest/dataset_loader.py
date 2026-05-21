"""Dataset ingestion for structured and unstructured files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from ingest.base import BaseLoader, DocumentRecord
from ingest.chunker import TextChunker


TEXT_FIELDS = ("text", "content", "body", "markdown", "description", "answer", "question")
TITLE_FIELDS = ("title", "name", "subject", "headline")


def _is_remote_uri(path_or_uri: str) -> bool:
    parsed = urlparse(path_or_uri)
    return parsed.scheme in {"http", "https", "file"}


def _source_name(path_or_uri: str) -> str:
    parsed = urlparse(path_or_uri)
    if parsed.scheme and parsed.path:
        return parsed.path
    return path_or_uri


def _read_text(path_or_uri: str, encoding: str = "utf-8") -> str:
    if _is_remote_uri(path_or_uri):
        with urlopen(path_or_uri) as response:
            return response.read().decode(encoding)
    return Path(path_or_uri).read_text(encoding=encoding)


def _read_bytes(path_or_uri: str) -> bytes:
    if _is_remote_uri(path_or_uri):
        with urlopen(path_or_uri) as response:
            return response.read()
    return Path(path_or_uri).read_bytes()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stem_or_name(path_or_uri: str) -> str:
    source = Path(_source_name(path_or_uri))
    return source.stem or source.name or path_or_uri


def _extension_of(path_or_uri: str) -> str:
    suffix = Path(_source_name(path_or_uri)).suffix.lower()
    return suffix or ""


def _metadata_base(path_or_uri: str, *, title: str | None = None, row_number: int | None = None, text: str = "") -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "original_path": path_or_uri,
        "extension": _extension_of(path_or_uri),
        "content_hash": _content_hash(text),
    }
    if row_number is not None:
        metadata["row_number"] = row_number
    if title:
        metadata["title"] = title
    return metadata


def _make_record(
    record_id: str,
    path_or_uri: str,
    text: str,
    *,
    doc_type: str,
    title: str | None = None,
    row_number: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> DocumentRecord:
    metadata = _metadata_base(path_or_uri, title=title, row_number=row_number, text=text)
    if extra_metadata:
        metadata.update(extra_metadata)
    return DocumentRecord(
        id=record_id,
        source=path_or_uri,
        text=text,
        metadata=metadata,
        doc_type=doc_type,
    )


def _extract_title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _extract_title_from_html(raw_html: str, fallback: str) -> str:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if not title_match:
        return fallback
    title = re.sub(r"\s+", " ", title_match.group(1)).strip()
    return html.unescape(title) or fallback


def _strip_html_tags(raw_html: str) -> str:
    text = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _extract_structured_text(row: Any) -> tuple[str, str | None]:
    if isinstance(row, dict):
        title = next((str(row[key]).strip() for key in TITLE_FIELDS if row.get(key)), None)
        parts = [str(row[key]).strip() for key in TEXT_FIELDS if row.get(key)]
        if parts:
            return "\n\n".join(part for part in parts if part), title
        return json.dumps(row, ensure_ascii=False, sort_keys=True), title
    if isinstance(row, list):
        return json.dumps(row, ensure_ascii=False), None
    return str(row), None


class TextFileLoader:
    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        text = _read_text(path_or_uri, encoding=kwargs.get("encoding", "utf-8"))
        title = _stem_or_name(path_or_uri)
        return [_make_record(title, path_or_uri, text, doc_type="text", title=title)]


class MarkdownLoader:
    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        text = _read_text(path_or_uri, encoding=kwargs.get("encoding", "utf-8"))
        title = _extract_title_from_markdown(text, _stem_or_name(path_or_uri))
        return [_make_record(_stem_or_name(path_or_uri), path_or_uri, text, doc_type="markdown", title=title)]


class JsonLoader:
    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        payload = json.loads(_read_text(path_or_uri, encoding=kwargs.get("encoding", "utf-8")))
        source_id = _stem_or_name(path_or_uri)

        if isinstance(payload, list):
            records: list[DocumentRecord] = []
            for index, row in enumerate(payload, start=1):
                text, title = _extract_structured_text(row)
                records.append(
                    _make_record(
                        f"{source_id}:{index}",
                        path_or_uri,
                        text,
                        doc_type="json",
                        title=title,
                        row_number=index,
                    )
                )
            return records

        text, title = _extract_structured_text(payload)
        return [_make_record(source_id, path_or_uri, text, doc_type="json", title=title)]


class JsonlLoader:
    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        lines = _read_text(path_or_uri, encoding=kwargs.get("encoding", "utf-8")).splitlines()
        source_id = _stem_or_name(path_or_uri)
        records: list[DocumentRecord] = []

        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            text, title = _extract_structured_text(row)
            records.append(
                _make_record(
                    f"{source_id}:{index}",
                    path_or_uri,
                    text,
                    doc_type="jsonl",
                    title=title,
                    row_number=index,
                )
            )

        return records


class CsvLoader:
    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        encoding = kwargs.get("encoding", "utf-8")
        source_id = _stem_or_name(path_or_uri)
        records: list[DocumentRecord] = []

        if _is_remote_uri(path_or_uri):
            from io import StringIO

            handle = StringIO(_read_text(path_or_uri, encoding=encoding))
            reader = csv.DictReader(handle)
            rows = list(reader)
        else:
            with Path(path_or_uri).open("r", encoding=encoding, newline="") as handle:
                rows = list(csv.DictReader(handle))

        for index, row in enumerate(rows, start=1):
            text, title = _extract_structured_text(row)
            records.append(
                _make_record(
                    f"{source_id}:{index}",
                    path_or_uri,
                    text,
                    doc_type="csv",
                    title=title,
                    row_number=index,
                )
            )

        return records


class HtmlLoader:
    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        raw_html = _read_text(path_or_uri, encoding=kwargs.get("encoding", "utf-8"))
        title = _extract_title_from_html(raw_html, _stem_or_name(path_or_uri))

        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(raw_html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        except ImportError:
            text = _strip_html_tags(raw_html)

        return [_make_record(_stem_or_name(path_or_uri), path_or_uri, text, doc_type="html", title=title)]


class ParquetLoader:
    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        source_id = _stem_or_name(path_or_uri)
        rows: list[dict[str, Any]]

        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(path_or_uri)
            rows = table.to_pylist()
        except ImportError:
            try:
                import pandas as pd  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Parquet ingestion requires an optional dependency. "
                    "Install `pyarrow` or `pandas`."
                ) from exc
            dataframe = pd.read_parquet(path_or_uri)
            rows = dataframe.to_dict(orient="records")

        records: list[DocumentRecord] = []
        for index, row in enumerate(rows, start=1):
            text, title = _extract_structured_text(row)
            records.append(
                _make_record(
                    f"{source_id}:{index}",
                    path_or_uri,
                    text,
                    doc_type="parquet",
                    title=title,
                    row_number=index,
                )
            )
        return records


class PdfLoader:
    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        source_id = _stem_or_name(path_or_uri)

        try:
            import fitz  # type: ignore

            document = fitz.open(path_or_uri)
            text = "\n\n".join(page.get_text("text") for page in document)
            title = document.metadata.get("title") or source_id
            return [_make_record(source_id, path_or_uri, text, doc_type="pdf", title=title)]
        except ImportError:
            pass

        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(path_or_uri) as document:
                text = "\n\n".join((page.extract_text() or "") for page in document.pages)
                metadata = getattr(document, "metadata", {}) or {}
                title = metadata.get("Title") or source_id
                return [_make_record(source_id, path_or_uri, text, doc_type="pdf", title=title)]
        except ImportError as exc:
            raise ImportError(
                "PDF ingestion requires an optional dependency. "
                "Install `PyMuPDF` or `pdfplumber`."
            ) from exc


LOADER_REGISTRY: dict[str, BaseLoader] = {
    ".txt": TextFileLoader(),
    ".md": MarkdownLoader(),
    ".json": JsonLoader(),
    ".jsonl": JsonlLoader(),
    ".csv": CsvLoader(),
    ".parquet": ParquetLoader(),
    ".html": HtmlLoader(),
    ".htm": HtmlLoader(),
    ".pdf": PdfLoader(),
}


def select_loader(path_or_uri: str, source_type: str | None = None) -> BaseLoader:
    """Select a loader by explicit source type or file extension."""
    if source_type:
        key = source_type.lower()
        if not key.startswith("."):
            key = f".{key}"
    else:
        key = _extension_of(path_or_uri)

    loader = LOADER_REGISTRY.get(key)
    if loader is None:
        raise ValueError(f"Unsupported dataset source: {path_or_uri} (type {key or 'unknown'})")
    return loader


def load_dataset_source(
    path_or_uri: str,
    **kwargs: Any,
) -> list[DocumentRecord]:
    """Load a dataset source into normalized document records."""
    source_type = kwargs.pop("source_type", None)
    chunk_size = kwargs.pop("chunk_size", None)
    chunk_overlap = kwargs.pop("chunk_overlap", 0)
    loader = select_loader(path_or_uri, source_type=source_type)
    records = loader.load(path_or_uri, **kwargs)

    if chunk_size is not None:
        chunker = TextChunker(chunk_size=int(chunk_size), overlap=int(chunk_overlap))
        return chunker.chunk_records(records)

    return records


def _write_jsonl(records: list[DocumentRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize arbitrary datasets into JSONL chunks.")
    parser.add_argument("--input", required=True, help="Input dataset path or URI.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--source-type", default=None, help="Optional explicit source type override.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for text splitting.")
    parser.add_argument("--encoding", default="utf-8", help="Text encoding for text-like files.")
    args = parser.parse_args()

    records = load_dataset_source(
        args.input,
        source_type=args.source_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        encoding=args.encoding,
    )
    _write_jsonl(records, Path(args.out))
    print(f"Wrote {len(records)} normalized records to {args.out}")


if __name__ == "__main__":
    main()
