import json
import subprocess
import sys
from pathlib import Path

import pytest

from ingest.chunker import TextChunker
from ingest.dataset_loader import load_dataset_source
from ingest.loader import Loader


def test_text_loader_returns_chunked_document_records(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    records = load_dataset_source(str(path), chunk_size=10, chunk_overlap=2)

    assert len(records) >= 2
    assert all(record.doc_type == "text" for record in records)
    assert all(record.metadata["original_path"] == str(path) for record in records)
    assert all(record.chunk_id for record in records)
    assert all(record.metadata["extension"] == ".txt" for record in records)


def test_markdown_loader_extracts_heading_title(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Sample Title\n\nParagraph body.", encoding="utf-8")

    records = load_dataset_source(str(path), chunk_size=None)

    assert len(records) == 1
    assert records[0].doc_type == "markdown"
    assert records[0].metadata["title"] == "Sample Title"


def test_jsonl_loader_preserves_row_metadata(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {"title": "First", "text": "First body"},
        {"name": "Second", "content": "Second body"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = load_dataset_source(str(path), chunk_size=None)

    assert len(records) == 2
    assert records[0].metadata["row_number"] == 1
    assert records[1].metadata["row_number"] == 2
    assert records[0].metadata["title"] == "First"
    assert records[1].metadata["title"] == "Second"


def test_csv_loader_uses_row_text_and_metadata(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("title,text\nRow One,Hello world\nRow Two,Goodbye world\n", encoding="utf-8")

    records = load_dataset_source(str(path), chunk_size=None)

    assert len(records) == 2
    assert records[0].doc_type == "csv"
    assert records[0].metadata["row_number"] == 1
    assert records[0].metadata["title"] == "Row One"
    assert "Hello world" in records[0].text


def test_html_loader_falls_back_without_bs4(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><head><title>Demo</title></head><body><h1>Hello</h1><p>World</p></body></html>", encoding="utf-8")

    records = load_dataset_source(str(path), chunk_size=None)

    assert len(records) == 1
    assert records[0].metadata["title"] == "Demo"
    assert "Hello" in records[0].text
    assert "World" in records[0].text


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("bs4") is None,
    reason="BeautifulSoup is not installed",
)
def test_html_loader_with_beautifulsoup_if_available(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><head><title>Soup Demo</title></head><body><p>Soup text</p></body></html>", encoding="utf-8")

    records = load_dataset_source(str(path), chunk_size=None)

    assert records[0].metadata["title"] == "Soup Demo"
    assert "Soup text" in records[0].text


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None
    and __import__("importlib").util.find_spec("pandas") is None,
    reason="Parquet dependencies are not installed",
)
def test_parquet_loader_if_dependency_available(tmp_path):
    path = tmp_path / "records.parquet"

    if __import__("importlib").util.find_spec("pyarrow") is not None:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        table = pa.table({"title": ["Parquet"], "text": ["Parquet body"]})
        pq.write_table(table, path)
    else:
        import pandas as pd  # type: ignore

        pd.DataFrame([{"title": "Parquet", "text": "Parquet body"}]).to_parquet(path)

    records = load_dataset_source(str(path), chunk_size=None)
    assert len(records) == 1
    assert records[0].doc_type == "parquet"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("fitz") is None
    and __import__("importlib").util.find_spec("pdfplumber") is None,
    reason="PDF dependencies are not installed",
)
def test_pdf_loader_if_dependency_available(tmp_path):
    path = tmp_path / "sample.pdf"

    if __import__("importlib").util.find_spec("fitz") is not None:
        import fitz  # type: ignore

        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "PDF sample text")
        document.save(path)
        document.close()
    else:
        pytest.skip("pdfplumber is installed but generating a sample PDF is not supported in this test")

    records = load_dataset_source(str(path), chunk_size=None)
    assert len(records) == 1
    assert records[0].doc_type == "pdf"
    assert "PDF sample text" in records[0].text


def test_dataset_loader_cli_writes_jsonl(tmp_path):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "artifacts" / "ingested.jsonl"
    input_path.write_text("CLI ingestion sample text.", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingest.dataset_loader",
            "--input",
            str(input_path),
            "--out",
            str(output_path),
            "--chunk-size",
            "12",
            "--chunk-overlap",
            "2",
        ],
        cwd="/home/valentin/Documents/GitHub/PPP223",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote" in completed.stdout
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines
    parsed = [json.loads(line) for line in lines]
    assert all("text" in row for row in parsed)
    assert all(row["chunk_id"] for row in parsed)


def test_text_chunker_overlap_validation():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, overlap=10)


def test_repo_loader_still_works_after_dataset_loader_changes():
    loader = Loader(".")
    chunks = loader.process_directory()
    assert chunks
