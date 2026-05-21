"""Shared ingestion interfaces for code and general datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class DocumentRecord:
    """Normalized record emitted by dataset loaders and text chunkers."""

    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_type: str = "text"
    created_at: str | None = None
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


DocumentChunk = DocumentRecord


@runtime_checkable
class BaseLoader(Protocol):
    """Interface for dataset loaders that emit normalized document records."""

    def load(self, path_or_uri: str, **kwargs: Any) -> list[DocumentRecord]:
        """Read a dataset source and return normalized document records."""


@runtime_checkable
class ChunkRecord(Protocol):
    """Minimal structural interface for extracted source-code chunks."""

    filepath: str
    chunk_type: str
    name: str
    qualified_name: str | None
    parent_class: str | None
    start_line: int
    end_line: int
    text: str
    signature: str


@runtime_checkable
class LoaderProtocol(Protocol):
    """Interface for repository loaders that emit code chunk records."""

    root_dir: str

    def process_directory(self) -> list[ChunkRecord]:
        """Scan a repository path and return extracted code chunks."""

