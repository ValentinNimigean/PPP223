"""Shared retrieval interfaces."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable

from ingest.base import ChunkRecord, DocumentRecord


class SearchResult(TypedDict):
    """Normalized retrieval row returned to agents and evaluators."""

    document: str
    snippet: str
    metadata: dict[str, Any]
    score: float


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Interface for code retrievers used by agents and evaluators."""

    def ingest_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Index repository chunks for later retrieval."""

    def ingest_documents(self, documents: list[DocumentRecord]) -> None:
        """Index normalized document records for later retrieval."""

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Return ranked chunks relevant to a user query."""
