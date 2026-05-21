from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List

from ingest.base import ChunkRecord, DocumentRecord
from ingest.metadata import CodeChunk
from rag.base import SearchResult

try:
    from qdrant_client import QdrantClient, models
except Exception:  # pragma: no cover - allows lexical-only operation in constrained envs
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid code retriever with a safe deterministic fallback.

    Primary path:
      - Qdrant in-memory collection
      - fastembed dense model
      - optional fastembed sparse BM25 model
      - Reciprocal Rank Fusion if vectors are available

    Fallback path:
      - local lexical scoring over the ingested CodeChunk objects

    Why the fallback exists:
      qdrant-client[fastembed] may create vector names that differ across
      versions. The older code assumed hard-coded vector names and failed with
      errors like "Dense vector fast-sparse-bm25 is not found in the collection".
      The agent should still answer code questions even if hybrid search fails.
    """

    DENSE_VECTOR_NAME = "fast-bge-small-en-v1.5"
    SPARSE_VECTOR_NAME = "fast-sparse-bm25"

    DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_SPARSE_MODEL = "Qdrant/bm25"

    def __init__(
        self,
        collection_name: str = "repo_chunks",
        use_qdrant: bool = True,
        dense_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
    ):
        self.collection_name = collection_name
        self.use_qdrant = use_qdrant and QdrantClient is not None
        self.dense_model = dense_model
        self.sparse_model = sparse_model

        self.client = None
        self.qdrant_ready = False
        self.sparse_ready = False

        self._records: List[CodeChunk | DocumentRecord] = []
        self._documents: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._doc_tokens: List[Counter[str]] = []
        self._idf: Dict[str, float] = {}

        if not self.use_qdrant:
            logger.warning("Qdrant is disabled or unavailable; using lexical fallback only.")
            return

        try:
            self.client = QdrantClient(":memory:")
            self.client.set_model(dense_model)
            try:
                self.client.set_sparse_model(sparse_model)
                self.sparse_ready = True
            except Exception as exc:
                self.sparse_ready = False
                logger.warning("Could not initialize sparse model %s: %s", sparse_model, exc)
        except Exception as exc:
            logger.warning("Could not initialize Qdrant; using lexical fallback only: %s", exc)
            self.client = None
            self.use_qdrant = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", (text or "").lower())

    @staticmethod
    def _chunk_metadata(chunk: CodeChunk) -> Dict[str, Any]:
        metadata = {
            "filepath": getattr(chunk, "filepath", ""),
            "type": getattr(chunk, "chunk_type", ""),
            "doc_type": "code",
            "name": getattr(chunk, "name", ""),
            "qualified_name": getattr(chunk, "qualified_name", None),
            "parent_class": getattr(chunk, "parent_class", None),
            "start_line": getattr(chunk, "start_line", None),
            "end_line": getattr(chunk, "end_line", None),
            "signature": getattr(chunk, "signature", ""),
            "source": getattr(chunk, "filepath", ""),
        }
        metadata["snippet"] = (getattr(chunk, "text", "") or "")[:240]
        return metadata

    @staticmethod
    def _document_metadata(document: DocumentRecord) -> Dict[str, Any]:
        metadata = dict(getattr(document, "metadata", {}) or {})
        metadata.setdefault("source", getattr(document, "source", ""))
        metadata.setdefault("filepath", getattr(document, "source", ""))
        metadata.setdefault("type", "document")
        metadata.setdefault("doc_type", getattr(document, "doc_type", "text"))
        metadata.setdefault("chunk_id", getattr(document, "chunk_id", None))
        metadata.setdefault("content_hash", metadata.get("content_hash"))
        metadata["id"] = getattr(document, "id", "")
        metadata["snippet"] = (getattr(document, "text", "") or "")[:240]
        return metadata

    @classmethod
    def _record_metadata(cls, record: CodeChunk | DocumentRecord) -> Dict[str, Any]:
        if hasattr(record, "filepath") and hasattr(record, "chunk_type"):
            return cls._chunk_metadata(record)  # type: ignore[arg-type]
        return cls._document_metadata(record)  # type: ignore[arg-type]

    def _rebuild_local_index(self) -> None:
        self._documents = [getattr(record, "text", "") or "" for record in self._records]
        self._metadata = [self._record_metadata(record) for record in self._records]

        self._doc_tokens = []
        df: Counter[str] = Counter()

        for doc, meta in zip(self._documents, self._metadata):
            searchable = " ".join(
                [
                    doc,
                    str(meta.get("filepath") or ""),
                    str(meta.get("name") or ""),
                    str(meta.get("qualified_name") or ""),
                    str(meta.get("signature") or ""),
                    str(meta.get("type") or ""),
                    str(meta.get("source") or ""),
                    str(meta.get("title") or ""),
                    str(meta.get("row_number") or ""),
                    str(meta.get("chunk_id") or ""),
                ]
            )
            counts = Counter(self._tokenize(searchable))
            self._doc_tokens.append(counts)
            for token in counts:
                df[token] += 1

        n_docs = max(len(self._documents), 1)
        self._idf = {
            token: math.log((n_docs + 1) / (freq + 0.5)) + 1.0
            for token, freq in df.items()
        }

    def ingest_chunks(self, chunks: List[CodeChunk] | List[ChunkRecord]) -> None:
        self._records = list(chunks or [])
        self._rebuild_local_index()

        if not self._records:
            print("No chunks to ingest.")
            return

        print(f"Ingesting {len(self._records)} elements into retriever index...")

        if not self.use_qdrant or self.client is None:
            print("Qdrant unavailable; lexical fallback index is ready.")
            return

        try:
            self.client.add(
                collection_name=self.collection_name,
                documents=self._documents,
                metadata=self._metadata,
                ids=list(range(len(self._records))),
            )
            self.qdrant_ready = True
            print("Qdrant ingestion complete.")
        except Exception as exc:
            self.qdrant_ready = False
            logger.warning("Qdrant ingestion failed; using lexical fallback only: %s", exc)
            print(f"Qdrant ingestion failed; lexical fallback index is ready: {exc}")

    def ingest_documents(self, documents: List[DocumentRecord]) -> None:
        self._records = list(documents or [])
        self._rebuild_local_index()

        if not self._records:
            print("No documents to ingest.")
            return

        print(f"Ingesting {len(self._records)} documents into retriever index...")

        if not self.use_qdrant or self.client is None:
            print("Qdrant unavailable; lexical fallback index is ready.")
            return

        try:
            self.client.add(
                collection_name=self.collection_name,
                documents=self._documents,
                metadata=self._metadata,
                ids=list(range(len(self._records))),
            )
            self.qdrant_ready = True
            print("Qdrant ingestion complete.")
        except Exception as exc:
            self.qdrant_ready = False
            logger.warning("Qdrant ingestion failed; using lexical fallback only: %s", exc)
            print(f"Qdrant ingestion failed; lexical fallback index is ready: {exc}")

    def _format_qdrant_point(self, point: Any) -> SearchResult:
        payload = point.payload or {}
        doc_text = payload.get("document", "")

        # qdrant-client stores user metadata next to the document payload.
        metadata = {k: v for k, v in payload.items() if k != "document"}

        return {
            "document": doc_text,
            "snippet": payload.get("snippet", doc_text[:240]),
            "metadata": metadata,
            "score": float(getattr(point, "score", 0.0) or 0.0),
        }

    def _search_qdrant_hybrid(self, query: str, limit: int) -> List[SearchResult]:
        if not self.qdrant_ready or self.client is None or models is None:
            return []

        if self.sparse_ready:
            points = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=query, using=self.SPARSE_VECTOR_NAME, limit=limit),
                    models.Prefetch(query=query, using=self.DENSE_VECTOR_NAME, limit=limit),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            ).points
            return [self._format_qdrant_point(point) for point in points]

        # Dense-only fallback through qdrant-client's high-level text query.
        # Keep this conservative because qdrant-client APIs vary by version.
        points = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=limit,
        )
        return [self._format_qdrant_point(point) for point in points]

    def _field_boost(self, query_text: str, query_tokens: List[str], meta: Dict[str, Any], doc: str) -> float:
        score = 0.0

        name = str(meta.get("name") or "").lower()
        qualified_name = str(meta.get("qualified_name") or "").lower()
        filepath = str(meta.get("filepath") or "").lower()
        signature = str(meta.get("signature") or "").lower()
        source = str(meta.get("source") or "").lower()
        title = str(meta.get("title") or "").lower()
        chunk_id = str(meta.get("chunk_id") or "").lower()
        doc_lower = (doc or "").lower()

        if not query_text:
            return 0.0

        if query_text in name:
            score += 18.0
        if query_text in qualified_name:
            score += 14.0
        if query_text in filepath:
            score += 10.0
        if query_text in signature:
            score += 8.0
        if query_text in source:
            score += 7.0
        if query_text in title:
            score += 7.0
        if query_text in chunk_id:
            score += 5.0
        if query_text in doc_lower:
            score += 6.0

        for token in query_tokens:
            if not token:
                continue
            if token in name:
                score += 7.0
            if token in qualified_name:
                score += 5.0
            if token in filepath:
                score += 4.0
            if token in signature:
                score += 3.0
            if token in source:
                score += 3.0
            if token in title:
                score += 3.0
            if token in chunk_id:
                score += 2.0

        return score

    def _search_local(self, query: str, limit: int) -> List[SearchResult]:
        if not self._records:
            return []

        query_text = " ".join(self._tokenize(query))
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        avg_len = max(sum(sum(c.values()) for c in self._doc_tokens) / max(len(self._doc_tokens), 1), 1.0)
        scored: List[SearchResult] = []

        for idx, (doc, meta, doc_counts) in enumerate(zip(self._documents, self._metadata, self._doc_tokens)):
            score = self._field_boost(query_text, query_tokens, meta, doc)

            # BM25-ish lexical score without external dependencies.
            doc_len = max(sum(doc_counts.values()), 1)
            k1 = 1.5
            b = 0.75

            for token, q_count in query_counts.items():
                tf = doc_counts.get(token, 0)
                if tf <= 0:
                    continue
                idf = self._idf.get(token, 1.0)
                denom = tf + k1 * (1.0 - b + b * (doc_len / avg_len))
                score += idf * ((tf * (k1 + 1.0)) / denom) * q_count

            if score > 0:
                scored.append(
                    {
                        "document": doc,
                        "snippet": doc[:240],
                        "metadata": meta,
                        "score": float(score),
                    }
                )

        scored.sort(key=lambda row: row["score"], reverse=True)

        if scored:
            return scored[:limit]

        # Last-resort behavior: return a few chunks so the caller has some context.
        return [
            {
                "document": doc,
                "snippet": doc[:240],
                "metadata": meta,
                "score": 0.0,
            }
            for doc, meta in list(zip(self._documents, self._metadata))[:limit]
        ]

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        query = query or ""
        limit = max(int(limit or 5), 1)

        if self.qdrant_ready:
            try:
                qdrant_results = self._search_qdrant_hybrid(query, limit)
                if qdrant_results:
                    return qdrant_results
            except Exception as exc:
                logger.warning("Qdrant search failed; falling back to lexical search: %s", exc)

        return self._search_local(query, limit)

    @staticmethod
    def smoke_test() -> None:
        chunks = [
            CodeChunk(
                filepath="rag/retriever.py",
                chunk_type="class",
                name="HybridRetriever",
                qualified_name="HybridRetriever",
                start_line=5,
                end_line=60,
                text="class HybridRetriever:\n    def search(self, query):\n        return FusionQuery(fusion=Fusion.RRF)",
                signature="class HybridRetriever",
            ),
            CodeChunk(
                filepath="ingest/metadata.py",
                chunk_type="class",
                name="CodeChunk",
                qualified_name="CodeChunk",
                start_line=4,
                end_line=15,
                text="class CodeChunk(BaseModel):\n    filepath: str\n    name: str",
                signature="class CodeChunk",
            ),
            CodeChunk(
                filepath="model/tools.py",
                chunk_type="method",
                name="grep_search",
                qualified_name="AgentTools.grep_search",
                parent_class="AgentTools",
                start_line=62,
                end_line=101,
                text="def grep_search(pattern: str, directory: str = '.') -> str:\n    return matches",
                signature="def grep_search(pattern: str, directory: str = '.') -> str",
            ),
        ]

        retriever = HybridRetriever(use_qdrant=False)
        retriever.ingest_chunks(chunks)

        results = retriever.search("What class handles hybrid vector search?", limit=2)
        assert results, "Expected local search results"
        assert results[0]["metadata"]["filepath"] == "rag/retriever.py"
        assert results[0]["metadata"]["name"] == "HybridRetriever"

        results = retriever.search("exact string pattern search tool", limit=2)
        assert any(r["metadata"]["name"] == "grep_search" for r in results)

        print("HybridRetriever smoke_test OK")


if __name__ == "__main__":
    HybridRetriever.smoke_test()
