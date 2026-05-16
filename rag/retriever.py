from qdrant_client import QdrantClient, models
from typing import List, Dict, Any
from ingest.metadata import CodeChunk

class HybridRetriever:
    DENSE_VECTOR_NAME = "fast-bge-small-en-v1.5"
    SPARSE_VECTOR_NAME = "fast-sparse-bm25"

    def __init__(self):
        # We use an in-memory database for fast prototyping.
        # This will automatically utilize fastembed for embeddings.
        self.client = QdrantClient(":memory:")
        self.collection_name = "repo_chunks"
        
        # Setup the dense embedding model
        self.client.set_model("BAAI/bge-small-en-v1.5")
        
        # Setup the sparse embedding model for BM25 hybrid search
        try:
            self.client.set_sparse_model("Qdrant/bm25")
        except Exception as e:
            print(f"Warning: Could not load sparse model -> {e}")

    def ingest_chunks(self, chunks: List[CodeChunk]):
        if not chunks:
            print("No chunks to ingest.")
            return
            
        documents = [chunk.text for chunk in chunks]
        metadata = [{"filepath": c.filepath, "type": c.chunk_type, "name": c.name} for c in chunks]
        ids = list(range(len(chunks)))
        
        print(f"Ingesting {len(chunks)} elements into Qdrant vector database...")
        self.client.add(
            collection_name=self.collection_name,
            documents=documents,
            metadata=metadata,
            ids=ids
        )
        print("Ingestion complete.")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=query, using=self.SPARSE_VECTOR_NAME, limit=limit),
                models.Prefetch(query=query, using=self.DENSE_VECTOR_NAME, limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        ).points

        extracted = []
        for result in results:
            payload = result.payload or {}
            doc_text = payload.get("document", "")
            metadata = {k: v for k, v in payload.items() if k != "document"}
            extracted.append({"document": doc_text, "metadata": metadata, "score": result.score})
        return extracted

    @staticmethod
    def smoke_test():
        """
        Validates retriever functionality with dummy data.
        """
        retriever = HybridRetriever()
        
        # Ingest 3 hardcoded CodeChunk fixtures
        chunks = [
            CodeChunk(
                filepath="src/main.py",
                chunk_type="function",
                name="main",
                start_line=1,
                end_line=10,
                text="def main():\n    print('hello world')",
                signature="def main()"
            ),
            CodeChunk(
                filepath="src/utils.py",
                chunk_type="function",
                name="helper",
                start_line=5,
                end_line=15,
                text="def helper(x):\n    return x * 2",
                signature="def helper(x)"
            ),
            CodeChunk(
                filepath="src/app.py",
                chunk_type="class",
                name="App",
                start_line=1,
                end_line=50,
                text="class App:\n    def run(self): pass",
                signature="class App"
            )
        ]
        
        retriever.ingest_chunks(chunks)
        
        # Perform a search
        results = retriever.search("hello", limit=2)
        
        # Verify results
        assert len(results) > 0, "Retriever should return at least one result"
        for res in results:
            assert "document" in res, "Result missing 'document' key"
            assert "metadata" in res, "Result missing 'metadata' key"
            assert "score" in res, "Result missing 'score' key"
            
        print("Retriever OK")
