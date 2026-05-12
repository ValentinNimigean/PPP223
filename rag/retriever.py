from qdrant_client import QdrantClient, models
from typing import List, Dict, Any
from ingest.metadata import CodeChunk

class HybridRetriever:
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
        """
        Executes a hybrid search (Dense + Sparse) against the ingested chunks.
        """
        # Utilizing client.query_points with FusionQuery performs a Hybrid Search with RRF
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=models.SparseVectorQuery(name="bm25", text=query), limit=limit),
                models.Prefetch(query=models.DenseVectorQuery(name="dense", text=query), limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        ).points
        
        extracted = []
        for result in results:
            extracted.append({
                "document": result.document,
                "metadata": result.metadata,
                "score": result.score
            })
        return extracted
