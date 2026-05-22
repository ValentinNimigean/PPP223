import os
import sys

# Ensure root path is accessible
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from qdrant_client import QdrantClient

def main():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(repo_dir, ".fastembed_cache")
    
    print(f"Initializing Qdrant Client to download models into: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        client = QdrantClient(":memory:")
        
        print("\n[1/2] Downloading dense model: BAAI/bge-small-en-v1.5 ...")
        client.set_model("BAAI/bge-small-en-v1.5", cache_dir=cache_dir)
        
        print("\n[2/2] Downloading sparse model: Qdrant/bm25 ...")
        client.set_sparse_model("Qdrant/bm25", cache_dir=cache_dir)
        
        print(f"\nDownload completed successfully! Models are stored in {cache_dir}")
        print("You can now run in offline/air-gapped environments.")
        
    except Exception as e:
        print(f"\nError occurred during model download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
