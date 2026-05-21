import argparse

from ingest.dataset_loader import load_dataset_source
from rag.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Index an arbitrary dataset into the HybridRetriever pipeline.")
    parser.add_argument("--input", required=True, help="Dataset path or URI to ingest.")
    parser.add_argument("--collection", default="dataset_chunks", help="Qdrant collection name.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for dataset text splitting.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for dataset text splitting.")
    parser.add_argument("--source-type", default=None, help="Optional source type override such as csv or jsonl.")
    parser.add_argument("--encoding", default="utf-8", help="Text encoding for text-like inputs.")
    parser.add_argument("--query", default=None, help="Optional query to run after indexing.")
    args = parser.parse_args()

    documents = load_dataset_source(
        args.input,
        source_type=args.source_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        encoding=args.encoding,
    )

    retriever = HybridRetriever(collection_name=args.collection)
    retriever.ingest_documents(documents)

    print(f"Indexed {len(documents)} documents into collection '{args.collection}'.")

    if args.query:
        results = retriever.search(args.query, limit=5)
        for index, result in enumerate(results, start=1):
            meta = result.get("metadata", {})
            print(
                f"[{index}] score={result.get('score', 0.0):.4f} "
                f"source={meta.get('source') or meta.get('filepath')} "
                f"doc_type={meta.get('doc_type')} "
                f"row={meta.get('row_number')} "
                f"chunk_id={meta.get('chunk_id')}"
            )
            print(result.get("snippet", ""))
            print()


if __name__ == "__main__":
    main()
