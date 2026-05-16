import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# Ensure root path is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from model.agent import SLMAgent

BENCHMARK = [
    {
        "question": "What class handles hybrid vector search?",
        "expected_entities": ["HybridRetriever"],
        "expected_files": ["rag/retriever.py"]
    },
    {
        "question": "What method is called to ingest code chunks into the vector database?",
        "expected_entities": ["ingest_chunks"],
        "expected_files": ["rag/retriever.py"]
    },
    {
        "question": "What happens when a Python file has a syntax error during loading?",
        "expected_entities": ["_fallback_tree_sitter_chunk", "SyntaxError"],
        "expected_files": ["ingest/loader.py"]
    },
    {
        "question": "Which file defines the CodeChunk data model?",
        "expected_entities": ["CodeChunk"],
        "expected_files": ["ingest/metadata.py"]
    },
    {
        "question": "What tool does the agent use to find exact string matches in code?",
        "expected_entities": ["grep_search"],
        "expected_files": ["model/tools.py"]
    },
    {
        "question": "What embedding model is used for dense vector search?",
        "expected_entities": ["BAAI/bge-small-en-v1.5", "bge-small-en-v1.5"],
        "expected_files": ["rag/retriever.py"]
    },
    {
        "question": "How does the dependency graph handle methods inside classes?",
        "expected_entities": ["get_chunk_deps", "parent_class", "qualified_name"],
        "expected_files": ["data/dep_graph.py"]
    },
    {
        "question": "What is the maximum number of agent tool-call turns before giving up?",
        "expected_entities": ["max_turns", "8"],
        "expected_files": ["model/agent.py"]
    },
    {
        "question": "What directories does the loader skip when scanning a repo?",
        "expected_entities": ["SKIP_DIRS", "__pycache__", ".git", "venv"],
        "expected_files": ["ingest/loader.py"]
    },
    {
        "question": "What fusion algorithm is used to combine dense and sparse search results?",
        "expected_entities": ["RRF", "FusionQuery"],
        "expected_files": ["rag/retriever.py"]
    }
]

def score_response(response: str, expected_entities: List[str], expected_files: List[str]):
    resp_lower = response.lower()
    
    found_entities = [e for e in expected_entities if e.lower() in resp_lower]
    entity_score = len(found_entities) / len(expected_entities) if expected_entities else 0.0
    
    found_files = [f for f in expected_files if f.lower() in resp_lower]
    file_score = len(found_files) / len(expected_files) if expected_files else 0.0
    
    combined_score = (entity_score + file_score) / 2
    
    return {
        "entity_score": entity_score,
        "file_score": file_score,
        "combined_score": combined_score,
        "matched_entities": found_entities,
        "missed_entities": [e for e in expected_entities if e not in found_entities],
        "matched_files": found_files,
        "missed_files": [f for f in expected_files if f not in found_files]
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent Performance")
    parser.add_argument("--repo", default=".", help="Repo path to index")
    parser.add_argument("--model", default="qwen2.5-coder:3b", help="Ollama model name")
    parser.add_argument("--out", default="eval_report.json", help="Output JSON report path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    repo_path = os.path.abspath(args.repo)
    
    print(f"--- Starting Evaluation on {repo_path} using model {args.model} ---")
    
    # Boot sequence
    print("[1/3] Scanning codebase...")
    loader = Loader(repo_path)
    chunks = loader.process_directory()
    repo_map = RepoMapGenerator().generate_map(chunks)
    
    print(f"[2/3] Ingesting {len(chunks)} chunks...")
    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)
    
    print(f"[3/3] Initializing agent...")
    agent = SLMAgent(repo_map_string=repo_map, model=args.model)
    agent.set_retriever(retriever)
    
    results = []
    total_entity_score = 0.0
    total_file_score = 0.0
    total_combined_score = 0.0

    print("\n--- Running Benchmark ---")
    for i, item in enumerate(BENCHMARK, 1):
        question = item["question"]
        expected_entities = item["expected_entities"]
        expected_files = item["expected_files"]
        
        print(f"Q{i}: {question}")
        response = agent.ask(question)
        
        scores = score_response(response, expected_entities, expected_files)
        
        res = {
            "question": question,
            "response": response,
            "entity_score": scores["entity_score"],
            "file_score": scores["file_score"],
            "combined_score": scores["combined_score"],
            "expected_entities": expected_entities,
            "expected_files": expected_files
        }
        results.append(res)
        
        total_entity_score += scores["entity_score"]
        total_file_score += scores["file_score"]
        total_combined_score += scores["combined_score"]
        
        if args.verbose:
            print(f"\nResponse: {response}")
            print(f"Matched Entities: {scores['matched_entities']}")
            print(f"Missed Entities: {scores['missed_entities']}")
            print(f"Matched Files: {scores['matched_files']}")
            print(f"Missed Files: {scores['missed_files']}")
            print("-" * 40)

    num_q = len(BENCHMARK)
    report = {
        "model": args.model,
        "repo": repo_path,
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "entity_score": total_entity_score / num_q,
            "file_score": total_file_score / num_q,
            "combined_score": total_combined_score / num_q,
            "questions_evaluated": num_q
        },
        "results": results
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "─" * 41)
    for i, res in enumerate(results, 1):
        print(f"Q{i}: {res['combined_score']:.2f} — {res['question']}")
    
    print("─" * 41)
    print(f"Overall combined score: {report['overall']['combined_score']:.2f} / 1.00")
    print(f"Report saved to {args.out}")

if __name__ == "__main__":
    main()
