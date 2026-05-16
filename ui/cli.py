import sys
import os
import logging

# Ensure root path is accessible when running from ui/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from model.agent import SLMAgent

import argparse

def run_cli():
    logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="SLM Agent CLI Interface")
    parser.add_argument("--repo", default=".", help="Path to the Python repo to analyze.")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo)
    if not os.path.isdir(repo_path):
        print(f"Error: {repo_path} is not a valid directory.")
        sys.exit(1)

    print(f"--- SLM Agent CLI Interface: Analyzing {repo_path} ---")
    
    print("[1/3] Scanning codebase and parsing AST...")
    loader = Loader(repo_path)
    chunks = loader.process_directory()
    repo_map = RepoMapGenerator().generate_map(chunks)
    
    print(f"[2/3] Ingesting {len(chunks)} logic chunks into Hybrid Qdrant...")
    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)
    
    print("[3/3] Booting Local Agent... (qwen2.5-coder:3b via Ollama)")
    agent = SLMAgent(repo_map_string=repo_map, hallucination_check=True)
    agent.set_retriever(retriever)
    agent.set_chunks(chunks)
    
    print("\n[READY] Start asking questions about your codebase.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("You> ")
            if query.strip().lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            response = agent.ask(query)
            print(f"\nAgent> {response}\n")
            
        except KeyboardInterrupt:
            print("\nExiting CLI.")
            break
        except Exception as e:
            print(f"\n[System Error]: {e}")

if __name__ == "__main__":
    run_cli()
