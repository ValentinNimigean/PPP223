import os
import sys
import logging
from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from model.agent import SLMAgent

import argparse

def main():
    logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="SLM Agent Demo")
    parser.add_argument("--repo", default=".", help="Path to the Python repo to analyze.")
    parser.add_argument("--question", default="What tool functions are available in the repository? Please grep for def grep_search.", help="Question for the agent.")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo)
    if not os.path.isdir(repo_path):
        print(f"Error: {repo_path} is not a valid directory.")
        sys.exit(1)

    # --- PHASE 1 ---
    print(f"\n[PHASE 1] Extracting AST Chunks from {repo_path} & Generating Repo Map...")
    loader = Loader(repo_path)
    chunks = loader.process_directory()
    
    repo_map_gen = RepoMapGenerator()
    repo_map_string = repo_map_gen.generate_map(chunks)
    
    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)
    
    # --- PHASE 2 ---
    print("\n[PHASE 2] Initializing SLM Agent (Connecting to local Ollama)...")
    agent = SLMAgent(repo_map_string=repo_map_string)
    agent.set_retriever(retriever)
    
    try:
        print("\n--------------------------------------------------")
        print(f"Question: {args.question}")
        response = agent.ask(args.question)
        print(f"Response: {response}")
        print("--------------------------------------------------")
    except Exception as e:
        print(f"\n[!] Failed to connect to local SLM. Please ensure Ollama is running.\nDetails: {e}")

if __name__ == "__main__":
    main()
