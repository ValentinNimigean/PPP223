import sys
import os

# Ensure root path is accessible when running from ui/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from model.agent import SLMAgent

def run_cli():
    print("--- SLM Agent CLI Interface ---")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("[1/3] Scanning codebase and parsing AST...")
    loader = Loader(root_dir)
    chunks = loader.process_directory()
    repo_map = RepoMapGenerator().generate_map(chunks)
    
    print(f"[2/3] Ingesting {len(chunks)} logic chunks into Hybrid Qdrant...")
    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)
    
    print("[3/3] Booting Local Agent...")
    agent = SLMAgent(repo_map_string=repo_map)
    agent.set_retriever(retriever)
    
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
