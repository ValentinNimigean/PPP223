import os
from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from model.agent import SLMAgent

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- PHASE 1 ---
    print("\n[PHASE 1] Extracting AST Chunks & Generating Repo Map...")
    loader = Loader(root_dir)
    chunks = loader.process_directory()
    
    repo_map_gen = RepoMapGenerator()
    repo_map_string = repo_map_gen.generate_map(chunks)
    
    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)
    
    # --- PHASE 2 ---
    print("\n[PHASE 2] Initializing SLM Agent (Connecting to local Ollama)...")
    agent = SLMAgent(repo_map_string=repo_map_string)
    agent.set_retriever(retriever)
    
    question = "What tool functions are available in the repository? Please grep for def grep_search."
    
    try:
        print("\n--------------------------------------------------")
        response = agent.ask(question)
        print(response)
        print("--------------------------------------------------")
    except Exception as e:
        print(f"\n[!] Failed to connect to local SLM. Please ensure Ollama is running.\nDetails: {e}")

if __name__ == "__main__":
    main()
