import logging
import os
import sys
import json

# Ensure root path is accessible
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from model.agent import SLMAgent

# Set up logging to intercept everything loggers emit
class LogInterceptor(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
    def emit(self, record):
        self.logs.append(self.format(record))

interceptor = LogInterceptor()
logging.getLogger().addHandler(interceptor)
logging.getLogger().setLevel(logging.DEBUG)

repo_path = os.path.abspath(".")
loader = Loader(repo_path)
chunks = loader.process_directory()
repo_map = RepoMapGenerator().generate_map(chunks)

retriever = HybridRetriever()
retriever.ingest_chunks(chunks)

agent = SLMAgent(repo_map_string=repo_map, model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b"))
agent.set_retriever(retriever)

question = "What class handles hybrid vector search?"

with open("scratch/test_agent_all_logs_output.txt", "w", encoding="utf-8") as out_file:
    out_file.write(f"Asking: {question}\n")
    response = agent.ask(question)
    out_file.write(f"\nFinal Response:\n{response}\n")
    
    out_file.write("\n--- Raw Logs ---\n")
    for log in interceptor.logs:
        out_file.write(log + "\n")
