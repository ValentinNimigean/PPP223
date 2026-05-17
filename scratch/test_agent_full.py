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
print("Scanning codebase...")
loader = Loader(repo_path)
chunks = loader.process_directory()
repo_map = RepoMapGenerator().generate_map(chunks)

print("Ingesting chunks...")
retriever = HybridRetriever()
retriever.ingest_chunks(chunks)

print("Initializing agent...")
agent = SLMAgent(repo_map_string=repo_map, model="qwen2.5-coder:3b")
agent.set_retriever(retriever)

question = "What class handles hybrid vector search?"
print(f"\nAsking: {question}")
response = agent.ask(question)
print(f"\nFinal Response:\n{response}")

print("\n--- Agent Logs ---")
for log in interceptor.logs:
    if "[Agent" in log or "Executing" in log or "Tool" in log or "http" in log or "Request" in log:
        print(log)
