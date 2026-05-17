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

# Enable logging
logging.basicConfig(level=logging.INFO)

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

# Let's inspect the call to the client
system_prompt = agent._build_system_prompt()
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
]
tools = agent.tools = from_tools_py = agent._execute_tool_by_name.__globals__['AgentTools'].get_tool_schemas()

print("\n--- System Prompt ---")
print(system_prompt)
print("\n--- Tools ---")
print(json.dumps(tools, indent=2))

response = agent.client.chat.completions.create(
    model=agent.model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print("\n--- Raw Response ---")
msg = response.choices[0].message
print("Content:", repr(msg.content))
print("Tool Calls:", repr(msg.tool_calls))
