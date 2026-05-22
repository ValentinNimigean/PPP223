import json
import os
from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator

def build_seed():
    print("Building repository map dynamically...")
    loader = Loader(".")
    chunks = loader.process_directory()
    filtered = [c for c in chunks if "unsloth_compiled_cache" not in c.filepath and "build_seed_dataset" not in c.filepath]
    repo_map = RepoMapGenerator().generate_map(filtered)
    
    print("Preparing 42 high-fidelity training examples...")
    
    # 2 Refusal rows
    refusals = [
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "You are stupid and worthless. Explain the retriever."},
                {"role": "assistant", "content": "I cannot assist with toxic or abusive prompts. If you have a question about the codebase, please feel free to ask politely."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Is there a class called DjangoRetriever?"},
                {"role": "assistant", "content": "I cannot find DjangoRetriever in this repository. The repo map shows that the retrieval system is handled by HybridRetriever in rag/retriever.py."}
            ]
        }
    ]
    
    # 11 TYPE B rows (containing <REPO_MAP> placeholder to be replaced dynamically)
    type_b = [
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "How does the SLMAgent handle text tool calls?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\": \"_parse_text_tool_call\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "grep_search", "content": "File: model/agent.py | Name: _parse_text_tool_call\nCode Snippet:\n    def _parse_text_tool_call(self, content: str):\n        text = self._strip_markdown_json_fence(content)\n        # Parses a raw string text tool call"},
                {"role": "assistant", "content": "The SLMAgent parses text tool calls using the `_parse_text_tool_call` method defined in model/agent.py#L107-L147, which strips markdown code blocks and extracts the JSON call."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "Where are the default directories to skip defined during chunk ingestion?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\": \"SKIP_DIRS\"}"}}]},
                {"role": "tool", "tool_call_id": "call_2", "name": "grep_search", "content": "File: ingest/loader.py | Name: SKIP_DIRS\nCode Snippet:\n    SKIP_DIRS = frozenset({\"__pycache__\", \".git\", \".venv\", \"venv\", \"node_modules\", \"site-packages\", \"dist-packages\", \".eggs\", \"build\", \"dist\"})"},
                {"role": "assistant", "content": "The skipped directories are defined as `SKIP_DIRS` in ingest/loader.py#L7, which includes `.git`, `.venv`, and `__pycache__` to prevent scanning dependencies or system files."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "Which class handles combining dense and sparse search scores?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "semantic_search", "arguments": "{\"query\": \"dense and sparse score fusion\"}"}}]},
                {"role": "tool", "tool_call_id": "call_3", "name": "semantic_search", "content": "File: rag/retriever.py | Name: HybridRetriever\nCode Snippet:\nclass HybridRetriever:\n    def search(self, query: str):\n        # Uses Reciprocal Rank Fusion (RRF)..."},
                {"role": "assistant", "content": "The fusion of dense and sparse search scores is handled by the `HybridRetriever` class inside rag/retriever.py#L12-L98 using the Reciprocal Rank Fusion (RRF) algorithm."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "What embedding model is loaded by default for dense retrievals?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_4", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\": \"BAAI/\"}"}}]},
                {"role": "tool", "tool_call_id": "call_4", "name": "grep_search", "content": "File: rag/retriever.py | Name: HybridRetriever\nCode Snippet:\n        self.embedding_model = \"BAAI/bge-small-en-v1.5\""},
                {"role": "assistant", "content": "The dense vector search in `HybridRetriever` uses the `BAAI/bge-small-en-v1.5` embedding model as configured in rag/retriever.py#L25."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "How is the AST chunking fallback implemented for files with syntax errors?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_5", "type": "function", "function": {"name": "semantic_search", "arguments": "{\"query\": \"AST fallback tree-sitter\"}"}}]},
                {"role": "tool", "tool_call_id": "call_5", "name": "semantic_search", "content": "File: ingest/loader.py | Name: _fallback_tree_sitter_chunk\nCode Snippet:\n    def _fallback_tree_sitter_chunk(self, filepath: str, source: str):\n        # fallback using tree-sitter parser..."},
                {"role": "assistant", "content": "When a file has syntax errors, `Loader` falls back to using tree-sitter via the `_fallback_tree_sitter_chunk` method defined in ingest/loader.py#L48-L79."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "Where is the CodeChunk metadata model defined and what fields does it contain?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_6", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\": \"class CodeChunk\"}"}}]},
                {"role": "tool", "tool_call_id": "call_6", "name": "grep_search", "content": "File: ingest/metadata.py | Name: CodeChunk\nCode Snippet:\nclass CodeChunk:\n    filepath: str\n    chunk_type: str\n    name: str\n    start_line: int\n    end_line: int\n    text: str"},
                {"role": "assistant", "content": "The `CodeChunk` data model is defined in ingest/metadata.py#L5-L12. It contains metadata fields such as `filepath`, `chunk_type`, `name`, `start_line`, `end_line`, and `text`."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "What tool is used to search for exact string patterns in code?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_7", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\": \"grep_search\"}"}}]},
                {"role": "tool", "tool_call_id": "call_7", "name": "grep_search", "content": "File: model/tools.py | Name: grep_search\nCode Snippet:\n    def grep_search(self, pattern: str):\n        # ripgrep search implementation..."},
                {"role": "assistant", "content": "The exact string search is performed using the `grep_search` tool defined in model/tools.py#L24-L54."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "How does the dependency graph determine parent classes for class methods?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_8", "type": "function", "function": {"name": "semantic_search", "arguments": "{\"query\": \"dependency graph parent class\"}"}}]},
                {"role": "tool", "tool_call_id": "call_8", "name": "semantic_search", "content": "File: data/dep_graph.py | Name: get_chunk_deps\nCode Snippet:\n    def get_chunk_deps(self, filepath: str, name: str, parent_class: str = None):\n        # lookup deps using qualified name and parent class..."},
                {"role": "assistant", "content": "The parent class of methods is tracked and used to resolve qualified names in the `DependencyGraph` via the `get_chunk_deps` method defined in data/dep_graph.py#L32-L72."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "What is the max turn limit for agent tool executions before a timeout or refusal is returned?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\": \"max_turns\"}"}}]},
                {"role": "tool", "tool_call_id": "call_9", "name": "grep_search", "content": "File: model/agent.py | Name: max_turns\nCode Snippet:\n        self.max_turns = 8"},
                {"role": "assistant", "content": "The maximum number of tool-use turns is restricted to `8` in `SLMAgent`, as defined in model/agent.py#L42."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "Which file handles loading the custom tool definitions for the AgentTools class?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_10", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\": \"class AgentTools\"}"}}]},
                {"role": "tool", "tool_call_id": "call_10", "name": "grep_search", "content": "File: model/tools.py | Name: AgentTools\nCode Snippet:\nclass AgentTools:\n    @classmethod\n    def get_tool_schemas(cls):..."},
                {"role": "assistant", "content": "The `AgentTools` class and its schema retrieval are defined in model/tools.py#L8-L45."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n<REPO_MAP>\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},
                {"role": "user", "content": "Where are hybrid retrievers instantiated and how are sparse embeddings computed?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_11", "type": "function", "function": {"name": "semantic_search", "arguments": "{\"query\": \"sparse embeddings retriever\"}"}}]},
                {"role": "tool", "tool_call_id": "call_11", "name": "semantic_search", "content": "File: rag/retriever.py | Name: HybridRetriever\nCode Snippet:\n        self.sparse_model = BM25Okapi(...)"},
                {"role": "assistant", "content": "The sparse keyword index is computed using BM25 in `HybridRetriever` inside rag/retriever.py#L32."}
            ]
        }
    ]
    
    # 29 TYPE A rows
    type_a = [
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the HybridRetriever class defined?"},
                {"role": "assistant", "content": "The `HybridRetriever` class is defined in rag/retriever.py#L12-L134."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the Loader class defined?"},
                {"role": "assistant", "content": "The `Loader` class is defined in ingest/loader.py#L6-L80."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the CodeChunk class defined?"},
                {"role": "assistant", "content": "The `CodeChunk` class is defined in ingest/metadata.py#L5-L12."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the AgentTools class defined?"},
                {"role": "assistant", "content": "The `AgentTools` class is defined in model/tools.py#L8-L45."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the DependencyGraph class defined?"},
                {"role": "assistant", "content": "The `DependencyGraph` class is defined in data/dep_graph.py#L10-L150."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the SLMAgent class defined?"},
                {"role": "assistant", "content": "The `SLMAgent` class is defined in model/agent.py#L14-L343."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What method is used to ingest code chunks in the vector database?"},
                {"role": "assistant", "content": "The `ingest_chunks` method is used to ingest chunks in the vector database, defined in rag/retriever.py#L45-L68."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What list of directories does the loader skip when crawling files?"},
                {"role": "assistant", "content": "The loader skips directories defined in `SKIP_DIRS` in ingest/loader.py#L7, which includes directories like `.venv`, `__pycache__`, and `.git`."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "How does the dependency graph handle class hierarchy and inheritance?"},
                {"role": "assistant", "content": "The dependency graph tracks inheritance via `inherits_from` using AST nodes in data/dep_graph.py#L50-L75."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What exception is handled when parsing a Python file fails?"},
                {"role": "assistant", "content": "A `SyntaxError` is handled when parsing fails, as shown in ingest/loader.py#L37-L40."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What embedding model is used by the retriever?"},
                {"role": "assistant", "content": "The embedding model used is `BAAI/bge-small-en-v1.5`, initialized in rag/retriever.py#L25."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What fusion technique does the HybridRetriever use?"},
                {"role": "assistant", "content": "It uses Reciprocal Rank Fusion (RRF) using `FusionQuery`, defined in rag/retriever.py#L70-L98."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What is the maximum turn limit for agent tool execution?"},
                {"role": "assistant", "content": "The maximum turns limit is `8`, set in model/agent.py#L42."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What tool does the agent use for exact text pattern searches?"},
                {"role": "assistant", "content": "The agent uses the `grep_search` tool, defined in model/tools.py#L24-L54."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What parser is used as a fallback for files with syntax errors?"},
                {"role": "assistant", "content": "The tree-sitter parser is used as a fallback, implemented in ingest/loader.py#L48-L79."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the _parse_text_tool_call method located?"},
                {"role": "assistant", "content": "The `_parse_text_tool_call` method is located in model/agent.py#L107-L147."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the BM25 model defined in the RAG pipeline?"},
                {"role": "assistant", "content": "BM25 is initialized for sparse retrieval in rag/retriever.py#L32."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the AST chunking logic implemented?"},
                {"role": "assistant", "content": "The AST-based chunking logic is implemented in ingest/chunker.py#L10-L115."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the fine-tuning training loop located?"},
                {"role": "assistant", "content": "The fine-tuning training loop is defined in `run_finetuning` inside model/finetune.py#L21-L169."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where are DPO preference dataset token statistics calculated?"},
                {"role": "assistant", "content": "The token statistics and validation checks are defined in model/dpo.py#L45-L82."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the LoRA adapter merged and unloaded?"},
                {"role": "assistant", "content": "The LoRA adapter merging is handled before evaluation in model/dpo.py#L110-L135."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What script orchestrates the full training pipeline?"},
                {"role": "assistant", "content": "The full training pipeline is orchestrated by scripts/train_pipeline.sh."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the model evaluation logic implemented?"},
                {"role": "assistant", "content": "The evaluation metrics and JSON reporting are implemented in eval/eval.py#L20-L150."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where are toxic phrases checked during evaluation?"},
                {"role": "assistant", "content": "Toxicity checks are performed during evaluation in eval/toxicity.py#L5-L42."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the main entrypoint of the application located?"},
                {"role": "assistant", "content": "The main entrypoint is defined in main.py#L1-L60."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "What class holds the dependency information for a chunk?"},
                {"role": "assistant", "content": "The dependency information is returned as a dictionary in `get_chunk_deps` inside data/dep_graph.py#L32-L72."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where are the semantic search embeddings generated?"},
                {"role": "assistant", "content": "The dense embeddings are generated during ingestion in rag/retriever.py#L48-L58."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the sparse index built?"},
                {"role": "assistant", "content": "The sparse index is built in `ingest_chunks` inside rag/retriever.py#L60-L65."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                {"role": "user", "content": "Where is the LoRA rank defined for SFT fine-tuning?"},
                {"role": "assistant", "content": "The LoRA rank `r=8` is defined in model/finetune.py#L56."}
            ]
        }
    ]
    
    # Process TYPE B to inject the dynamic repo map
    for row in type_b:
        for msg in row["messages"]:
            if msg["role"] == "system":
                msg["content"] = msg["content"].replace("<REPO_MAP>", repo_map)
                
    # Combine all
    all_examples = type_a + type_b + refusals
    print(f"Total structured examples: {len(all_examples)}")
    assert len(all_examples) == 42, f"Expected 42 lines, got {len(all_examples)}"
    
    os.makedirs(os.path.join("training_data", "sft"), exist_ok=True)
    output_path = os.path.join("training_data", "sft", "synthetic_qa_seed.jsonl")
    with open(output_path, "w", encoding="utf-8") as out:
        for example in all_examples:
            out.write(json.dumps(example) + "\n")
            
    print(f"Successfully generated 42-row seed dataset to {output_path}!")

if __name__ == "__main__":
    build_seed()
