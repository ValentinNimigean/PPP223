import json
import os
import random
import re
import sys
from typing import Optional

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from ingest.loader import Loader
from data.dep_graph import DependencyGraph

TEACHER_SYSTEM = """You generate high-quality SFT training data for a Python code-understanding agent.

The target model is a small Qwen2.5-Coder model. The model must learn to answer repository questions with grounded file paths and line ranges.

You must output raw JSONL only: one valid JSON object per line, no markdown, no preamble.

Allowed row shapes:

TYPE A — direct grounded Q&A:
{"messages":[{"role":"system","content":"You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},{"role":"user","content":"<question>"},{"role":"assistant","content":"<answer with exact file path and line range>"}]}

TYPE B — tool-use trajectory:
{"messages":[{"role":"system","content":"You are a Python Expert Assistant. You have tools to explore code.\\nHere is the exact repository layout:\\n<REPO_MAP>\\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."},{"role":"user","content":"<question>"},{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"grep_search","arguments":"{\\"pattern\\":\\"<exact symbol or regex>\\",\\"directory\\":\\".\\"}"}}]},{"role":"tool","tool_call_id":"call_1","name":"grep_search","content":"<realistic tool result copied from the provided chunk/context>"},{"role":"assistant","content":"<final grounded answer with exact file path and line range>"}]}

Use grep_search when the question asks for exact names, classes, methods, constants, imports, or file definitions.
Use semantic_search when the question asks about behavior, architecture, flow, fallback logic, or conceptual implementation.
Use direct TYPE A answers for obvious code-location facts visible in the chunk.

Strict rules:
- Every final assistant answer must cite the exact file path.
- Every final assistant answer must include a line range if line information is available.
- Never invent classes, methods, files, imports, or line numbers.
- Never make the final assistant answer raw JSON.
- Tool-call JSON is allowed only inside assistant.tool_calls, not inside assistant.content.
- If the answer cannot be grounded in the provided chunk/dependency facts, produce a refusal: "I cannot find <entity> in this repository from the provided context."
- Prefer short answers. One or two sentences is ideal.
- Generate a mix of direct Q&A, tool-use rows, no-match rows, and edge-case rows.
"""


def _message_roles(messages):
    return [m.get("role") for m in messages if isinstance(m, dict)]


def _looks_like_raw_json_answer(text: str) -> bool:
    text = (text or "").strip()
    return text.startswith("{") or text.startswith("```json") or text.startswith("```")


def validate_training_row(obj, allowed_files):
    if not isinstance(obj, dict) or "messages" not in obj:
        return False, "missing messages"

    messages = obj["messages"]
    if not isinstance(messages, list) or len(messages) < 3:
        return False, "messages too short"

    roles = _message_roles(messages)
    if "user" not in roles or "assistant" not in roles:
        return False, "missing user or assistant"

    final_assistant = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            final_assistant = msg
            break

    if final_assistant is None:
        return False, "missing final assistant content"

    content = str(final_assistant.get("content") or "").strip()
    if not content:
        return False, "empty final answer"

    if _looks_like_raw_json_answer(content):
        return False, "raw json final answer"

    mentioned_files = set(re.findall(r"[A-Za-z0-9_./-]+\.py", content))
    for file_path in mentioned_files:
        normalized = file_path.lstrip("./")
        if normalized not in allowed_files and file_path not in allowed_files:
            return False, f"mentions unknown file {file_path}"

    return True, "ok"


def generate_synthetic_data(repo_root: str, output_file: str, model: str = "gpt-4o", repo_map_string: str = "", examples_per_chunk: int = 2, max_chunks: Optional[int] = None):
    print(f"Loading repo from {repo_root}...")
    loader = Loader(repo_root)
    chunks = loader.process_directory()
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    
    print(f"Building dependency graph...")
    pkg_dir = os.path.abspath(repo_root)
    pkg_name = os.path.basename(pkg_dir)
    graph = DependencyGraph(pkg_name=pkg_name, pkg_dir=pkg_dir)
    graph.build()
    
    try:
        client = OpenAI()
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set.")
        return
        
    seed_path = os.path.join(os.path.dirname(__file__), "..", "training_data", "sft", "synthetic_qa_seed.jsonl")
    few_shot_examples = ""
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as sf:
            lines = [l.strip() for l in sf if l.strip()]
        single_turn = [l for l in lines if '"tool_calls"' not in l and '"cannot find"' not in l][:2]
        tool_use = [l for l in lines if '"tool_calls"' in l][:2]
        refusal = [l for l in lines if '"cannot find"' in l or 'I cannot find' in l][:1]
        few_shot_examples = "\n".join(single_turn + tool_use + refusal)
        print(f"[synth] Loaded {len(single_turn + tool_use + refusal)} few-shot seed examples.")

    allowed_files = {c.filepath for c in chunks}
    parsed_count = 0
    accepted_count = 0
    rejected_count = 0

    print(f"Generating Q&A for {len(chunks)} chunks using {model}...")
    
    with open(output_file, "w", encoding="utf-8") as out:
        for chunk in chunks:
            deps = graph.get_chunk_deps(chunk.filepath, chunk.name, chunk.parent_class)
            
            prompt_content = (
                (f"Here are example output rows in the correct format:\n{few_shot_examples}\n\n---\n\n" if few_shot_examples else "")
                + f"<chunk path={chunk.filepath} name={chunk.name}>\n"
                + f"{chunk.text}\n"
                + f"</chunk>\n"
                + f"<calls_to>{deps['calls_to']}</calls_to>\n"
                + f"<called_by>{deps['called_by']}</called_by>\n"
                + f"<inherits>{deps['inherits_from']}</inherits>\n"
                + f"<imported_by>{deps['imported_by']}</imported_by>\n\n"
                + f"Generate exactly {examples_per_chunk} JSONL rows for this chunk if possible."
            )
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": TEACHER_SYSTEM.replace("<REPO_MAP>", repo_map_string or "")},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.4
                )
                
                output = response.choices[0].message.content
                raw = output.strip()
                
                parsed_rows = []
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or not line.startswith('{'):
                        continue
                    try:
                        obj = json.loads(line)
                        parsed_rows.append(obj)
                    except json.JSONDecodeError:
                        pass
                
                if not parsed_rows:
                    try:
                        obj = json.loads(raw)
                        parsed_rows.append(obj)
                    except Exception:
                        pass
                
                parsed_count += len(parsed_rows)
                for obj in parsed_rows:
                    if "messages" not in obj and "question" in obj and "answer" in obj:
                        obj = {
                            "messages": [
                                {"role": "system", "content": "You are a Python code understanding assistant. Always cite the file path and line numbers when referring to code."},
                                {"role": "user", "content": obj["question"]},
                                {"role": "assistant", "content": obj["answer"]}
                            ]
                        }
                    
                    is_valid, reason = validate_training_row(obj, allowed_files)
                    if is_valid:
                        accepted_count += 1
                        out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    else:
                        rejected_count += 1
                        if rejected_count <= 5 or random.random() < 0.05:
                            print(f"[synth] Rejected row due to: {reason}")
            except Exception as e:
                print(f"Failed generation for {chunk.name}: {e}")

    print(f"Synthetic generation complete:")
    print(f"  Parsed rows:   {parsed_count}")
    print(f"  Accepted rows: {accepted_count}")
    print(f"  Rejected rows: {rejected_count}")


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.loader import Loader
    from rag.repo_map import RepoMapGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="training_data/sft/synthetic_qa_auto.jsonl")
    parser.add_argument("--model", default="gpt-4o", help="Teacher LLM model name.")
    parser.add_argument("--examples-per-chunk", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    _loader = Loader(os.path.abspath(args.repo))
    _chunks = _loader.process_directory()
    _repo_map = RepoMapGenerator().generate_map(_chunks)

    generate_synthetic_data(
        args.repo,
        args.out,
        model=args.model,
        repo_map_string=_repo_map,
        examples_per_chunk=args.examples_per_chunk,
        max_chunks=args.max_chunks
    )
