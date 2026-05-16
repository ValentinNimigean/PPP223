import json
import os
from openai import OpenAI
from ingest.loader import Loader
from data.dep_graph import DependencyGraph

TEACHER_SYSTEM = """You generate Q&A training pairs for a code-understanding model.
The target model is Qwen2.5-Coder-3B; generate answers calibrated for a smaller model: shorter, more explicit, and step-by-step rather than assuming implicit reasoning.
Given a Python code chunk and structured facts about its relationships in a repo,
produce 3 diverse natural-language Q&A pairs. Vary phrasing: some lookup-style
("where is X defined?"), some semantic ("what does X do?"), some multi-hop
("how do A and B interact?"). Return JSONL format only, keys 'question' and 'answer'.
Ground every answer in the provided code; if unsupported, refuse."""

def generate_synthetic_data(repo_root: str, output_file: str, model: str = "gpt-4o"):
    print(f"Loading repo from {repo_root}...")
    loader = Loader(repo_root)
    chunks = loader.process_directory()
    
    print(f"Building dependency graph...")
    # Passing the repo root folder name or '.' if we're inside it
    pkg_dir = os.path.abspath(repo_root)
    pkg_name = os.path.basename(pkg_dir)
    graph = DependencyGraph(pkg_name=pkg_name, pkg_dir=pkg_dir)
    graph.build()
    
    # Initialize teacher LLM (requires OPENAI_API_KEY if using OpenAI, or configure local base_url)
    try:
        client = OpenAI()
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set.")
        return
        
    print(f"Generating Q&A for {len(chunks)} chunks using {model}...")
    
    with open(output_file, "w", encoding="utf-8") as out:
        for chunk in chunks:
            deps = graph.get_chunk_deps(chunk.filepath, chunk.name, chunk.parent_class)
            
            prompt_content = (
                f"<chunk path={chunk.filepath} name={chunk.name}>\n"
                f"{chunk.text}\n"
                f"</chunk>\n"
                f"<calls_to>{deps['calls_to']}</calls_to>\n"
                f"<called_by>{deps['called_by']}</called_by>\n"
                f"<inherits>{deps['inherits_from']}</inherits>\n"
                f"<imported_by>{deps['imported_by']}</imported_by>"
            )
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": TEACHER_SYSTEM},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.7
                )
                
                output = response.choices[0].message.content
                # Write conversational JSONL format as preferred for TRL
                for line in output.split('\n'):
                    if line.strip().startswith('{'):
                        try:
                            qa = json.loads(line)
                            if 'question' in qa and 'answer' in qa:
                                out.write(json.dumps({
                                    "messages": [
                                        {"role": "system", "content": "You are a Python code understanding assistant."},
                                        {"role": "user", "content": qa['question']},
                                        {"role": "assistant", "content": qa['answer']}
                                    ]
                                }) + '\n')
                        except Exception:
                            pass
            except Exception as e:
                print(f"Failed generation for {chunk.name}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="synthetic_qa.jsonl")
    parser.add_argument("--model", default="gpt-4o", help="Teacher LLM model name.")
    args = parser.parse_args()
    generate_synthetic_data(args.repo, args.out, model=args.model)
