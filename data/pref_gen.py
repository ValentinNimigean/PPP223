import json
import argparse
import os
import sys
import time
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' package not found. Please install it with 'pip install openai'.")
    sys.exit(1)

def get_cli_args():
    parser = argparse.ArgumentParser(description="Generate DPO preference data for RLHF.")
    parser.add_argument("--input", required=True, help="Path to synthetic_qa.jsonl")
    parser.add_argument("--output", default="preference_data.jsonl", help="Path to preference_data.jsonl")
    parser.add_argument("--model", default="gpt-4o", help="Teacher model")
    parser.add_argument("--judge", default="gpt-4o", help="Judge model")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--limit", type=int, help="Max rows to process")
    return parser.parse_args()

def load_existing_prompts(output_path: str) -> set:
    seen_prompts = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "prompt" in data:
                        seen_prompts.add(data["prompt"])
                except json.JSONDecodeError:
                    continue
    return seen_prompts

def main():
    args = get_cli_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    
    seen_prompts = load_existing_prompts(args.output)
    
    written_count = 0
    skipped_count = 0
    failed_count = 0
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    judge_system_prompt = (
        "You are a strict judge evaluating answers about a Python codebase. "
        "Return ONLY a JSON object with keys 'best' (int, 0-2), 'worst' (int, 0-2), "
        "and 'reasoning' (str). Prefer answers that: are grounded in provided code, "
        "do not hallucinate function names, are concise, and correctly identify "
        "relationships. Do not include markdown or explanation outside the JSON."
    )
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)
        
    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "a", encoding="utf-8") as f_out:
        
        rows_processed = 0
        for line in f_in:
            if args.limit and rows_processed >= args.limit:
                break
                
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            # Extract user question
            messages = row.get("messages", [])
            user_question = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_question = msg.get("content", "")
                    break
            
            if not user_question:
                continue
                
            if user_question in seen_prompts:
                skipped_count += 1
                continue
            
            # Generate 3 candidates
            candidates = []
            temperatures = [0.9, 0.9, 0.1]
            
            gen_failed = False
            for temp in temperatures:
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": user_question}],
                        temperature=temp
                    )
                    candidates.append(response.choices[0].message.content)
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens
                except Exception as e:
                    print(f"Warning: Teacher generation failed: {e}")
                    gen_failed = True
                    break
            
            if gen_failed:
                failed_count += 1
                continue
                
            # Call Judge
            judge_user_content = f"Question: {user_question}\n\n"
            for i, cand in enumerate(candidates):
                judge_user_content += f"[{i}]: {cand}\n\n"
            
            try:
                judge_response = client.chat.completions.create(
                    model=args.judge,
                    messages=[
                        {"role": "system", "content": judge_system_prompt},
                        {"role": "user", "content": judge_user_content}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                judge_text = judge_response.choices[0].message.content
                judge_data = json.loads(judge_text)
                
                best_idx = judge_data.get("best")
                worst_idx = judge_data.get("worst")
                
                if not isinstance(best_idx, int) or not isinstance(worst_idx, int):
                    raise ValueError("Judge returned non-integer indices")
                
                if best_idx < 0 or best_idx > 2 or worst_idx < 0 or worst_idx > 2:
                    raise ValueError("Judge returned out-of-bounds indices")
                
                total_prompt_tokens += judge_response.usage.prompt_tokens
                total_completion_tokens += judge_response.usage.completion_tokens
                
                # Write to output
                output_row = {
                    "prompt": user_question,
                    "chosen": candidates[best_idx],
                    "rejected": candidates[worst_idx]
                }
                f_out.write(json.dumps(output_row) + "\n")
                f_out.flush()
                
                seen_prompts.add(user_question)
                written_count += 1
                rows_processed += 1
                
            except Exception as e:
                print(f"Warning: Judge evaluation failed for row: {e}")
                failed_count += 1
                continue

    # Final summary
    print(f"Done. Written: {written_count} rows. Skipped (already exists): {skipped_count} rows. Failed (judge error): {failed_count} rows.")
    
    # Cost calculation
    # $5/1M input $15/1M output
    cost_input = (total_prompt_tokens / 1_000_000) * 5.0
    cost_output = (total_completion_tokens / 1_000_000) * 15.0
    total_cost = cost_input + cost_output
    
    print(f"Total prompt tokens used: {total_prompt_tokens}")
    print(f"Total completion tokens used: {total_completion_tokens}")
    print(f"Estimated cost: ${total_cost:.4f}")

if __name__ == "__main__":
    main()
