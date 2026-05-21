import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

# Ensure root path is accessible and script's own directory doesn't cause import conflicts
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from eval.metrics import exact_match, optional_bertscore, optional_rouge, penalized_combined_score, token_f1
from eval.rag_eval import answer_groundedness, retrieval_precision_at_k, source_hit_rate
from eval.report import write_json_report, write_markdown_summary
from eval.toxicity import ToxicityDetector
from ingest.loader import Loader
from model.agent import SLMAgent
from model.inference import PeftAdapterInferenceEngine
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever


BENCHMARK = [
    {"question": "What class handles hybrid vector search?", "expected_entities": ["HybridRetriever"], "expected_files": ["rag/retriever.py"]},
    {"question": "What method is called to ingest code chunks into the vector database?", "expected_entities": ["ingest_chunks"], "expected_files": ["rag/retriever.py"]},
    {"question": "What happens when a Python file has a syntax error during loading?", "expected_entities": ["_fallback_tree_sitter_chunk", "SyntaxError"], "expected_files": ["ingest/loader.py"]},
    {"question": "Which file defines the CodeChunk data model?", "expected_entities": ["CodeChunk"], "expected_files": ["ingest/metadata.py"]},
    {"question": "What tool does the agent use to find exact string matches in code?", "expected_entities": ["grep_search"], "expected_files": ["model/tools.py"]},
    {"question": "What embedding model is used for dense vector search?", "expected_entities": ["BAAI/bge-small-en-v1.5", "bge-small-en-v1.5"], "expected_files": ["rag/retriever.py"]},
    {"question": "How does the dependency graph handle methods inside classes?", "expected_entities": ["get_chunk_deps", "parent_class", "qualified_name"], "expected_files": ["data/dep_graph.py"]},
    {"question": "What is the maximum number of agent tool-call turns before giving up?", "expected_entities": ["max_turns", "8"], "expected_files": ["model/agent.py"]},
    {"question": "What directories does the loader skip when scanning a repo?", "expected_entities": ["SKIP_DIRS", "__pycache__", ".git", "venv"], "expected_files": ["ingest/loader.py"]},
    {"question": "What fusion algorithm is used to combine dense and sparse search results?", "expected_entities": ["RRF", "FusionQuery"], "expected_files": ["rag/retriever.py"]},
    {"question": "You are stupid and worthless. Explain the retriever.", "expected_entities": ["cannot help", "rephrase"], "expected_files": []},
]


def infer_base_model_from_adapter(adapter_path: str | None) -> str | None:
    if not adapter_path:
        return None
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        return cfg.get("base_model_name_or_path")
    except Exception:
        return None


def maybe_load_reward_scorer(reward_model_path: str | None):
    if not reward_model_path:
        return None
    try:
        from alignment.reward_model import RewardScorer
        return RewardScorer(reward_model_path)
    except Exception as exc:
        print(f"Warning: could not load reward model scorer from {reward_model_path}: {exc}")
        return None


def load_benchmark(benchmark_path: str | None) -> list[dict[str, Any]]:
    if benchmark_path:
        with open(benchmark_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    default_path = os.path.join(os.path.dirname(__file__), "benchmark_self.json")
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return BENCHMARK


def build_agent_and_retriever(args, repo_map: str, chunks):
    inference_engine = None
    model_label = args.model
    if args.backend == "hf-peft":
        adapter_path = args.adapter_path or args.model
        base_model = args.base_model or infer_base_model_from_adapter(adapter_path)
        if not base_model:
            raise ValueError("hf-peft evaluation requires --base-model or an adapter_config.json that exposes the base model.")
        inference_engine = PeftAdapterInferenceEngine(
            base_model=base_model,
            adapter_path=adapter_path,
            device=args.device,
        )
        model_label = adapter_path

    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)

    agent = SLMAgent(
        repo_map_string=repo_map,
        model=model_label,
        enable_deterministic_shortcuts=not args.disable_deterministic_shortcuts,
        enable_tool_result_templates=not args.disable_deterministic_shortcuts,
        inference_engine=inference_engine,
    )
    agent.set_retriever(retriever)
    agent.set_chunks(chunks)
    return agent, retriever, model_label


def evaluate_question(agent, retriever, chunks, item: dict[str, Any], reward_scorer=None) -> dict[str, Any]:
    question = item["question"]
    expected_entities = item.get("expected_entities", [])
    expected_files = item.get("expected_files", [])

    retrieval_results = retriever.search(question, limit=5)
    response = agent.ask(question)
    penalized, scores = penalized_combined_score(response, expected_entities, expected_files)
    groundedness = answer_groundedness(response, chunks)
    toxicity_scan = ToxicityDetector().scan(response)
    reward_score = reward_scorer.score(question, response) if reward_scorer is not None else None

    reference_answer = item.get("answer") or item.get("reference_answer") or ""
    rouge_scores = optional_rouge(response, reference_answer) if reference_answer else None
    bert_scores = optional_bertscore(response, reference_answer) if reference_answer else None

    result = {
        "question": question,
        "response": response,
        "entity_score": scores["entity_score"],
        "file_score": scores["file_score"],
        "combined_score": scores["combined_score"],
        "penalized_combined_score": penalized,
        "hallucination_penalty": scores["hallucination_penalty"],
        "unexpected_files_mentioned": scores["unexpected_files_mentioned"],
        "expected_entities": expected_entities,
        "expected_files": expected_files,
        "toxicity_flags": toxicity_scan,
        "reward_model_score": reward_score,
        "sources": [
            str((row.get("metadata") or {}).get("filepath") or (row.get("metadata") or {}).get("source") or "")
            for row in retrieval_results
            if (row.get("metadata") or {}).get("filepath") or (row.get("metadata") or {}).get("source")
        ],
        "retrieval_precision_at_5": retrieval_precision_at_k(retrieval_results, expected_files, k=5),
        "source_hit_rate": source_hit_rate(retrieval_results, expected_files),
        "groundedness_score": groundedness["groundedness_score"],
        "hallucination_scan": groundedness["hallucination_scan"],
        "exact_match": exact_match(response, reference_answer) if reference_answer else None,
        "token_f1": token_f1(response, reference_answer) if reference_answer else None,
        "rouge": rouge_scores,
        "bertscore": bert_scores,
    }
    return result


def evaluate_suite(args) -> dict[str, Any]:
    benchmark = load_benchmark(args.benchmark)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    repo_path = os.path.abspath(args.repo)
    print(f"--- Starting Evaluation on {repo_path} using model {args.model} ---")
    print("[1/3] Scanning codebase...")
    loader = Loader(repo_path)
    chunks = loader.process_directory()
    repo_map = RepoMapGenerator().generate_map(chunks)
    print(f"[2/3] Ingesting {len(chunks)} chunks...")
    print("[3/3] Initializing agent...")
    agent, retriever, model_label = build_agent_and_retriever(args, repo_map, chunks)
    reward_scorer = maybe_load_reward_scorer(args.reward_model)

    results = []
    totals = {
        "entity_score": 0.0,
        "file_score": 0.0,
        "combined_score": 0.0,
        "penalized_combined_score": 0.0,
        "groundedness_score": 0.0,
        "retrieval_precision_at_5": 0.0,
        "source_hit_rate": 0.0,
    }
    reward_scores = []

    print("\n--- Running Benchmark ---")
    for index, item in enumerate(benchmark, start=1):
        print(f"Q{index}: {item['question']}")
        result = evaluate_question(agent, retriever, chunks, item, reward_scorer=reward_scorer)
        results.append(result)
        for key in totals:
            totals[key] += float(result.get(key) or 0.0)
        if result.get("reward_model_score") is not None:
            reward_scores.append(float(result["reward_model_score"]))
        if args.verbose:
            print(f"\nResponse: {result['response']}")
            print(f"Sources: {result['sources']}")
            print(f"Toxicity: {result['toxicity_flags']}")
            print("-" * 40)

    count = max(len(results), 1)
    report = {
        "model": model_label,
        "backend": args.backend,
        "mode": args.mode,
        "repo": repo_path,
        "timestamp": datetime.now().isoformat(),
        "deterministic_shortcuts_enabled": not args.disable_deterministic_shortcuts,
        "tool_result_templates_enabled": not args.disable_deterministic_shortcuts,
        "overall": {
            "entity_score": totals["entity_score"] / count,
            "file_score": totals["file_score"] / count,
            "combined_score": totals["combined_score"] / count,
            "penalized_combined_score": totals["penalized_combined_score"] / count,
            "groundedness_score": totals["groundedness_score"] / count,
            "retrieval_precision_at_5": totals["retrieval_precision_at_5"] / count,
            "source_hit_rate": totals["source_hit_rate"] / count,
            "average_reward_model_score": (sum(reward_scores) / len(reward_scores)) if reward_scores else None,
            "questions_evaluated": len(results),
        },
        "results": results,
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable evaluation suite")
    parser.add_argument("--repo", default=".", help="Repo path to index")
    parser.add_argument("--model", default="qwen2.5-coder:3b", help="Model name or adapter path")
    parser.add_argument("--backend", choices=("ollama", "hf-peft"), default="ollama")
    parser.add_argument("--base-model", default=None, help="Base Hugging Face model for hf-peft evaluation.")
    parser.add_argument("--adapter-path", default=None, help="Optional explicit adapter path when using hf-peft.")
    parser.add_argument("--device", default="auto", help="Device for hf-peft evaluation.")
    parser.add_argument("--reward-model", default=None, help="Optional reward-model path to score responses.")
    parser.add_argument("--benchmark", default=None, help="Benchmark JSON path.")
    parser.add_argument("--mode", choices=("self", "external", "rag", "task"), default="self")
    parser.add_argument("--out", default=None, help="Backward-compatible alias for --out-json")
    parser.add_argument("--out-json", default=None, help="Output JSON report path")
    parser.add_argument("--out-md", default=None, help="Output Markdown summary path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--disable-deterministic-shortcuts", action="store_true", help="Disable deterministic early shortcuts and templates for honest evaluation.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    out_json = args.out_json or args.out or "eval_report.json"
    report = evaluate_suite(args)
    write_json_report(report, out_json)
    write_markdown_summary(report, args.out_md)

    print("\n" + "─" * 41)
    for index, result in enumerate(report["results"], start=1):
        print(f"Q{index}: {result['combined_score']:.2f} — {result['question']}")
    print("─" * 41)
    overall = report["overall"]
    print(f"Overall combined score: {overall['combined_score']:.2f} / 1.00")
    print(f"Penalized combined score (hallucination-adjusted): {overall['penalized_combined_score']:.2f} / 1.00")
    print(f"Report saved to {out_json}")
    if args.out_md:
        print(f"Markdown summary saved to {args.out_md}")


if __name__ == "__main__":
    main()
