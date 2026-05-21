import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure root path is accessible when running from ui/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser():
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="SLM Agent CLI Interface")
    parser.add_argument("--repo", default=".", help="Path to the Python repo to analyze.")
    parser.add_argument("--question", help="Optional one-shot question to answer and exit.")
    parser.add_argument(
        "--backend",
        choices=("ollama", "hf-peft"),
        default="hf-peft",
        help="Inference backend to use. The default production path is local hf-peft with a fine-tuned adapter.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="Base Hugging Face model for local fine-tuned SLM inference.",
    )
    parser.add_argument(
        "--adapter-path",
        help="Path to a LoRA/PEFT adapter directory. If omitted, the CLI prefers results_ppo/adapter then results_sft/adapter.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for hf-peft inference. Use 'auto', 'cpu', 'cuda', or a specific torch device.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--disable-deterministic-shortcuts",
        action="store_true",
        help="Disable deterministic early shortcuts and templates.",
    )
    return parser


def build_agent(args, repo_map):
    """Construct an agent for the selected backend."""
    from model.agent import SLMAgent
    from model.runtime import resolve_preferred_adapter

    inference_engine = None

    if args.backend == "hf-peft":
        from model.inference import PeftAdapterInferenceEngine

        adapter_path_str = args.adapter_path
        if not adapter_path_str:
            adapter_path_str, _label = resolve_preferred_adapter(Path(args.repo))
        adapter_path = Path(adapter_path_str)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        inference_engine = PeftAdapterInferenceEngine(
            base_model=args.base_model,
            adapter_path=str(adapter_path),
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    return SLMAgent(
        repo_map_string=repo_map,
        hallucination_check=True,
        enable_deterministic_shortcuts=not args.disable_deterministic_shortcuts,
        enable_tool_result_templates=not args.disable_deterministic_shortcuts,
        inference_engine=inference_engine,
    )


def run_cli():
    """Run the interactive or one-shot CLI workflow."""
    logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo)
    if not os.path.isdir(repo_path):
        print(f"Error: {repo_path} is not a valid directory.")
        sys.exit(1)

    from ingest.loader import Loader
    from rag.repo_map import RepoMapGenerator
    from rag.retriever import HybridRetriever

    print(f"--- SLM Agent CLI Interface: Analyzing {repo_path} ---")

    print("[1/3] Scanning codebase and parsing AST...")
    loader = Loader(repo_path)
    chunks = loader.process_directory()
    repo_map = RepoMapGenerator().generate_map(chunks)

    print(f"[2/3] Ingesting {len(chunks)} logic chunks into Hybrid Qdrant...")
    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)

    try:
        if args.backend == "hf-peft":
            from model.runtime import resolve_preferred_adapter

            adapter_path = args.adapter_path
            adapter_label = None
            if not adapter_path:
                adapter_path, adapter_label = resolve_preferred_adapter(repo_path)
            print(
                "[3/3] Booting Fine-Tuned Local SLM... "
                f"({args.base_model} + adapter {adapter_path}"
                f"{f' [{adapter_label}]' if adapter_label else ''})"
            )
            args.adapter_path = adapter_path
        else:
            print("[3/3] Booting Local Base SLM... (qwen2.5-coder via local backend)")
        agent = build_agent(args, repo_map)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    agent.set_retriever(retriever)
    agent.set_chunks(chunks)

    if args.question:
        response = agent.ask(args.question)
        guardrail = agent.last_guardrail_result
        if guardrail and (not guardrail.allowed or guardrail.hallucination_risk in {"medium", "high"}):
            print(
                "[Safety] "
                f"reason={guardrail.reason} "
                f"toxicity={guardrail.toxicity_score:.2f} "
                f"hallucination={guardrail.hallucination_score:.2f}"
            )
            if guardrail.unsupported_entities:
                print("[Safety] Unsupported entities:", ", ".join(guardrail.unsupported_entities))
            if guardrail.unsupported_files:
                print("[Safety] Unsupported files:", ", ".join(guardrail.unsupported_files))
        print(f"\nAgent> {response}\n")
        return

    print("\n[READY] Start asking questions about your codebase.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You> ")
            if query.strip().lower() in ["exit", "quit"]:
                break
            if not query.strip():
                continue

            response = agent.ask(query)
            guardrail = agent.last_guardrail_result
            if guardrail and (not guardrail.allowed or guardrail.hallucination_risk in {"medium", "high"}):
                print(
                    "[Safety] "
                    f"reason={guardrail.reason} "
                    f"toxicity={guardrail.toxicity_score:.2f} "
                    f"hallucination={guardrail.hallucination_score:.2f}"
                )
                if guardrail.unsupported_entities:
                    print("[Safety] Unsupported entities:", ", ".join(guardrail.unsupported_entities))
                if guardrail.unsupported_files:
                    print("[Safety] Unsupported files:", ", ".join(guardrail.unsupported_files))
            print(f"\nAgent> {response}\n")

        except KeyboardInterrupt:
            print("\nExiting CLI.")
            break
        except Exception as exc:
            print(f"\n[System Error]: {exc}")


if __name__ == "__main__":
    run_cli()
