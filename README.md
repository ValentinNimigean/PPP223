# PPP223
# Python Code Understanding SLM Agent


# Team Mobtrap

# Members
* Nimigean Valentin
* Balog David
* Soptelea Sebastian
* Szarics Iulia

For setup, demo commands, evaluation, and fine-tuning instructions, see [HOW_TO_USE.md](HOW_TO_USE.md).
For PPO-based RLHF, reward modeling, and dry-run commands, see [docs/RLHF_PPO.md](docs/RLHF_PPO.md).

## Overview
This project focuses on building a Small Language Model (SLM) agent designed for deep Python code understanding. The core objective is to move beyond simple text-matching to create a system that understands the hierarchical nature of Python.

The main implementation path uses a fine-tuned local/open SLM, grounded with RAG and aligned with PPO-based RLHF. Commercial models are not the primary solution in this repository.

Training uses cleaned final datasets only:
* `data/final/sft_train.jsonl` and `data/final/sft_val.jsonl` for SFT
* `data/final/preferences_train.jsonl` and `data/final/preferences_val.jsonl` for reward modeling / PPO
* Raw helper files such as `synthetic_qa_combined_2048.jsonl` and `preference_data_combined.jsonl` are not valid training inputs


---

## Installation

Choose the installation path that matches your use case:

- **Dev/Demo (Runtime Only):**
  ```bash
  pip install -r requirements.txt
  ```
  *Note: Requires Ollama to be running locally for SLM inference.*

- **Training (GPU Machine Only):**
  ```bash
  pip install -r requirements-train.txt
  ```
  *Note: Requires NVIDIA GPU with CUDA 12.1+. VRAM: 4-6GB (SFT), 6GB+ (DPO).*

- **Contributing (Development):**
  ```bash
  pip install -r requirements-dev.txt
  ```

---

## Architecture
The project is organized around a small set of extension points so ingestion, retrieval, inference, evaluation, and training can evolve independently.

### Codebase modules
* `ingest/`: AST chunking, repository scanning, and chunk metadata. `ingest/base.py` defines the shared chunk and loader interfaces, while `Loader` and `CodeChunk` remain the concrete defaults.
* `rag/`: Repository map generation and retrieval. `rag/base.py` defines the retriever contract, and `HybridRetriever` provides the current Qdrant-plus-lexical implementation.
* `model/`: Agent orchestration, inference backends, and compatibility wrappers for training/inference entry points. `model/base.py` defines chat/prompt backend interfaces used by the agent layer.
* `train/`: Task-aware SFT training for chat, Q&A, classification, summarization, and code-repo Q&A. This package owns dataset validation, task formatting, and the main QLoRA SFT CLI.
* `alignment/`: PPO-based RLHF, canonical preference validation, reward-model training, PPO prompt generation, and rule-based reward shaping. DPO remains available separately as a baseline.
* `eval/`: Benchmarks, safety checks, and evaluation scripts. `eval/base.py` provides normalized metric records for evaluators.
* `data/`: Synthetic data generation, dependency graph analysis, and preference-data preparation for SFT and DPO.
* `ui/`: CLI and Streamlit entry points that compose the loader, retriever, and agent into user-facing flows.
* `scripts/`: Shell and Python utilities for training, auditing, evaluation, and validation.
* `tests/`: Regression coverage for imports, retrieval behavior, hallucination checks, and CLI parsing.

### Ingestion and chunking
* **AST-first parsing:** Python files are split into classes, methods, and functions instead of arbitrary text spans.
* **Rich chunk metadata:** Chunks preserve file paths, line ranges, signatures, decorators, inheritance, and qualified names.
* **Extensible loaders:** Additional repository loaders can implement the `LoaderProtocol` without changing downstream retrieval code.

### Retrieval and RAG
* **Repo map retrieval:** The agent receives a high-level structural map of the repository before tool use.
* **Hybrid retrieval:** `HybridRetriever` combines Qdrant embeddings with a deterministic lexical fallback, so the app still works when vector search is unavailable.
* **Stable retriever interface:** New retrievers can implement `RetrieverProtocol` and plug into the existing agent/eval flows.

### Inference, training, and alignment
* **Primary models:** Qwen2.5-Coder-3B is the main local inference target, with 1.5B variants used in lightweight training flows.
* **Default model path:** the final/default system prefers `results_ppo/adapter`, then `results_sft/adapter`, over an unfine-tuned base model.
* **Backend abstraction:** The recommended runtime uses adapter-backed Hugging Face inference for the fine-tuned SLM path, while local base-model backends remain available for debugging.
* **Training pipeline:** `train/` handles task-specific SFT, `alignment/` handles PPO-based RLHF, and RAG grounds answers against repository evidence.

### Evaluation and safety
* **Self and external benchmarks:** `eval/benchmark_*.json` files and `eval/eval.py` support both fast self-repo checks and external-repo evaluation.
* **Safety layers:** Toxicity and hallucination detectors live under `eval/` and are reused by the agent/UI flows.
* **Normalized metrics:** `MetricRecord` provides a common shape for evaluator outputs as the benchmark layer grows.
* **Verification focus:** evaluation explicitly checks answer quality, hallucination handling, toxicity handling, and RAG-grounded file/entity citations.

---

## Project Roadmap (Weeks 7-12)

### Phase 1: Sprinting Foundations (Weeks 7-8) 
* **Week 7:** Implementing the AST parser and metadata attachment.
* **Week 8:** Finalizing the "Repo Map" and hybrid retrieval logic in Qdrant.

### Phase 2: Fine-Tuning & Logic (Weeks 9-10)
* **Week 9:** Integrating the SLM and adding tool-calling capabilities (e.g., `grep`) to enable repository exploration.
* **Week 10:** Performing QLORA fine-tuning on curated Python datasets to improve structured output.

### Phase 3: Alignment & Deployment (Weeks 11-12)
* **Week 11:** Using preference optimization baselines and PPO-based RLHF to reduce hallucinations. DPO remains optional only.
* **Week 12:** Applying 4-bit model quantization for local speed and launching the Streamlit-based UI.

---

## Evaluation modes

Self-repo benchmarks are useful for fast iteration, but can suffer from contamination because the agent's deterministic shortcuts and system prompts were developed around the repository structure itself. To ensure transparency, we distinguish between three evaluation modes:

* **Demo mode**: Uses deterministic guardrails and tool-result templates to make the user-facing application fast and reliable. This mode is the default and produces high stabilized scores, serving as a rapid capability demonstration rather than a pure measure of model generalization.
* **Honest eval mode**: Disables all deterministic shortcuts and templates, measuring the pure RAG + LLM loop directly. The agent is forced to use actual tool-calling, reasoning, and retrieval.
* **External-repo eval**: Evaluates the agent on a completely separate, unseen repository. This is the preferred method for judging true generalization capability.

### Command Examples

**1. Run Demo-mode evaluation (Self-repo):**
```bash
python -m eval.eval \
  --repo . \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_self.json \
  --out eval_report_demo.json
```

**2. Run Honest-mode evaluation (Self-repo):**
```bash
python -m eval.eval \
  --repo . \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_self.json \
  --out eval_report_honest.json \
  --disable-deterministic-shortcuts
```

**3. Run Honest External-repo evaluation (on httpx):**
```bash
git clone https://github.com/encode/httpx.git ../httpx
bash scripts/eval_external_httpx.sh ../httpx qwen2.5-coder:3b
```

---

## Fine-tuned SLM

To optimize the agent for local inference and specialized Python code understanding, we perform fine-tuning:

* **Base model**: `unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit` (lightweight, instruct-aligned fallback).
* **Method**: QLoRA parameter-efficient fine-tuning with **Unsloth** for 2x faster training and memory optimization.
* **Adapter output**: Saved to `results_sft/adapter` (containing SFT LoRA adapters).
* **Note on Inference**: The Streamlit UI currently uses the base Ollama model unless the SFT/DPO adapter is exported/merged, or served through an adapter-aware backend.

Recommended SFT command:
```bash
python -m model.finetune \
  --task chat \
  --data data/final/sft_train.jsonl \
  --val-data data/final/sft_val.jsonl \
  --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
  --out results_sft
```

---

## Data hygiene

To prevent training contamination and keep model reasoning high-quality, we enforce strict data hygiene checks prior to SFT, reward-model, PPO, or optional DPO training:

* **Dataset Auditing (`scripts/audit_datasets.py`)**: Audits final SFT and preference datasets for duplicates, empty answers, malformed rows, contamination, and train/eval leakage. Rejections exit with code `1` to stop downstream automation pipelines.
* **Preference Validation (`python -m alignment.validate_preferences`)**: Ensures chosen/rejected pairs are structurally valid before reward-model, DPO, or PPO training.
* **Impact**: Absolute local paths are strictly rejected from training data to avoid teaching the model bad citation behaviors. Contaminated preference labels are detected and filtered before DPO alignment.
* **Final training inputs**: Run `python scripts/audit_datasets.py --sft data/final/sft_train.jsonl data/final/sft_val.jsonl --preferences data/final/preferences_train.jsonl data/final/preferences_val.jsonl` before training. The training scripts now do this automatically.
