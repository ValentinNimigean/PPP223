# PPP223: Python Code Understanding SLM Agent

Developed by **Team Mobtrap**:
* Nimigean Valentin
* Balog David
* Soptelea Sebastian
* Szarics Iulia

For quick-start setup, demo commands, fine-tuning scripts, and environment configuration, refer to [HOW_TO_USE.md](HOW_TO_USE.md).

---

## 🌟 Overview

This repository houses an agentic, Small Language Model (SLM)-based assistant designed for deep Python code understanding. Moving beyond simple text/character-count matching, the system extracts code semantics and structural hierarchies using Abstract Syntax Trees (AST). It generates an interactive codebase map, retrieves relevant contexts via a hybrid dense-sparse vector database, and uses an agentic loop equipped with file-system search tools to explore, reason, and accurately answer user queries.

---

## 📂 Repository Structure

Below is an overview of the directories and primary files in the workspace:

```
├── data/                           # Codebase logic analysis and synthetic data tools
│   ├── dep_graph.py                # Analyzes imports and calls to build a Python dependency graph
│   ├── synth.py                    # Generates synthetic Q&A reasoning pairs based on the dependency graph
│   ├── pref_gen.py                 # Compiles preference pairs (chosen/rejected) for DPO alignment
│   └── failure_dpo_from_eval.py    # Extracts and prepares failed evaluation samples for DPO training
│
├── eval/                           # Codebase evaluation and validation modules
│   ├── eval.py                     # Primary benchmark evaluator running questions against the agent
│   ├── hallucination.py            # HallucinationDetector checking responses against ingested files/symbols
│   ├── toxicity.py                 # ToxicityDetector filtering toxic or abusive inputs
│   ├── run_comparison.py           # Compares performance metrics across model configurations
│   ├── benchmark_self.json         # Self-repo evaluation dataset
│   ├── benchmark_httpx.json        # External validation dataset for the encode/httpx repository
│   └── benchmark_unseen_self.json  # Additional unseen self-eval questions
│
├── ingest/                         # Codebase scanning and processing pipeline
│   ├── chunker.py                  # Structural parsing of Python files into AST chunks (classes, methods, functions)
│   ├── loader.py                   # Recursive project crawler with directory filter and tree-sitter syntax fallback
│   └── metadata.py                 # Defines the CodeChunk data model and metadata schemas
│
├── model/                          # Agentic loop, training pipelines, and local inference
│   ├── agent.py                    # Main SLMAgent implementation handling tool orchestration and safety checks
│   ├── tools.py                    # Implementation of agent tools (grep_search and semantic_search)
│   ├── finetune.py                 # SFT training pipeline using Unsloth QLoRA
│   ├── dpo.py                      # Direct Preference Optimization (DPO) alignment script
│   └── inference.py                # Command-line utility to query models
│
├── rag/                            # Retrieval-Augmented Generation context builders
│   ├── repo_map.py                 # Generates global repository maps summarizing file hierarchies and symbols
│   └── retriever.py                # HybridRetriever implementing Qdrant vector search and lexical fallback
│
├── scripts/                        # Automation shell/PowerShell scripts
│   ├── train_pipeline.sh / .ps1    # Automated SFT & DPO training and validation pipeline
│   ├── run_eval_modes.sh           # Script running evaluations in Demo and Honest modes
│   ├── eval_external_httpx.sh      # Clones and runs evaluation against the external httpx codebase
│   ├── download_models.py          # Pre-downloads embedding models to local cache for offline execution
│   ├── audit_training_data.py      # Checks dataset files for path leaks and contamination before training
│   └── validate_dpo.py             # Validates DPO dataset structure, divergence, and citations
│
├── tests/                          # Unit and integration tests
│   ├── test_agent_retriever.py     # Tests retriever functionality and agent paths
│   └── test_hallucination_det.py   # Tests the hallucination detector rules
│
├── ui/                             # User interface applications
│   ├── app.py                      # Streamlit-based web app with RLHF feedback loop (👍/👎 + correction)
│   └── cli.py                      # Interactive loop-based command line assistant
│
├── main.py                         # Single-query entrypoint script
├── requirements.txt                # Core dependency definitions
├── requirements-train-local.txt    # Local training requirements (Unsloth + PyTorch)
├── requirements-train.txt          # Server training requirements
└── requirements-dev.txt            # Development and testing requirements (pytest, ruff)
```

---

## 🛠️ Installation & Setup

Choose the installation path that matches your use case:

### 1. Dev/Demo (Runtime Only)
Installs the lightweight requirements to run the CLI, Streamlit UI, and local agent:
```bash
pip install -r requirements.txt
```
> [!NOTE]
> Requires [Ollama](https://ollama.com) to be running locally with the target model (e.g. `qwen2.5-coder:3b`).

### 2. Training (GPU Machine Only)
To run SFT fine-tuning or DPO alignment, install the GPU-compatible training environment:
```bash
pip install -r requirements-train-local.txt
```
> [!IMPORTANT]
> Requires an NVIDIA GPU with CUDA 12.1+ and 6GB+ VRAM. Unsloth is compiled from source.

### 3. Contributing & Testing
To run tests and formatting checks:
```bash
pip install -r requirements-dev.txt
pytest
```

---

## 🧠 System Architecture

### 1. Dataset Selection & Synthetic Generation
To optimize local SLM code reasoning, the repository compiles datasets from multiple source layers:
* **Structural Synthesis:** Synthesizes QA pairs dynamically via `data/synth.py` by traversing codebase dependencies to build multi-file traversal questions.
* **Alignment Tuning:** Refined using preference optimization datasets collected from UI corrections or generated programmatically via `data/pref_gen.py`.

### 2. AST-Based Chunking
Instead of traditional character-limit text splitting, `ingest/chunker.py` and `ingest/loader.py` apply structure-aware parsing:
* **Symbol Extraction:** Divides Python source code by Classes, Methods, and Functions.
* **Context Preservation:** Keeps relevant docstrings, function signatures, decorators, and base classes attached to each chunk.
* **Syntax Fallback:** Automatically falls back to `tree-sitter-python` if standard AST parsing fails due to syntax errors.

### 3. Hybrid Retrieval & Resilient Fallback
Implemented in `rag/retriever.py`:
* **Dense & Sparse Vectors:** Combines `BAAI/bge-small-en-v1.5` embeddings for semantic intent and `Qdrant/bm25` for exact variable/keyword matches.
* **Fusion Logic:** Utilizes Reciprocal Rank Fusion (RRF) to blend vector results.
* **Lexical Fallback:** Gracefully falls back to pure TF-IDF/lexical search if Qdrant or system dependencies fail, ensuring uninterrupted agent execution.
* **Offline Execution:** Supports offline caching via `scripts/download_models.py`, storing models in `.fastembed_cache`.

### 4. Agentic Control Loop & Guardrails
Orchestrated in `model/agent.py`:
* **Tool Access:** Native functions like `grep_search` and `semantic_search` allow the model to actively query code.
* **Toxicity Filter:** Rejects offensive prompts using `eval/toxicity.py`.
* **Hallucination Detection:** Uses `eval/hallucination.py` to compare code identifiers and paths cited in the output against retrieved codebase context, issuing warning banners for unverified entities.

---

## 📊 Evaluation Modes

To benchmark and audit the agent without self-repo contamination, three evaluation modes are defined:

* **Demo mode**: Uses deterministic early-exit shortcuts and formatted templates. This mode yields high scores (e.g. ~0.92) and provides a fast, stable demonstration of capabilities.
* **Honest mode**: Disables early shortcuts and templates, forcing the agent to use raw tool-calling, reasoning, and retrieval to resolve the query.
* **External-repo mode**: Evaluates the agent against an entirely separate repository (such as `httpx`), testing the real-world generalization of the RAG system and model.

### Quick Commands

**Run Demo-mode Evaluation (Self-repo):**
```bash
python eval/eval.py \
  --repo . \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_self.json \
  --out evaluation_reports/eval_report_demo.json
```

**Run Honest-mode Evaluation (Self-repo):**
```bash
python eval/eval.py \
  --repo . \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_self.json \
  --out evaluation_reports/eval_report_honest.json \
  --disable-deterministic-shortcuts
```

**Run External Evaluation (httpx):**
```bash
git clone https://github.com/encode/httpx.git ../httpx
bash scripts/eval_external_httpx.sh ../httpx qwen2.5-coder:3b
```

---

## 🧹 Data Hygiene & Validation

Ensuring high-quality inputs is essential before executing SFT or DPO. We run strict automated validation:

* **Contamination Audit (`scripts/audit_training_data.py`)**: Automatically scans and rejects training files that contain local machine absolute paths (e.g., `file://`, `/Users/`, `/home/`), ensuring the agent does not memorize local setup environments.
* **DPO Validation (`scripts/validate_dpo.py`)**: Checks preference files for schema validity, response divergence, fake refusal patterns, and correct symbol citations. Pipeline runs will automatically abort if errors are encountered.



