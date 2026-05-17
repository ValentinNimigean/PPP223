# PPP223
# Python Code Understanding SLM Agent


# Team Mobtrap

# Members
* Nimigean Valentin
* Balog David
* Soptelea Sebastian
* Szarics Iulia

## Overview
This project focuses on building a Small Language Model (SLM) agent designed for deep Python code understanding. The core objective is to move beyond simple text-matching to create a system that understands the hierarchical nature of Python.


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
### 1. Dataset Selection: The Training Mixture 
To ensure excellence in code reasoning, the agent utilizes a three-tier data strategy:
* **High-Signal Code:** Leveraging The Stack v2 (Python Subset) for diverse repository exposure and CodeSearchNet for alignment between logic and natural language.
* **Logic Benchmarking:** Integrating Human Eval and PyCode Bench during the SFT (Supervised Fine-Tuning) phase for logical consistency .
* **Synthetic Reasoning:** Generating synthetic Q&A pairs from complex dependency graphs to teach the model how different repository files interact.

### 2. Chunking Strategy: AST-Based Parsing
We have moved away from character-count chunking in favor of Abstract Syntax Tree (AST) Chunking :
* **Structural Integrity:** Code is split into logical blocks such as Classes, Methods, and Functions .
* **Context Preservation:** Every chunk includes decorators, function signatures, and inheritance info to ensure the model maintains the "context" of a code snippet.

### 3. Model & Vector Database Choices
* **Primary Models:** Qwen2.5-Coder-3B as the primary model, 1.5B as the lightweight fallback
* **Vector Storage:** Qdrant is selected for its advanced filtering capabilities, allowing for specific metadata queries (e.g., finding methods within a specific class).

### 4. RAG Strategies
The system experiments with two advanced retrieval methods:
* **Repo Map Retrieval:** Providing a "bird's eye view" of the entire file structure before fetching specific code.
* **Hybrid Search:** Combining Vector embeddings for semantic meaning with BM25 keyword search for finding specific variable or function names.

---

## Project Roadmap (Weeks 7-12)

### Phase 1: Sprinting Foundations (Weeks 7-8) 
* **Week 7:** Implementing the AST parser and metadata attachment.
* **Week 8:** Finalizing the "Repo Map" and hybrid retrieval logic in Qdrant.

### Phase 2: Fine-Tuning & Logic (Weeks 9-10)
* **Week 9:** Integrating the SLM and adding tool-calling capabilities (e.g., `grep`) to enable repository exploration.
* **Week 10:** Performing QLORA fine-tuning on curated Python datasets to improve structured output.

### Phase 3: Alignment & Deployment (Weeks 11-12)
* **Week 11:** Using DPO (Direct Preference Optimization) to rank 500-1000 outputs, reducing hallucinations.
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
python eval/eval.py \
  --repo . \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_self.json \
  --out eval_report_demo.json
```

**2. Run Honest-mode evaluation (Self-repo):**
```bash
python eval/eval.py \
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

---

## Data hygiene

To prevent training contamination and keep model reasoning high-quality, we enforce strict data hygiene checks prior to SFT/DPO training:

* **Contamination Auditing (`scripts/audit_training_data.py`)**: A reusable script that parses JSONL files and automatically rejects the dataset (exiting with code `1`) if any line contains absolute local machine paths (e.g., `file://`, `/home/`, `/Users/`, `C:\`, `Documents/GitHub`) or contaminated terms (e.g., cloud SDK mentions, framework leaks).
* **DPO Validation (`scripts/validate_dpo.py`)**: Ensures that chosen DPO candidate answers do not contain vague phrases or toxic prompt anomalies.
* **Impact**: Absolute local paths are strictly rejected from training data to avoid teaching the model bad citation behaviors. Contaminated preference labels are detected and filtered before DPO alignment.


