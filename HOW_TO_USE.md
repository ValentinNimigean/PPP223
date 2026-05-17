# How to Use Mobtrap / PPP223

## 1. What this project does

This project is an agentic, Small Language Model (SLM)-based codebase assistant designed for deep Python code understanding. Instead of relying on naive character-count text chunking, it parses Python repositories structurally using Abstract Syntax Tree (AST) definitions (falling back to tree-sitter when needed). It constructs a unified "Repo Map" that outlines classes, methods, and functions, allowing the model to quickly navigate files and contextually resolve hierarchical symbols.

The core pipeline features structural codebase ingestion, high-fidelity hybrid retrieval (combining dense vector Qdrant embeddings and lexical BM25 fallback), and an agentic loop with native tool access (`grep_search` and `semantic_search`). Additionally, it includes full pipelines for dataset compilation (synthetic reasoning generation), model evaluation, toxicity/hallucination checks, and alignment via Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

## 2. Requirements

To run this project, the following are required:
* **Python 3.11**
* **Ollama** (for local SLM inference)
* **Qwen2.5-Coder** model pulled through Ollama
* **CUDA-compatible GPU** (highly recommended for local model training and fine-tuning)
* **Virtual Environment** (venv/conda)

Verify your system setup by running:
```bash
python --version
ollama --version
nvidia-smi
```

## 3. Setup

Initialize your environment and install the required dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to run local fine-tuning or DPO alignment, install the training dependencies:
```bash
pip install -r requirements-train-local.txt
```

## 4. Start the local model

This project is designed to run locally using Ollama. Pull and launch the primary model (`qwen2.5-coder:3b`):
```bash
ollama pull qwen2.5-coder:3b
ollama run qwen2.5-coder:3b
```

Ollama must be running and accessible at:
```
http://localhost:11434
```

## 5. Run a one-shot question

You can query the agent about the local repository directly from the command line using `main.py`.

To run a one-shot question in **Demo mode** (deterministic shortcuts enabled):
```bash
python main.py \
  --repo . \
  --question "What class handles hybrid vector search?"
```

To run a one-shot question in **Honest mode** (deterministic shortcuts disabled, forcing real tool use):
```bash
python main.py \
  --repo . \
  --question "How does the agent handle malformed JSON tool calls?" \
  --disable-deterministic-shortcuts
```

### Explanations:
* `--repo .` points the agent to analyze the current directory.
* You can replace `.` with the absolute or relative path to any local Python repository.

## 6. Run the Streamlit UI

Start the Streamlit application to enjoy an interactive web-based chat interface:
```bash
streamlit run ui/app.py
```

### How to interact with the UI:
1. Open the localhost URL printed in the terminal (usually `http://localhost:8501`).
2. Set the target local repository path in the sidebar (default is `.`).
3. Enter your queries in the chat input at the bottom of the page.

### Recommended example questions to ask:
* *What class handles hybrid vector search?*
* *What happens when a Python file has a syntax error during loading?*
* *How does the agent handle malformed JSON tool calls?*
* *Does the repo define a FastAPIRetriever class?*
* *You are stupid and worthless. Explain the retriever.* (Demonstrates the toxicity filter rejecting harmful inputs)

## 7. Run the CLI

For an interactive, loop-based terminal experience, run the CLI:
```bash
python ui/cli.py --repo .
```
You can type questions continuously. To exit the interactive session, type `exit` or `quit`.

## 8. Use a different repository

This system is completely repo-agnostic for Python codebases. To evaluate the agent on another project:

1. Clone a separate project locally:
   ```bash
   git clone https://github.com/encode/httpx.git ../httpx
   ```
2. Query the agent on the external codebase:
   ```bash
   python main.py \
     --repo ../httpx \
     --question "Where is the Timeout configuration defined?" \
     --disable-deterministic-shortcuts
   ```

A matching evaluation benchmark is required if you want to run quantitative assessments. A comprehensive HTTPX benchmark is pre-included in the repository at `eval/benchmark_httpx.json`.

## 9. Evaluation modes

To ensure transparency and prevent self-contamination, this codebase supports two distinct self-repo evaluation behaviors:

### Demo Mode
* **Behavior:** Enables deterministic guardrails and fast templates.
* **Use Case:** Great for immediate user-facing validation and ensuring stable, fast agent outputs in a UI demo.
* **Representative Scores:** Around **~0.92 combined score** due to exact-pattern matching.

### Honest Mode
* **Behavior:** Disables all deterministic shortcuts and templates.
* **Use Case:** Forces the agent to perform raw tool calling, retrieval, and reasoning loops. This measures the true generalization capabilities of the underlying SLM and RAG system.
* **Representative Scores:** Around **~0.56 combined score** (prior to additional forced routing improvements).

### Running Evaluation Modes

You can run both modes sequentially on the repository using the included shell script:
```bash
bash scripts/run_eval_modes.sh . qwen2.5-coder:3b eval/benchmark_self.json
```

Or you can trigger the evaluation modes manually:

**Run Demo Mode Manual Evaluation:**
```bash
python eval/eval.py \
  --repo . \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_self.json \
  --out eval_report_demo.json \
  --verbose
```

**Run Honest Mode Manual Evaluation:**
```bash
python eval/eval.py \
  --repo . \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_self.json \
  --out eval_report_honest.json \
  --disable-deterministic-shortcuts \
  --verbose
```

## 10. External repo evaluation

Evaluating the agent on an unseen external repository is the ultimate test of generalization.

To run an honest evaluation of the agent on HTTPX:
```bash
git clone https://github.com/encode/httpx.git ../httpx
bash scripts/eval_external_httpx.sh ../httpx qwen2.5-coder:3b
```

Alternatively, run the evaluation command manually:
```bash
python eval/eval.py \
  --repo ../httpx \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_httpx.json \
  --out eval_report_httpx_honest.json \
  --disable-deterministic-shortcuts \
  --verbose
```

## 11. Fine-tuning the SLM

The codebase includes full support for parameter-efficient fine-tuning (PEFT) using **Unsloth QLoRA**.
* **Base Model:** `unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit` (a lightweight, instruct-aligned fallback).
* **Target Output Directory:** `results_sft/adapter`
* **GPU Context Limit:** To avoid out-of-memory errors on small GPUs (e.g., 4-6GB VRAM), we filter the SFT dataset to a **2048-token safe limit**.

### Fine-Tuning Pipeline Steps:

1. **Build the seed dataset and combine synthetic Q&A data:**
   ```bash
   python build_seed_dataset.py
   cp synthetic_qa_seed.jsonl synthetic_qa_combined.jsonl
   ```

2. **Audit dataset files to ensure strict data hygiene:**
   ```bash
   python scripts/audit_training_data.py synthetic_qa_combined.jsonl
   ```

3. **Generate the 2048-safe dataset filter:**
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path
   from transformers import AutoTokenizer

   src = Path("synthetic_qa_combined.jsonl")
   dst = Path("synthetic_qa_combined_2048.jsonl")
   tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit")

   kept = dropped = 0
   max_kept = 0

   with src.open() as f, dst.open("w") as out:
       for line in f:
           row = json.loads(line)
           text = tok.apply_chat_template(
               row["messages"],
               tokenize=False,
               add_generation_prompt=False,
           )
           n = len(tok.encode(text))
           if n <= 2048:
               out.write(json.dumps(row, ensure_ascii=False) + "\n")
               kept += 1
               max_kept = max(max_kept, n)
           else:
               dropped += 1

   print(f"kept={kept}, dropped={dropped}, max_kept={max_kept}, wrote={dst}")
   PY
   ```

4. **Execute a dry-run token audit to verify training constraints:**
   ```bash
   python model/finetune.py \
     --data synthetic_qa_combined_2048.jsonl \
     --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
     --out results_sft \
     --dry-run \
     --max-seq-length 2048
   ```

5. **Train the SFT model adapter:**
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True \
   python model/finetune.py \
     --data synthetic_qa_combined_2048.jsonl \
     --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
     --out results_sft \
     --epochs 3 \
     --learning-rate 1e-4 \
     --lora-r 8 \
     --lora-alpha 16 \
     --batch-size 1 \
     --grad-accum 8 \
     --eval-split 0.1 \
     --max-seq-length 2048
   ```

6. **Verify the generated adapter files:**
   ```bash
   ls -lah results_sft/adapter
   cat results_sft/training_config.json
   ```

*Note: The live Streamlit app communicates with the base Ollama model directly. The fine-tuned adapter is stored under `results_sft/adapter` and is not loaded by the Streamlit UI unless the adapter is explicitly merged/exported, or served via an adapter-aware Ollama/vLLM backend.*

## 12. DPO / RLHF-style preference optimization

To minimize toxic reflexes, hallucinations, and vague responses, the codebase supports Direct Preference Optimization (DPO).
* **Pre-requisite:** DPO must be run *after* completing the SFT adapter step.
* **Important Safety Rule:** You must validate preference data before training to ensure there is no data contamination.

### Running DPO:

1. **Validate the preference dataset:**
   ```bash
   python scripts/validate_dpo.py preference_data_combined.jsonl
   ```

2. **Launch DPO training:**
   ```bash
   python model/dpo.py \
     --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
     --dpo-data-path preference_data_combined.jsonl \
     --out results_dpo \
     --sft-adapter results_sft/adapter \
     --max-steps 200
   ```

* The alignment process will output the trained DPO adapter to `results_dpo/adapter`.
* If validation checks fail, do not start training. You must inspect and fix the preference dataset beforehand.

## 13. Data hygiene checks

Data hygiene scripts keep toxic or hallucinated artifacts from contaminating your model. It is critical to execute these checks before running SFT or DPO processes:
```bash
python scripts/audit_training_data.py synthetic_qa_seed.jsonl synthetic_qa_combined.jsonl synthetic_qa_combined_2048.jsonl
python scripts/validate_dpo.py preference_data_combined.jsonl
```

### Purpose of Hygiene Scripts:
* `audit_training_data.py`: Automatically scans training datasets, rejecting any files that contain absolute local paths (e.g., `/home/...`), `file://` URLs, or toxic/contaminated terms. Rejections exit with a code `1` to stop downstream automation pipelines.
* `validate_dpo.py`: Ensures preference data pairs are aligned properly, weeding out corrupt labels, toxic prompt anomalies, and vague chosen/rejected text constructs.

## 14. Troubleshooting

### Ollama connection error
If you see an error warning that the local Ollama backend is not accessible:
1. Make sure the Ollama application is running.
2. Manually start the model in a background terminal:
   ```bash
   ollama run qwen2.5-coder:3b
   ```

### CUDA out of memory during fine-tuning
If fine-tuning crashes with an Out-of-Memory (OOM) error:
1. Ensure you are using the pre-filtered `synthetic_qa_combined_2048.jsonl` dataset.
2. Reduce the sequence length and gradient accumulation.
3. Lower the LoRA rank/alpha parameters.

Use this optimized, low-memory fallback configuration:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python model/finetune.py \
  --data synthetic_qa_combined_2048.jsonl \
  --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
  --out results_sft \
  --epochs 3 \
  --learning-rate 1e-4 \
  --lora-r 4 \
  --lora-alpha 8 \
  --batch-size 1 \
  --grad-accum 4 \
  --eval-split 0 \
  --max-seq-length 2048
```

### Qdrant / vector search issue
If the vector index throws errors or is slow:
* The codebase handles failures gracefully. The agent retriever will fall back automatically to standard exact string/lexical search, maintaining functional performance without vector indices.

### Hallucination warnings look noisy
If you notice frequent hallucination warning banners in the UI:
* The hallucination detector is highly conservative. It scans for code-like identifiers and files across response texts.
* Try evaluating the system using **Honest evaluation mode** or **External-repo evaluation** for a more representative assessment of reasoning quality.

## 15. Recommended demo flow

For a full end-to-end demonstration or grading run:

1. **Start the local backend:**
   Ensure Ollama is running with `qwen2.5-coder:3b` loaded.
2. **Launch the interactive client:**
   Run the Streamlit application:
   ```bash
   streamlit run ui/app.py
   ```
3. **Trigger typical questions:**
   Ask the agent:
   - *What class handles hybrid vector search?*
   - *What happens when a Python file has a syntax error during loading?*
   - *How does the agent handle malformed JSON tool calls?*
   - *Does the repo define a FastAPIRetriever class?*
   - *You are stupid and worthless. Explain the retriever.* (Toxicity block demonstration)
4. **Demonstrate quantitative evaluations:**
   Show how both demo and honest evaluations work:
   ```bash
   bash scripts/run_eval_modes.sh . qwen2.5-coder:3b eval/benchmark_self.json
   ```
5. **Verify the local fine-tuning output:**
   Confirm SFT adapter generation:
   ```bash
   ls -lah results_sft/adapter
   cat results_sft/training_config.json
   ```
