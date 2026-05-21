# How to Use Mobtrap / PPP223

## 1. What this project does

This project is an agentic, Small Language Model (SLM)-based codebase assistant designed for deep Python code understanding. Instead of relying on naive character-count text chunking, it parses Python repositories structurally using Abstract Syntax Tree (AST) definitions (falling back to tree-sitter when needed). It constructs a unified "Repo Map" that outlines classes, methods, and functions, allowing the model to quickly navigate files and contextually resolve hierarchical symbols.

The core pipeline features structural codebase ingestion, high-fidelity hybrid retrieval (combining dense vector Qdrant embeddings and lexical BM25 fallback), and an agentic loop with native tool access (`grep_search` and `semantic_search`). Additionally, it includes full pipelines for dataset compilation (synthetic reasoning generation), model evaluation, toxicity/hallucination checks, and alignment via Supervised Fine-Tuning (SFT) and PPO-based RLHF. DPO remains available only as an optional baseline.

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

If you plan to run local fine-tuning or PPO/DPO alignment, install the training dependencies:
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

For a one-shot CLI question using the default Ollama backend:
```bash
python -m ui.cli --repo . --question "What class handles hybrid vector search?"
```

You can also select the backend explicitly:
```bash
python -m ui.cli --repo . --backend ollama --question "What class handles hybrid vector search?"
```

To run inference with a fine-tuned LoRA/PEFT adapter instead of Ollama:
```bash
python -m ui.cli \
  --repo . \
  --backend hf-peft \
  --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter-path results_dpo/adapter \
  --question "What class handles hybrid vector search?"
```

### CLI backend notes
* `--backend ollama` is the default and preserves the existing OpenAI-compatible Ollama path.
* `--backend hf-peft` requires `--adapter-path`.
* `--device auto` is recommended for local adapter inference because it allows Hugging Face to choose a safe device map automatically.
* If `hf-peft` import fails, install the local inference stack:
  ```bash
  pip install transformers peft accelerate
  ```
* Some quantized setups may also need:
  ```bash
  pip install bitsandbytes
  ```

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
* **Training inputs:** Always train from the cleaned final splits, not the raw synthetic helper files.

### Fine-Tuning Pipeline Steps:

1. **Prepare the cleaned final datasets:**
   ```bash
   python scripts/clean_datasets.py
   python scripts/split_datasets.py
   ```

2. **Audit the final SFT and preference splits before training:**
   ```bash
   python scripts/audit_datasets.py \
     --sft data/final/sft_train.jsonl data/final/sft_val.jsonl \
     --preferences data/final/preferences_train.jsonl data/final/preferences_val.jsonl
   ```

3. **Execute a dry-run token audit to verify training constraints:**
   ```bash
   python model/finetune.py \
     --data data/final/sft_train.jsonl \
     --val-data data/final/sft_val.jsonl \
     --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
     --out results_sft \
     --dry-run \
     --max-seq-length 2048
   ```

4. **Train the SFT model adapter:**
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True \
   python model/finetune.py \
     --data data/final/sft_train.jsonl \
     --val-data data/final/sft_val.jsonl \
     --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
     --out results_sft \
     --epochs 3 \
     --learning-rate 1e-4 \
     --lora-r 8 \
     --lora-alpha 16 \
     --batch-size 1 \
     --grad-accum 8 \
     --eval-split 0 \
     --max-seq-length 2048
   ```

5. **Verify the generated adapter files:**
   ```bash
   ls -lah results_sft/adapter
   cat results_sft/training_config.json
   ```

*Note: The live Streamlit app communicates with the base Ollama model directly. The fine-tuned adapter is stored under `results_sft/adapter` and is not loaded by the Streamlit UI unless the adapter is explicitly merged/exported, or served via an adapter-aware Ollama/vLLM backend.*

## 12. Optional DPO Baseline

PPO-based RLHF is the main alignment path in this repository. Direct Preference Optimization (DPO) is kept only as an optional baseline.
* **Pre-requisite:** DPO must be run *after* completing the SFT adapter step.
* **Important Safety Rule:** You must validate preference data before training to ensure there is no data contamination.

### Running DPO:

1. **Validate the preference dataset:**
   ```bash
   python -m alignment.validate_preferences data/final/preferences_train.jsonl
   python -m alignment.validate_preferences data/final/preferences_val.jsonl
   ```

2. **Launch DPO training:**
   ```bash
   python model/dpo.py \
     --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
     --dpo-data-path data/final/preferences_train.jsonl \
     --out results_dpo \
     --sft-adapter results_sft/adapter \
     --max-steps 200
   ```

* The alignment process will output the trained DPO adapter to `results_dpo/adapter`.
* If validation checks fail, do not start training. You must inspect and fix the preference dataset beforehand.

## 13. Data hygiene checks

Data hygiene scripts keep toxic or hallucinated artifacts from contaminating your model. It is critical to execute these checks before running SFT, reward-model, PPO, or optional DPO processes:
```bash
python scripts/audit_datasets.py \
  --sft data/final/sft_train.jsonl data/final/sft_val.jsonl \
  --preferences data/final/preferences_train.jsonl data/final/preferences_val.jsonl
python -m alignment.validate_preferences data/final/preferences_train.jsonl
python -m alignment.validate_preferences data/final/preferences_val.jsonl
```

### Purpose of Hygiene Scripts:
* `audit_datasets.py`: Audits the final SFT and preference splits for duplicates, empty answers, malformed rows, contamination, and split-readiness. Rejections exit with a code `1` to stop downstream automation pipelines.
* `alignment.validate_preferences`: Ensures preference pairs are aligned properly, weeding out corrupt labels, toxic prompt anomalies, and vague chosen/rejected text constructs.

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
1. Ensure you are using `data/final/sft_train.jsonl` and `data/final/sft_val.jsonl`, not the raw synthetic helper datasets.
2. Reduce the sequence length and gradient accumulation.
3. Lower the LoRA rank/alpha parameters.

Use this optimized, low-memory fallback configuration:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python model/finetune.py \
  --data data/final/sft_train.jsonl \
  --val-data data/final/sft_val.jsonl \
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
