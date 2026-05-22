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

This project is designed to run locally using Ollama. Pull and launch the default model (`qwen2.5-coder:3b`):
```bash
ollama pull qwen2.5-coder:3b
ollama run qwen2.5-coder:3b
```

Ollama must be running and accessible at:
```
http://localhost:11434
```

> [!TIP]
> If you have resources constraints or want to run a smaller model locally (e.g. `qwen2.5-coder:1.5b`), you can configure the agent to use it by setting the `OLLAMA_MODEL` environment variable before running scripts or booting Streamlit:
> * PowerShell: `$env:OLLAMA_MODEL="qwen2.5-coder:1.5b"`
> * Bash: `export OLLAMA_MODEL=qwen2.5-coder:1.5b`
> * Or simply change the model name directly in the sidebar input box in the Streamlit UI.

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
4. **Collect RLHF/DPO Preference Feedback:** Under each assistant response, you can click **👍 Like** or **👎 Correction**.
   * Clicking **👍 Like** logs the response as a positive/chosen sample in the preference dataset.
   * Clicking **👎 Correction** opens an input form where you can provide the correct/preferred response. Submitting it saves the correction as the `chosen` response and the original agent output as the `rejected` response.
   * All feedbacks are automatically appended to `training_data/preferences/preference_data_rlhf.jsonl` in the active repository directory, allowing you to feed human feedback directly into the DPO alignment pipeline (see Section 12).


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
  --out evaluation_reports/eval_report_demo.json \
  --verbose
```

**Run Honest Mode Manual Evaluation:**
```bash
python eval/eval.py \
  --repo . \
  --model qwen2.5-coder:3b \
  --benchmark eval/benchmark_self.json \
  --out evaluation_reports/eval_report_honest.json \
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
  --out evaluation_reports/eval_report_httpx_honest.json \
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
   cp training_data/sft/synthetic_qa_seed.jsonl training_data/sft/synthetic_qa_combined.jsonl
   ```

2. **Audit dataset files to ensure strict data hygiene:**
   ```bash
   python scripts/audit_training_data.py training_data/sft/synthetic_qa_combined.jsonl
   ```

3. **Generate the 2048-safe dataset filter:**
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path
   from transformers import AutoTokenizer

   src = Path("training_data/sft/synthetic_qa_combined.jsonl")
   dst = Path("training_data/sft/synthetic_qa_combined_2048.jsonl")
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
     --data training_data/sft/synthetic_qa_combined_2048.jsonl \
     --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
     --out results_sft \
     --dry-run \
     --max-seq-length 2048
   ```

5. **Train the SFT model adapter:**
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True \
   python model/finetune.py \
     --data training_data/synthetic_qa_combined_2048.jsonl \
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

### The RLHF Feedback & Collection Flow:
1. **Interactive Feedback in UI:** Under each assistant response in the Streamlit interface, you can click **👍 Like** or **👎 Correction**.
   * **Like**: Appends a preference row to `training_data/preferences/preference_data_rlhf.jsonl` marking the assistant's response as the `chosen` output.
   * **Correction**: Displays a form allowing you to type the correct or preferred response. Submitting it logs your typed response as `chosen` and the original agent output as `rejected` in `training_data/preferences/preference_data_rlhf.jsonl`.
2. **Data Preservation**: Human feedback is written to the standalone file `training_data/preferences/preference_data_rlhf.jsonl`. This isolates manual human annotations and prevents them from being accidentally overwritten or deleted when automated SFT/DPO runs clean and rebuild synthetic datasets.
3. **Pipeline Merging**: When you run the training pipeline (`bash scripts/train_pipeline.sh` on Bash or `.\scripts\train_pipeline.ps1` in PowerShell), Step 8 automatically merges `training_data/preferences/preference_data_rlhf.jsonl` along with all other preference datasets (e.g., failure-driven DPO rows and auto-generated SFT outputs) into a unified `training_data/preferences/preference_data_combined.jsonl`.

### Running the Automated Training Pipeline:
You can run SFT, SFT evaluations, preference compilation, validation checks, and DPO alignment sequentially using the unified pipeline script:
* **Linux/macOS (Bash):**
  ```bash
  bash scripts/train_pipeline.sh
  ```
* **Windows (PowerShell):**
  ```powershell
  .\scripts\train_pipeline.ps1
  ```

### Running DPO Manually:

1. **Validate the preference dataset:**
   ```bash
   python scripts/validate_dpo.py training_data/preferences/preference_data_combined.jsonl
   ```

2. **Launch DPO training:**
   ```bash
   python model/dpo.py \
     --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
     --dpo-data-path training_data/preferences/preference_data_combined.jsonl \
     --out results_dpo \
     --sft-adapter results_sft/adapter \
     --max-steps 200
   ```

* The alignment process will output the trained DPO adapter to `results_dpo/adapter`.
* If validation checks fail, do not start training. You must inspect and fix the preference dataset beforehand.

## 13. Data hygiene and validation checks (The "Clean" Script)

Data hygiene scripts keep toxic or contaminated artifacts from entering SFT/DPO training. Always run these checks before training:
```bash
python scripts/audit_training_data.py training_data/sft/synthetic_qa_seed.jsonl training_data/sft/synthetic_qa_combined.jsonl
python scripts/validate_dpo.py training_data/preferences/preference_data_combined.jsonl
```

### How the validation/clean scripts work:

#### 1. DPO Validation (`scripts/validate_dpo.py`)
This script audits preference datasets to ensure high-quality DPO training. It executes the following checks on every row:
* **Schema Integrity**: Verifies that the required keys (`prompt`, `chosen`, `rejected`) are present and non-empty.
* **Divergence**: Ensures that the `chosen` text is not identical to the `rejected` text.
* **Contamination Filtering**: Rejects rows where the `chosen` answer contains codebase leakage terms (e.g. references to nonexistent `generate_preferences` calls, system framework keywords, or local file system structures).
* **Vague Phrase Filtering**: Blocks chosen responses containing generic filler text such as `The answer is` to enforce professional output formatting.
* **False Refusal Detection**: Ensures that non-toxic questions do not mistakenly trigger toxic-refusal template responses (e.g. "I cannot process this abusive request") in the chosen response.
* **Required File Citations**: Cross-references metadata to verify that any code files marked as expected are actually cited in the chosen response.
* **Pipeline Block**: If any fatal violations are detected, the script outputs error details and exits with code `1`, halting automated training pipelines to protect model weights.

#### 2. SFT/General Auditing (`scripts/audit_training_data.py`)
* Automatically scans training datasets, rejecting any files containing absolute local paths (e.g., `/home/...`, `/Users/...`, `C:\`, `Documents/GitHub`), `file://` URLs, or toxic terms. This prevents the model from memorizing the host machine's directory paths.

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
1. Ensure you are using the pre-filtered `training_data/sft/synthetic_qa_combined_2048.jsonl` dataset.
2. Reduce the sequence length and gradient accumulation.
3. Lower the LoRA rank/alpha parameters.

Use this optimized, low-memory fallback configuration:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python model/finetune.py \
  --data training_data/sft/synthetic_qa_combined_2048.jsonl \
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

### Running Offline / Local Caching of Embedding Models
If your environment lacks internet access, or you want to prevent network connection attempts to Hugging Face Hub during start-up, you can pre-download the models:
1. Run the download helper script:
   ```bash
   python scripts/download_models.py
   ```
2. This script downloads the dense and sparse models directly into a local `.fastembed_cache` folder in the repository root.
3. The retriever automatically detects this directory at start-up and loads the models locally in offline mode (`local_files_only=True`).

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
