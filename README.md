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
* **Primary Models:** Qwen-2.5-Coder (7B/14B) is used as the baseline for code reasoning, with Phi-4-mini as a lightweight alternative for basic tasks.
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
