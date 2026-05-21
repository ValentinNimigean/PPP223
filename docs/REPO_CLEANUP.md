# Repository Cleanup

This repository keeps source code, benchmark inputs under `eval/`, and seed/training datasets that active scripts still read.

Ignored and removed generated artifacts:
- Root-level `eval_report*.json` files are outputs from `eval/eval.py` and shell wrappers.
- `results/exp_*.json` files are experiment outputs, not source inputs.
- `results_sft/`, `results_dpo/`, and `results_sft_test/` contain fine-tuning outputs and checkpoints.
- `qdrant_storage/`, `fastembed_cache/`, and `unsloth_compiled_cache/` are local vector DB or model caches.
- `scratch/` contains ad hoc local test scripts and output logs.
- `human_feedback.jsonl`, `synthetic_qa_auto.jsonl`, `preference_data_auto.jsonl`, `preference_data_failures*.jsonl`, and `test_pref*.jsonl` are runtime or intermediate datasets that can be regenerated.
- `repomix-output.txt` is a generated export artifact.

Intentionally kept:
- Files under `eval/` because they are benchmark inputs and evaluation code.
- `synthetic_qa_seed.jsonl`, `synthetic_qa_combined.jsonl`, and `synthetic_qa_combined_2048.jsonl` because the documented training flow still references them.
- `preference_data_combined.jsonl` because the Streamlit UI merges accepted human feedback into it and the training pipeline validates and trains from it.

Validation after cleanup should include:
- `pytest`
- `python -m compileall .`
