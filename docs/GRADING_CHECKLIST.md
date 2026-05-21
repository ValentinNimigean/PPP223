The project uses a fine-tuned SLM as the main model path.

PPO-based RLHF is the main RLHF implementation in this repository. DPO is optional only.

Training must use the final split datasets under `data/final/`:
- `data/final/sft_train.jsonl`
- `data/final/sft_val.jsonl`
- `data/final/preferences_train.jsonl`
- `data/final/preferences_val.jsonl`

Raw helper files such as `synthetic_qa_combined_2048.jsonl` and `preference_data_combined.jsonl` are not valid training inputs.
