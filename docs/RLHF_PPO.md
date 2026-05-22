# RLHF With PPO

This project satisfies the “Incorporate RLHF” requirement through a PPO-based RLHF pipeline. DPO remains in the repository only as an optional baseline.

Training-ready inputs must come from `data/final/*.jsonl`. Raw helper datasets such as `synthetic_qa_combined_2048.jsonl` and `preference_data_combined.jsonl` are not valid training inputs.

This PPO implementation is pinned to `trl==0.11.4`. Install the training stack from `requirements-train.txt` or run `pip install trl==0.11.4` explicitly if your environment has a different TRL version.

## What RLHF Means Here

The RLHF pipeline has four stages:

1. Supervised fine-tuning initializes the policy from task or repository Q&A data.
2. Human or synthetic preference pairs train a reward model.
3. PPO optimizes the policy against that reward model with a KL penalty to a frozen reference model.
4. Evaluation compares the base model, SFT adapter, and PPO adapter.

## Why PPO

PPO is the primary RLHF implementation because it directly optimizes the policy against scalar rewards while controlling policy drift through a reference-model KL penalty. That makes it the canonical answer for the project’s RLHF requirement.

## Preference Data Flow

Canonical preference rows live in the schema implemented by `alignment/feedback_schema.py`:

```json
{
  "prompt": "string",
  "chosen": "string",
  "rejected": "string",
  "source": "human | synthetic | eval_failure",
  "rating": "optional string",
  "metadata": {},
  "safety_labels": {}
}
```

Legacy `human_feedback.jsonl` rows can be converted with:

```bash
python - <<'PY'
from alignment.feedback_schema import convert_old_feedback_to_preferences
print(convert_old_feedback_to_preferences("human_feedback.jsonl", "preference_data_canonical.jsonl"))
PY
```

## Reward Model Training

Reward-model training uses canonical preference pairs and converts them into positive and negative examples:

- `prompt + chosen -> 1`
- `prompt + rejected -> 0`

Before training, audit the final splits and validate both preference files:

```bash
python scripts/audit_datasets.py \
  --sft data/final/sft_train.jsonl data/final/sft_val.jsonl \
  --preferences data/final/preferences_train.jsonl data/final/preferences_val.jsonl
python -m alignment.validate_preferences data/final/preferences_train.jsonl
python -m alignment.validate_preferences data/final/preferences_val.jsonl
```

The training entry point is:

```bash
python -m alignment.reward_model \
  --data data/final/preferences_train.jsonl \
  --eval-data data/final/preferences_val.jsonl \
  --base-model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
  --out results_reward \
  --max-steps 200
```

The reward-model CLI refuses `preference_data_combined.jsonl` directly unless `--allow-raw-dataset` is passed.

Unsloth is preferred where compatible, but the current reward-model path falls back to `AutoModelForSequenceClassification` with PEFT because that head is the stable option for reward scoring in this repository.

## PPO Training

PPO uses:

- the SFT adapter as the initial policy
- a frozen reference model for KL regularization
- the trained reward model for scalar rewards
- additional rule-based reward shaping from `alignment/rewards.py`

Run PPO with:

```bash
python -m alignment.ppo_trainer \
  --prompts data/rlhf_prompts.jsonl \
  --sft-model results_sft/adapter \
  --reward-model results_reward \
  --base-model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
  --out results_ppo \
  --total-episodes 1000 \
  --learning-rate 3e-6 \
  --mini-batch-size 1 \
  --batch-size 4 \
  --target-kl 0.1
```

For a smoke test, run PPO first with `--total-episodes 100` and confirm the dry-run and short training behave correctly before starting the full `--total-episodes 1000` run.

If reward-model training has already completed, you can retry PPO directly without retraining the reward model:

```bash
python -m alignment.ppo_trainer \
  --prompts data/rlhf_prompts.jsonl \
  --sft-model results_sft/adapter \
  --reward-model results_reward \
  --base-model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
  --out results_ppo \
  --total-episodes 100 \
  --learning-rate 3e-6 \
  --mini-batch-size 1 \
  --batch-size 4 \
  --target-kl 0.1
```

Rule-based shaping adds bonuses or penalties for:

- grounded answers that cite retrieved files
- hallucinated files or missing expected entities
- toxic generations
- malformed tool calls
- missing end-of-response punctuation / EOS-like endings

## Unsloth Usage

Unsloth is used in the project for efficient SFT and adapter-centric model workflows. PPO itself is implemented with TRL’s `PPOTrainer`, and the reward-model path falls back to Hugging Face sequence classification where a dedicated Unsloth reward head is not a clean fit.

## Dry-Run

The full RLHF orchestration script supports CPU-safe dry-runs:

```bash
bash scripts/train_rlhf_ppo.sh \
  --dry-run \
  --preferences data/final/preferences_train.jsonl \
  --preferences-val data/final/preferences_val.jsonl \
  --sft-prompts-source data/final/sft_train.jsonl
```

Dry-run behavior:

- audits the final SFT and preference datasets first
- validates canonical preference data
- builds the PPO prompt dataset from `data/final/preferences_train.jsonl` and `data/final/sft_train.jsonl`
- validates reward-model and PPO configs
- prints the exact training and evaluation commands
- does not start long GPU training

## Full PPO Training

```bash
bash scripts/train_rlhf_ppo.sh \
  --train \
  --preferences data/final/preferences_train.jsonl \
  --preferences-val data/final/preferences_val.jsonl \
  --sft-prompts-source data/final/sft_train.jsonl
```

If `results_reward/adapter` already exists and you only want to retry PPO, add `--skip-reward-if-exists` to reuse the existing reward model artifacts.

## Evaluation

Compare base, SFT, and PPO using:

```bash
python -m eval.eval \
  --repo . \
  --model results_ppo/adapter \
  --backend hf-peft \
  --base-model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
  --reward-model results_reward \
  --benchmark eval/benchmark_self.json \
  --out eval_report_ppo.json \
  --verbose
```

The report includes:

- `entity_score`
- `file_score`
- `combined_score`
- `hallucination_penalty`
- toxicity flags
- average reward-model score when a reward model is supplied

## DPO Status

DPO is still available in `model/dpo.py`, but it is only an optional baseline. PPO-based RLHF is the main alignment path in this repository.
