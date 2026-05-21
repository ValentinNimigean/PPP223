from __future__ import annotations

from pathlib import Path


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
PREFERRED_ADAPTER_CANDIDATES = (
    ("PPO", Path("results_ppo/adapter")),
    ("SFT", Path("results_sft/adapter")),
)


def recommended_training_commands() -> list[str]:
    return [
        "python model/finetune.py --task chat --data data/final/sft_train.jsonl --val-data data/final/sft_val.jsonl --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit --out results_sft",
        "bash scripts/train_rlhf_ppo.sh --train",
    ]


def missing_adapter_message(root_dir: str | Path = ".") -> str:
    commands = recommended_training_commands()
    return (
        "No fine-tuned adapter was found. The default system uses the local fine-tuned SLM path "
        "(prefer `results_ppo/adapter`, then `results_sft/adapter`). "
        "Create one with:\n"
        f"1. {commands[0]}\n"
        f"2. {commands[1]}"
    )


def resolve_preferred_adapter(root_dir: str | Path = ".") -> tuple[str, str]:
    base = Path(root_dir)
    for label, relative_path in PREFERRED_ADAPTER_CANDIDATES:
        candidate = (base / relative_path).resolve()
        if candidate.exists():
            return str(candidate), label
    raise FileNotFoundError(missing_adapter_message(root_dir))
