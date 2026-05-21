"""Configuration types for supervised fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
DEFAULT_TASK = "chat"


@dataclass
class SFTTrainConfig:
    task: str = DEFAULT_TASK
    base_model: str = DEFAULT_MODEL
    dataset_path: str = "data/final/sft_train.jsonl"
    val_dataset_path: str | None = "data/final/sft_val.jsonl"
    output_dir: str = "results_sft"
    dry_run: bool = False
    allow_raw_dataset: bool = False
    epochs: int = 2
    max_seq_length: int = 2048
    lora_r: int = 8
    lora_alpha: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 16
    eval_split: float = 0.1
    seed: int = 3407
    max_samples: int | None = None

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "max_seq_length": self.max_seq_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "eval_split": self.eval_split,
            "seed": self.seed,
            "max_samples": self.max_samples,
            "dry_run": self.dry_run,
        }

    def to_training_config(
        self,
        *,
        dataset_hash: str,
        token_audit: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "task": self.task,
            "base_model": self.base_model,
            "dataset_path": self.dataset_path,
            "val_dataset_path": self.val_dataset_path,
            "dataset_hash": dataset_hash,
            "output_dir": self.output_dir,
            "hyperparameters": self.hyperparameters(),
            "token_audit": token_audit,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
