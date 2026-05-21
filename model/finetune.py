from __future__ import annotations

from train.config import DEFAULT_TASK, SFTTrainConfig
from train.sft import build_arg_parser, main as train_main, run_finetuning as _run_finetuning


def run_finetuning(
    model_id: str = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
    dataset_path: str = "data/final/sft_train.jsonl",
    val_dataset_path: str | None = "data/final/sft_val.jsonl",
    output_dir: str = "results_sft",
    dry_run: bool = False,
    allow_raw_dataset: bool = False,
    epochs: int = 2,
    max_seq_length: int = 2048,
    lora_r: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 16,
    eval_split: float = 0.1,
    seed: int = 3407,
    max_samples: int | None = None,
    task: str = DEFAULT_TASK,
) -> None:
    """Compatibility wrapper for the legacy fine-tuning Python API."""
    cfg = SFTTrainConfig(
        task=task,
        base_model=model_id,
        dataset_path=dataset_path,
        val_dataset_path=val_dataset_path,
        output_dir=output_dir,
        dry_run=dry_run,
        allow_raw_dataset=allow_raw_dataset,
        epochs=epochs,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        eval_split=eval_split,
        seed=seed,
        max_samples=max_samples,
    )
    _run_finetuning(cfg)


def main() -> None:
    """Compatibility wrapper for the legacy CLI entry point."""
    train_main()


__all__ = [
    "build_arg_parser",
    "main",
    "run_finetuning",
]


if __name__ == "__main__":
    main()
