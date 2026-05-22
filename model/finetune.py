from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from unsloth import FastLanguageModel
    from datasets import Dataset, load_dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    raise ImportError(
        "Fine-tuning dependencies are missing. Install with: "
        "pip install -r requirements-train.txt"
    ) from exc


DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Qwen chat templates can choke on None content.
    Keep tool_calls intact, but replace None content with empty string.
    """
    sanitized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        msg = dict(message)
        if msg.get("content") is None:
            msg["content"] = ""
        sanitized.append(msg)
    return sanitized


def load_tool_schemas() -> List[Dict[str, Any]]:
    try:
        from model.tools import AgentTools
    except ModuleNotFoundError:
        from tools import AgentTools
    return AgentTools.get_tool_schemas()


def format_dataset(dataset: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    tool_schemas = load_tool_schemas()

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            messages = sanitize_messages(messages)
            has_tool_use = any(
                isinstance(m, dict) and (m.get("tool_calls") or m.get("role") == "tool")
                for m in messages
            )

            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tools=tool_schemas if has_tool_use else None,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Last resort: render without tools schema.
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    remove_cols = [col for col in dataset.column_names if col != "text"]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)

    return dataset


def token_audit(dataset: Dataset, tokenizer, max_seq_length: int) -> Dict[str, Any]:
    lengths = [len(tokenizer.encode(row["text"])) for row in dataset]
    if not lengths:
        return {
            "rows": 0,
            "max": 0,
            "mean": 0,
            "p95": 0,
            "over_limit": 0,
        }

    sorted_lengths = sorted(lengths)
    p95_index = min(int(len(sorted_lengths) * 0.95), len(sorted_lengths) - 1)

    return {
        "rows": len(lengths),
        "max": max(lengths),
        "mean": sum(lengths) // len(lengths),
        "p95": sorted_lengths[p95_index],
        "over_limit": sum(1 for length in lengths if length > max_seq_length),
    }


def tokenize_dataset(dataset: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    def tokenize_batch(examples):
        batch = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        batch["labels"] = [ids[:] for ids in batch["input_ids"]]
        return batch

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


def split_dataset(dataset: Dataset, eval_split: float, seed: int):
    if eval_split <= 0 or len(dataset) < 10:
        return dataset, None

    split = dataset.train_test_split(test_size=eval_split, seed=seed)
    return split["train"], split["test"]


def build_sft_trainer(model, tokenizer, cfg, train_dataset, eval_dataset=None):
    kwargs = {
        "model": model,
        "args": cfg,
        "train_dataset": train_dataset,
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    }

    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset

    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return SFTTrainer(**kwargs)


def run_finetuning(
    model_id: str = DEFAULT_MODEL,
    dataset_path: str = "synthetic_qa.jsonl",
    output_dir: str = "results_sft",
    dry_run: bool = False,
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
):
    random.seed(seed)
    torch.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model/tokenizer: {model_id}")
    print(
        "Config: "
        f"max_seq_length={max_seq_length}, "
        f"LoRA r={lora_r}, "
        f"LoRA alpha={lora_alpha}, "
        f"batch_size={batch_size}, "
        f"grad_accum={grad_accum}, "
        f"lr={learning_rate}, "
        f"epochs={epochs}"
    )

    if dry_run:
        from transformers import AutoTokenizer

        print("Dry run mode: loading tokenizer only.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = None
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

    if tokenizer.pad_token is None or tokenizer.pad_token == "<|endoftext|>":
        tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.eos_token = "<|im_end|>"

    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    if "messages" not in dataset.column_names:
        raise ValueError(f"Expected dataset column `messages`; found {dataset.column_names}")

    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Using max_samples={len(dataset)}")

    dataset = format_dataset(dataset, tokenizer, max_seq_length=max_seq_length)

    audit = token_audit(dataset, tokenizer, max_seq_length)
    print(
        "Dataset token audit: "
        f"rows={audit['rows']}, "
        f"max={audit['max']}, "
        f"mean={audit['mean']}, "
        f"p95={audit['p95']}, "
        f"over_limit={audit['over_limit']}"
    )

    if dry_run:
        print("\n--- DRY RUN: first 3 formatted rows ---")
        for i in range(min(3, len(dataset))):
            text = dataset[i]["text"]
            tokens = len(tokenizer.encode(text))
            print(f"\nRow {i + 1} ({tokens} tokens):")
            print(text[:1200])
            if len(text) > 1200:
                print("...")
        print("Dry run complete. Exiting without training.")
        return

    dataset = tokenize_dataset(dataset, tokenizer, max_seq_length=max_seq_length)

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU required for training. Aborting.")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    train_dataset, eval_dataset = split_dataset(dataset, eval_split=eval_split, seed=seed)
    print(f"Train rows: {len(train_dataset)}")
    print(f"Eval rows: {len(eval_dataset) if eval_dataset is not None else 0}")

    cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=max_seq_length,
        packing=False,
        report_to=[],
    )

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting QLoRA SFT training...")
    trainer.train()

    if eval_dataset is not None:
        print("Running final eval on held-out split...")
        metrics = trainer.evaluate()
        print(metrics)

    adapter_dir = output_path / "adapter"
    print(f"Saving LoRA adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    training_config = {
        "model_id": model_id,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "epochs": epochs,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "eval_split": eval_split,
        "seed": seed,
        "max_samples": max_samples,
        "token_audit": audit,
    }
    with open(output_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    print("SFT complete.")


def main():
    parser = argparse.ArgumentParser(description="Run QLoRA SFT for Python code-understanding agent.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=str, default="synthetic_qa.jsonl")
    parser.add_argument("--out", type=str, default="results_sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    run_finetuning(
        model_id=args.model,
        dataset_path=args.data,
        output_dir=args.out,
        dry_run=args.dry_run,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_split=args.eval_split,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
