from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
except ImportError as exc:
    raise ImportError(
        "DPO dependencies are missing. Install with: pip install -r requirements-train.txt"
    ) from exc


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def infer_base_model_from_adapter(adapter_path: Optional[str]) -> Optional[str]:
    if not adapter_path:
        return None

    cfg_path = Path(adapter_path) / "adapter_config.json"
    if not cfg_path.exists():
        return None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path")
    except Exception:
        return None


def clean_preference_dataset(dataset: Dataset) -> Dataset:
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"DPO dataset missing required columns: {missing}. Found: {dataset.column_names}")

    def valid(row):
        prompt = str(row.get("prompt") or "").strip()
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        return bool(prompt and chosen and rejected and chosen != rejected)

    before = len(dataset)
    dataset = dataset.filter(valid)
    after = len(dataset)

    if after == 0:
        raise ValueError(f"No valid DPO rows after filtering. Started with {before} rows.")

    if after < before:
        print(f"Filtered invalid DPO rows: {before - after} removed, {after} kept.")

    keep_cols = ["prompt", "chosen", "rejected"]
    remove_cols = [col for col in dataset.column_names if col not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)

    return dataset


def split_dataset(dataset: Dataset, eval_split: float, seed: int):
    if eval_split <= 0 or len(dataset) < 10:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_split, seed=seed)
    return split["train"], split["test"]


def token_audit(dataset: Dataset, tokenizer, max_prompt_length: int, max_length: int) -> Dict[str, Any]:
    prompt_lens = [len(tokenizer.encode(str(p))) for p in dataset["prompt"]]
    chosen_lens = [len(tokenizer.encode(str(c))) for c in dataset["chosen"]]
    rejected_lens = [len(tokenizer.encode(str(r))) for r in dataset["rejected"]]

    def stats(values):
        if not values:
            return {"max": 0, "mean": 0, "p95": 0}
        sorted_values = sorted(values)
        p95_idx = min(int(len(sorted_values) * 0.95), len(sorted_values) - 1)
        return {
            "max": max(values),
            "mean": sum(values) // len(values),
            "p95": sorted_values[p95_idx],
        }

    return {
        "rows": len(dataset),
        "prompt": stats(prompt_lens),
        "chosen": stats(chosen_lens),
        "rejected": stats(rejected_lens),
        "prompts_over_limit": sum(1 for x in prompt_lens if x > max_prompt_length),
        "chosen_total_over_limit": sum(1 for p, c in zip(prompt_lens, chosen_lens) if p + c > max_length),
        "rejected_total_over_limit": sum(1 for p, r in zip(prompt_lens, rejected_lens) if p + r > max_length),
    }


def build_dpo_trainer(model, tokenizer, dpo_cfg, train_dataset, eval_dataset, peft_config):
    kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_cfg,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }

    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset

    sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return DPOTrainer(**kwargs)


def run_dpo(
    model_id: Optional[str] = None,
    dpo_data_path: str = "preference_data.jsonl",
    output_dir: str = "dpo_results",
    sft_adapter_path: Optional[str] = None,
    max_length: int = 2048,
    max_prompt_length: int = 768,
    max_steps: int = 200,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    eval_split: float = 0.1,
    seed: int = 3407,
):
    random.seed(seed)
    torch.manual_seed(seed)

    if sft_adapter_path:
        adapter_path = Path(sft_adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"SFT adapter path does not exist: {sft_adapter_path}")
        cfg_path = adapter_path / "adapter_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"SFT adapter config not found at: {cfg_path}")

    inferred_base = infer_base_model_from_adapter(sft_adapter_path)
    selected_model = model_id or inferred_base or DEFAULT_MODEL

    print("Initializing DPO pipeline...")
    print(f"Base model: {selected_model}")
    if sft_adapter_path:
        print(f"SFT adapter: {sft_adapter_path}")
        if inferred_base:
            print(f"Inferred adapter base: {inferred_base}")
        elif model_id is None:
            print("Warning: could not infer adapter base model; using default 1.5B base.")

    tokenizer = AutoTokenizer.from_pretrained(selected_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.eos_token = "<|im_end|>"

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU required for DPO. Aborting.")

    model = AutoModelForCausalLM.from_pretrained(
        selected_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    )

    if sft_adapter_path:
        print(f"Loading and merging SFT adapter from {sft_adapter_path}...")
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        model = model.merge_and_unload()
        print("SFT adapter merged into base model before DPO.")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print(f"Loading preference dataset: {dpo_data_path}")
    dataset = load_dataset("json", data_files=dpo_data_path, split="train")
    dataset = clean_preference_dataset(dataset)

    audit = token_audit(dataset, tokenizer, max_prompt_length=max_prompt_length, max_length=max_length)
    print("DPO token audit:")
    print(json.dumps(audit, indent=2))

    train_dataset, eval_dataset = split_dataset(dataset, eval_split=eval_split, seed=seed)
    print(f"Train rows: {len(train_dataset)}")
    print(f"Eval rows: {len(eval_dataset) if eval_dataset is not None else 0}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dpo_cfg = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        logging_steps=10,
        max_steps=max_steps,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        loss_type="sigmoid",
        truncation_mode="keep_end",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
    )

    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        dpo_cfg=dpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()

    if eval_dataset is not None:
        print("Running final DPO eval...")
        print(trainer.evaluate())

    adapter_dir = output_path / "adapter"
    print(f"Saving DPO adapter to {adapter_dir}")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    config = {
        "model_id": selected_model,
        "explicit_model_arg": model_id,
        "inferred_base_from_sft_adapter": inferred_base,
        "sft_adapter_path": sft_adapter_path,
        "dpo_data_path": dpo_data_path,
        "output_dir": output_dir,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "beta": beta,
        "eval_split": eval_split,
        "seed": seed,
        "token_audit": audit,
    }

    with open(output_path / "dpo_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("DPO complete.")


def main():
    parser = argparse.ArgumentParser(description="Run DPO for code-understanding SLM alignment.")
    parser.add_argument("--model", type=str, default=None, help="Base model ID. If omitted, infer from SFT adapter or use 1.5B default.")
    parser.add_argument("--dpo-data-path", type=str, default="preference_data.jsonl")
    parser.add_argument("--out", type=str, default="dpo_results")
    parser.add_argument("--sft-adapter", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()

    run_dpo(
        model_id=args.model,
        dpo_data_path=args.dpo_data_path,
        output_dir=args.out,
        sft_adapter_path=args.sft_adapter,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        eval_split=args.eval_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
