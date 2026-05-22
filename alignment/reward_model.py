"""Reward-model training for PPO-based RLHF."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from alignment.feedback_schema import load_preferences
from alignment.reward_dataset import build_reward_examples, format_reward_input
from train.datasets import compute_dataset_hash


REWARD_IMPORT_ERROR = (
    "Reward-model training dependencies are missing. Install with: "
    "pip install -r requirements-train.txt"
)
RAW_PREFERENCE_DATASET = "preference_data_combined.jsonl"


def load_reward_training_dependencies(import_module=importlib.import_module) -> dict[str, Any]:
    """Load reward-training dependencies lazily with a clear error message."""
    try:
        modules = {
            "datasets": import_module("datasets"),
            "torch": import_module("torch"),
            "peft": import_module("peft"),
            "transformers": import_module("transformers"),
        }
    except ImportError as exc:
        raise ImportError(REWARD_IMPORT_ERROR) from exc
    return modules


def resolve_reward_backend(import_module=importlib.import_module) -> dict[str, str]:
    """Prefer Unsloth when possible, otherwise fall back to HF sequence classification."""
    try:
        import_module("unsloth")
    except Exception as exc:
        return {
            "preferred_backend": "unsloth",
            "selected_backend": "hf_sequence_classification_fallback",
            "reason": (
                "Unsloth is unavailable in the current environment; using Hugging Face sequence "
                f"classification fallback. Details: {exc}"
            ),
        }

    return {
        "preferred_backend": "unsloth",
        "selected_backend": "hf_sequence_classification_fallback",
        "reason": (
            "Installed Unsloth does not expose a stable reward-model classification head in this repo path; "
            "using Hugging Face AutoModelForSequenceClassification with PEFT."
        ),
    }


def _tokenize_reward_dataset(dataset, tokenizer, max_length: int):
    def preprocess(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = [float(label) for label in examples["label"]]
        return tokenized

    tokenized = dataset.map(preprocess, batched=True)
    remove_cols = [col for col in tokenized.column_names if col not in {"input_ids", "attention_mask", "labels"}]
    if remove_cols:
        tokenized = tokenized.remove_columns(remove_cols)
    return tokenized


def build_training_arguments(
    transformers,
    *,
    output_dir: str,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    eval_dataset,
):
    common_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "logging_steps": 10,
        "save_steps": max(50, max_steps),
        "report_to": [],
        "remove_unused_columns": False,
    }
    has_eval = eval_dataset is not None
    eval_kwargs: dict[str, Any] = {}
    if has_eval:
        eval_kwargs["evaluation_strategy"] = "steps"
        eval_kwargs["eval_steps"] = 25
    else:
        eval_kwargs["evaluation_strategy"] = "no"

    try:
        return transformers.TrainingArguments(**common_kwargs, **eval_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        fallback_eval_kwargs = dict(eval_kwargs)
        if "evaluation_strategy" in fallback_eval_kwargs:
            fallback_eval_kwargs["eval_strategy"] = fallback_eval_kwargs.pop("evaluation_strategy")
        return transformers.TrainingArguments(**common_kwargs, **fallback_eval_kwargs)


def _refuse_raw_preference_dataset(path: str, allow_raw_dataset: bool) -> None:
    if Path(path).name == RAW_PREFERENCE_DATASET and not allow_raw_dataset:
        raise ValueError(
            "Refusing to use raw preference_data_combined.jsonl directly for reward-model training. "
            "Use `data/final/preferences_train.jsonl` or pass --allow-raw-dataset to override."
        )


class RewardScorer:
    """Load a trained reward model and score prompt/response pairs."""

    def __init__(self, model_path: str, device: str = "cpu", import_module=importlib.import_module):
        modules = load_reward_training_dependencies(import_module=import_module)
        torch = modules["torch"]
        transformers = modules["transformers"]
        peft = modules["peft"]

        model_root = Path(model_path)
        adapter_path = model_root / "adapter" if (model_root / "adapter").exists() else model_root
        reward_config_path = model_root / "reward_config.json"
        reward_config = json.loads(reward_config_path.read_text(encoding="utf-8")) if reward_config_path.exists() else {}
        base_model = reward_config.get("base_model")
        if base_model is None:
            adapter_cfg = adapter_path / "adapter_config.json"
            if adapter_cfg.exists():
                base_model = json.loads(adapter_cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if base_model is None:
            raise ValueError("Could not infer reward-model base model. Provide a saved reward_config.json.")

        self.device = device
        tokenizer_source = str(adapter_path) if adapter_path.joinpath("tokenizer_config.json").exists() else base_model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model = transformers.AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
        adapter_config = adapter_path / "adapter_config.json"
        if adapter_config.exists():
            model = peft.PeftModel.from_pretrained(model, str(adapter_path))
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(str(adapter_path))
        model.eval()
        if device != "cpu":
            model.to(device)
        self._model = model
        self._torch = torch

    def score(self, prompt: str, response: str) -> float:
        text = format_reward_input(prompt, response)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        if self.device != "cpu":
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits.squeeze()
        return float(logits.item() if hasattr(logits, "item") else logits)


def train_reward_model(
    data_path: str,
    base_model: str,
    output_dir: str,
    eval_data_path: str | None = None,
    max_steps: int = 200,
    learning_rate: float = 1e-5,
    batch_size: int = 2,
    eval_split: float = 0.1,
    max_length: int = 1024,
    dry_run: bool = False,
    allow_raw_dataset: bool = False,
) -> dict[str, Any]:
    """Train or dry-run a reward model from canonical preference data."""
    _refuse_raw_preference_dataset(data_path, allow_raw_dataset)
    if eval_data_path:
        _refuse_raw_preference_dataset(eval_data_path, allow_raw_dataset)
    preferences = load_preferences(data_path, allow_pending=False)
    reward_rows = build_reward_examples(preferences)
    dataset_hash = compute_dataset_hash(data_path)
    eval_preferences = load_preferences(eval_data_path, allow_pending=False) if eval_data_path else None
    eval_reward_rows = build_reward_examples(eval_preferences) if eval_preferences is not None else None
    eval_dataset_hash = compute_dataset_hash(eval_data_path) if eval_data_path else None
    backend = resolve_reward_backend()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reward_config = {
        "base_model": base_model,
        "dataset_path": data_path,
        "dataset_hash": dataset_hash,
        "eval_dataset_path": eval_data_path,
        "eval_dataset_hash": eval_dataset_hash,
        "selected_backend": backend["selected_backend"],
        "backend_reason": backend["reason"],
        "loss_type": "binary_classification_fallback",
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "eval_split": eval_split,
        "max_length": max_length,
        "dry_run": dry_run,
    }
    (output_path / "reward_config.json").write_text(json.dumps(reward_config, indent=2), encoding="utf-8")

    if dry_run:
        return {
            "rows": len(reward_rows),
            "eval_rows": len(eval_reward_rows) if eval_reward_rows is not None else 0,
            "backend": backend["selected_backend"],
            "dry_run": True,
        }

    modules = load_reward_training_dependencies()
    datasets = modules["datasets"]
    torch = modules["torch"]
    peft = modules["peft"]
    transformers = modules["transformers"]

    dataset = datasets.Dataset.from_list(reward_rows)
    if eval_reward_rows is not None:
        split = {"train": dataset, "test": datasets.Dataset.from_list(eval_reward_rows)}
    else:
        split = dataset.train_test_split(test_size=eval_split, seed=3407) if eval_split > 0 and len(dataset) >= 10 else {"train": dataset, "test": None}

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = None
    try:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception:
        quantization_config = None

    model_kwargs: dict[str, Any] = {"num_labels": 1}
    if quantization_config is not None and torch.cuda.is_available():
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = transformers.AutoModelForSequenceClassification.from_pretrained(base_model, **model_kwargs)
    if hasattr(peft, "prepare_model_for_kbit_training") and quantization_config is not None and torch.cuda.is_available():
        model = peft.prepare_model_for_kbit_training(model)

    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = peft.get_peft_model(model, lora_config)

    train_dataset = _tokenize_reward_dataset(split["train"], tokenizer, max_length=max_length)
    eval_dataset = _tokenize_reward_dataset(split["test"], tokenizer, max_length=max_length) if split["test"] is not None else None

    training_args = build_training_arguments(
        transformers,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_dataset=eval_dataset,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (logits.reshape(-1) > 0).astype(int)
        labels = labels.reshape(-1)
        accuracy = float((preds == labels).mean()) if len(labels) else 0.0
        return {"accuracy": accuracy}

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    validation_report = trainer.evaluate() if eval_dataset is not None else {"eval_rows": 0}

    adapter_dir = output_path / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (output_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_path / "validation_report.json").write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    return {"metrics": metrics, "validation_report": validation_report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reward model from canonical preference pairs.")
    parser.add_argument("--data", required=True, help="Preference JSONL path.")
    parser.add_argument("--eval-data", default=None, help="Optional explicit validation preference JSONL path.")
    parser.add_argument("--base-model", required=True, help="Base model identifier.")
    parser.add_argument("--out", required=True, help="Output directory for reward-model artifacts.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-raw-dataset", action="store_true")
    args = parser.parse_args()

    result = train_reward_model(
        data_path=args.data,
        base_model=args.base_model,
        output_dir=args.out,
        eval_data_path=args.eval_data,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_split=args.eval_split,
        max_length=args.max_length,
        dry_run=args.dry_run,
        allow_raw_dataset=args.allow_raw_dataset,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
