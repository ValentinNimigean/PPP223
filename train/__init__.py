"""Task-aware training package."""

from train.config import DEFAULT_MODEL, SFTTrainConfig
from train.datasets import compute_dataset_hash, load_and_prepare_records, validate_jsonl_dataset
from train.tasks import TASK_SCHEMAS, format_record_as_messages, validate_task_record

__all__ = [
    "DEFAULT_MODEL",
    "SFTTrainConfig",
    "TASK_SCHEMAS",
    "compute_dataset_hash",
    "format_record_as_messages",
    "load_and_prepare_records",
    "validate_jsonl_dataset",
    "validate_task_record",
]
