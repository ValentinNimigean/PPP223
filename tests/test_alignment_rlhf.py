import json
import subprocess
import sys
from pathlib import Path

import pytest

from alignment.feedback_schema import (
    PENDING_CORRECTION_TOKEN,
    convert_old_feedback_to_preferences,
    load_preferences,
    validate_preference_row,
)
from alignment.ppo_trainer import PPO_IMPORT_ERROR, build_ppo_config_dict, load_ppo_training_dependencies
from alignment.reward_dataset import build_prompt_dataset, build_reward_examples
from alignment.reward_model import REWARD_IMPORT_ERROR, load_reward_training_dependencies
from alignment.rewards import score_rule_based_reward


def test_validate_preference_row_adds_toxic_safety_labels():
    row = {
        "prompt": "You are stupid and worthless. Explain the retriever.",
        "chosen": "I cannot help with abusive language. Please rephrase.",
        "rejected": "The retriever lives in rag/retriever.py.",
        "source": "human",
    }

    validated = validate_preference_row(row)

    assert validated["safety_labels"]["toxic_prompt"] is True
    assert "stupid" in validated["safety_labels"]["toxicity_terms"]


def test_validate_preference_row_rejects_contaminated_paths():
    row = {
        "prompt": "Use /home/valentin/Documents/GitHub/PPP223/model/agent.py",
        "chosen": "Chosen",
        "rejected": "Rejected",
        "source": "synthetic",
    }

    with pytest.raises(ValueError, match="contaminated local paths"):
        validate_preference_row(row)


def test_convert_old_feedback_to_preferences(tmp_path):
    source = tmp_path / "human_feedback.jsonl"
    target = tmp_path / "preferences.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Where is HybridRetriever defined?",
                        "chosen": "It is defined in rag/retriever.py.",
                        "rejected": "It is in model/agent.py.",
                        "source": "human",
                        "rating": "negative",
                        "timestamp": "2025-01-01T00:00:00Z",
                        "correction_provided": True,
                    }
                ),
                json.dumps(
                    {
                        "prompt": "Explain the loader.",
                        "chosen": PENDING_CORRECTION_TOKEN,
                        "rejected": "Bad answer",
                        "source": "human",
                        "rating": "negative",
                        "timestamp": "2025-01-01T00:00:01Z",
                        "correction_provided": False,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    stats = convert_old_feedback_to_preferences(str(source), str(target))
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]

    assert stats["converted"] == 2
    assert stats["pending"] == 1
    assert rows[0]["source"] == "human"
    assert rows[1]["metadata"]["pending"] is True


def test_load_preferences_rejects_pending_by_default(tmp_path):
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "Prompt",
                "chosen": PENDING_CORRECTION_TOKEN,
                "rejected": "Rejected",
                "source": "human",
                "metadata": {"pending": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pending rows require allow_pending=True"):
        load_preferences(str(path))


def test_reward_dataset_creation_and_prompt_builder(tmp_path):
    pref_path = tmp_path / "prefs.jsonl"
    pref_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "How do refunds work?",
                        "chosen": "Refunds are allowed within 30 days.",
                        "rejected": "No refunds.",
                        "source": "human",
                        "metadata": {"expected_files": ["policy.md"]},
                    }
                ),
                json.dumps(
                    {
                        "prompt": "What is the shipping time?",
                        "chosen": "Shipping takes 3 business days.",
                        "rejected": "Unknown.",
                        "source": "eval_failure",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    synthetic_path = tmp_path / "synthetic.jsonl"
    synthetic_path.write_text(json.dumps({"question": "Synthetic prompt", "answer": "Answer"}) + "\n", encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps([{"question": "Benchmark prompt"}]), encoding="utf-8")

    preferences = load_preferences(str(pref_path))
    reward_rows = build_reward_examples(preferences)
    prompt_rows = build_prompt_dataset(
        str(pref_path),
        synthetic_paths=[str(synthetic_path)],
        benchmark_paths=[str(benchmark_path)],
    )

    assert len(reward_rows) == 4
    assert reward_rows[0]["label"] == 1
    assert reward_rows[1]["label"] == 0
    assert any(row["prompt"] == "Synthetic prompt" for row in prompt_rows)
    assert any(row["prompt"] == "Benchmark prompt" for row in prompt_rows)


def test_rule_based_reward_scores_components():
    reward = score_rule_based_reward(
        prompt="Where is HybridRetriever defined?",
        response="`HybridRetriever` is defined in rag/retriever.py.",
        metadata={"expected_files": ["rag/retriever.py"], "expected_entities": ["HybridRetriever"]},
    )

    assert reward["components"]["grounding_bonus"] > 0
    assert reward["components"]["hallucination_penalty"] == 0
    assert reward["total"] > 0


def test_rule_based_reward_penalizes_bad_outputs():
    reward = score_rule_based_reward(
        prompt="Explain the retriever.",
        response='{"name": "semantic_search", "arguments": ',
        metadata={"expected_files": ["rag/retriever.py"]},
    )

    assert reward["components"]["malformed_tool_call_penalty"] < 0
    assert reward["total"] < 0


def test_build_ppo_config_dict():
    cfg = build_ppo_config_dict(
        prompts_path="data/rlhf_prompts.jsonl",
        sft_model="results_sft/adapter",
        reward_model="results_reward",
        base_model="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
        output_dir="results_ppo",
        total_episodes=1000,
        learning_rate=3e-6,
        mini_batch_size=1,
        batch_size=4,
        target_kl=0.1,
        dry_run=True,
    )

    assert cfg["uses_trl_ppo_trainer"] is True
    assert cfg["dry_run"] is True
    assert cfg["target_kl"] == 0.1


def test_dependency_failures_are_clear():
    def broken_import(name):
        raise ImportError(name)

    with pytest.raises(ImportError, match=REWARD_IMPORT_ERROR):
        load_reward_training_dependencies(import_module=broken_import)

    with pytest.raises(ImportError, match=PPO_IMPORT_ERROR):
        load_ppo_training_dependencies(import_module=broken_import)


def test_train_rlhf_ppo_dry_run_script(tmp_path):
    preferences = tmp_path / "preferences.jsonl"
    prompts_out = tmp_path / "rlhf_prompts.jsonl"
    sft_model = tmp_path / "results_sft" / "adapter"
    sft_model.mkdir(parents=True)
    reward_out = tmp_path / "results_reward"
    ppo_out = tmp_path / "results_ppo"
    preferences.write_text(
        json.dumps(
            {
                "prompt": "Where is HybridRetriever defined?",
                "chosen": "It is defined in rag/retriever.py.",
                "rejected": "It is in model/agent.py.",
                "source": "human",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "scripts/train_rlhf_ppo.sh",
            "--dry-run",
            "--preferences",
            str(preferences),
            "--prompts-out",
            str(prompts_out),
            "--sft-model",
            str(sft_model),
            "--reward-out",
            str(reward_out),
            "--ppo-out",
            str(ppo_out),
        ],
        cwd="/home/valentin/Documents/GitHub/PPP223",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Dry-run completed" in completed.stdout
    assert prompts_out.exists()
    assert (reward_out / "reward_config.json").exists()
    assert (ppo_out / "ppo_config.json").exists()
