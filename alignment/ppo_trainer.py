"""PPO-based RLHF training built on Hugging Face TRL."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from alignment.reward_model import REWARD_IMPORT_ERROR, RewardScorer
from alignment.rewards import score_rule_based_reward


PPO_IMPORT_ERROR = (
    "PPO training dependencies are missing. Install with: "
    "pip install -r requirements-train.txt"
)
PPO_TRL_VERSION = "0.11.4"
PPO_TRL_COMPAT_ERROR = (
    f"This PPO implementation requires trl=={PPO_TRL_VERSION}. "
    f"Run: pip install trl=={PPO_TRL_VERSION}"
)


def load_ppo_training_dependencies(import_module=importlib.import_module) -> dict[str, Any]:
    """Load PPO training dependencies lazily with a clear error message."""
    try:
        return {
            "torch": import_module("torch"),
            "trl": import_module("trl"),
            "transformers": import_module("transformers"),
            "peft": import_module("peft"),
        }
    except ImportError as exc:
        raise ImportError(PPO_IMPORT_ERROR) from exc


def build_ppo_config_dict(
    *,
    prompts_path: str,
    sft_model: str,
    reward_model: str,
    base_model: str,
    output_dir: str,
    total_episodes: int,
    learning_rate: float,
    mini_batch_size: int,
    batch_size: int,
    target_kl: float,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build a serializable PPO config record."""
    return {
        "prompts_path": prompts_path,
        "sft_model": sft_model,
        "reward_model": reward_model,
        "base_model": base_model,
        "output_dir": output_dir,
        "total_episodes": total_episodes,
        "learning_rate": learning_rate,
        "mini_batch_size": mini_batch_size,
        "batch_size": batch_size,
        "target_kl": target_kl,
        "dry_run": dry_run,
        "uses_trl_ppo_trainer": True,
    }


def check_ppo_trl_compatibility(trl) -> None:
    version = getattr(trl, "__version__", None)
    trainer_signature = inspect.signature(trl.PPOTrainer.__init__)
    required = {"config", "model", "tokenizer"}
    supported = set(trainer_signature.parameters)
    if version != PPO_TRL_VERSION or not required.issubset(supported):
        raise ValueError(PPO_TRL_COMPAT_ERROR)


def build_compatible_ppo_config(
    trl,
    *,
    learning_rate: float,
    batch_size: int,
    mini_batch_size: int,
    target_kl: float,
):
    supported = inspect.signature(trl.PPOConfig).parameters
    kwargs: dict[str, Any] = {}

    if "learning_rate" in supported:
        kwargs["learning_rate"] = learning_rate
    if "batch_size" in supported:
        kwargs["batch_size"] = batch_size
    if "mini_batch_size" in supported:
        kwargs["mini_batch_size"] = mini_batch_size
    if "target_kl" in supported:
        kwargs["target_kl"] = target_kl
    if "target" in supported:
        kwargs["target"] = target_kl
    if "log_with" in supported:
        kwargs["log_with"] = None

    return trl.PPOConfig(**kwargs)


def _load_prompt_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            prompt = str(row.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(f"line {line_number}: prompt must be non-empty.")
            rows.append({"prompt": prompt, "metadata": dict(row.get("metadata") or {})})
    if not rows:
        raise ValueError("Prompt dataset is empty.")
    return rows


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_ppo_training(
    *,
    prompts_path: str,
    sft_model: str,
    reward_model: str,
    base_model: str,
    output_dir: str,
    total_episodes: int = 1000,
    learning_rate: float = 3e-6,
    mini_batch_size: int = 1,
    batch_size: int = 4,
    target_kl: float = 0.1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or dry-run PPO-based RLHF training."""
    prompt_rows = _load_prompt_rows(prompts_path)
    config = build_ppo_config_dict(
        prompts_path=prompts_path,
        sft_model=sft_model,
        reward_model=reward_model,
        base_model=base_model,
        output_dir=output_dir,
        total_episodes=total_episodes,
        learning_rate=learning_rate,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size,
        target_kl=target_kl,
        dry_run=dry_run,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "ppo_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if dry_run:
        return {"dry_run": True, "prompt_rows": len(prompt_rows), "config": config}

    modules = load_ppo_training_dependencies()
    torch = modules["torch"]
    trl = modules["trl"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    check_ppo_trl_compatibility(trl)

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU required for PPO RLHF training. Use --dry-run on CPU-only machines.")

    reward_scorer = RewardScorer(reward_model, device="cuda")

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

    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    policy_base = transformers.AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    policy_base = peft.PeftModel.from_pretrained(policy_base, sft_model)

    policy_model = trl.AutoModelForCausalLMWithValueHead.from_pretrained(policy_base)
    ref_model = trl.create_reference_model(policy_model)

    ppo_config = build_compatible_ppo_config(
        trl,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        target_kl=target_kl,
    )

    ppo_trainer = trl.PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    metrics_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    generation_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "top_k": 0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
    }

    processed = 0
    while processed < total_episodes:
        batch = prompt_rows[processed : processed + batch_size]
        if not batch:
            break

        queries = [row["prompt"] for row in batch]
        query_tensors = [tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0).to(policy_model.pretrained_model.device) for prompt in queries]
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        responses = [tokenizer.decode(tensor.squeeze(), skip_special_tokens=True) for tensor in response_tensors]

        rewards = []
        for row, response in zip(batch, responses):
            reward_score = reward_scorer.score(row["prompt"], response)
            rule_reward = score_rule_based_reward(row["prompt"], response, row.get("metadata") or {})
            total_reward = reward_score + rule_reward["total"]
            rewards.append(torch.tensor(total_reward, device=policy_model.pretrained_model.device))
            sample_rows.append(
                {
                    "prompt": row["prompt"],
                    "response": response,
                    "reward_model_score": reward_score,
                    "rule_reward": rule_reward,
                    "total_reward": total_reward,
                }
            )

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        metrics_row = {
            "step": processed // batch_size,
            "reward_score": float(sum(r.item() for r in rewards) / len(rewards)),
            "kl": float(stats.get("objective/kl", stats.get("kl", 0.0))),
            "entropy": float(stats.get("objective/entropy", stats.get("entropy", 0.0))),
            "policy_loss": float(stats.get("ppo/loss/policy", stats.get("policy/loss", 0.0))),
            "value_loss": float(stats.get("ppo/loss/value", stats.get("loss/value", 0.0))),
            "approx_kl": float(stats.get("ppo/policy/approxkl_avg", stats.get("approx_kl", 0.0))),
            "clip_fraction": float(stats.get("ppo/policy/clipfrac_avg", stats.get("clip_fraction", 0.0))),
        }
        metrics_rows.append(metrics_row)
        processed += len(batch)

    adapter_dir = output_path / "adapter"
    policy_model.pretrained_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _save_jsonl(output_path / "ppo_metrics.jsonl", metrics_rows)
    _save_jsonl(output_path / "sample_generations.jsonl", sample_rows)

    return {"episodes": processed, "metrics_rows": len(metrics_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO-based RLHF training with TRL PPOTrainer.")
    parser.add_argument("--prompts", required=True, help="Prompt dataset JSONL path.")
    parser.add_argument("--sft-model", required=True, help="Initial SFT adapter path.")
    parser.add_argument("--reward-model", required=True, help="Reward model directory.")
    parser.add_argument("--base-model", required=True, help="Base model identifier.")
    parser.add_argument("--out", required=True, help="Output directory for PPO artifacts.")
    parser.add_argument("--total-episodes", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_ppo_training(
        prompts_path=args.prompts,
        sft_model=args.sft_model,
        reward_model=args.reward_model,
        base_model=args.base_model,
        output_dir=args.out,
        total_episodes=args.total_episodes,
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        target_kl=args.target_kl,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
