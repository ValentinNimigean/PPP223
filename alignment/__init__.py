"""PPO-based RLHF alignment package."""

from alignment.feedback_schema import (
    PENDING_CORRECTION_TOKEN,
    convert_old_feedback_to_preferences,
    load_preferences,
    validate_preference_row,
)
from alignment.reward_dataset import build_prompt_dataset, build_reward_examples
from alignment.rewards import score_rule_based_reward

__all__ = [
    "PENDING_CORRECTION_TOKEN",
    "build_prompt_dataset",
    "build_reward_examples",
    "convert_old_feedback_to_preferences",
    "load_preferences",
    "score_rule_based_reward",
    "validate_preference_row",
]
