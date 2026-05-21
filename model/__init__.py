"""Model package exports."""

from model.agent import SLMAgent
from model.base import ChatInferenceBackend, ChatMessage, PromptInferenceBackend
from model.inference import LocalInferenceEngine, PeftAdapterInferenceEngine
from model.runtime import DEFAULT_BASE_MODEL, resolve_preferred_adapter

__all__ = [
    "ChatInferenceBackend",
    "ChatMessage",
    "DEFAULT_BASE_MODEL",
    "LocalInferenceEngine",
    "PeftAdapterInferenceEngine",
    "PromptInferenceBackend",
    "SLMAgent",
    "resolve_preferred_adapter",
]
