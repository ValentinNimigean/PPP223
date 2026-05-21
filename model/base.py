"""Shared model and inference backend interfaces."""

from __future__ import annotations

from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable


class ChatMessage(TypedDict):
    """Normalized chat message format used by agent backends."""

    role: str
    content: str


class GenerationConfig(TypedDict, total=False):
    """Optional generation controls for prompt-oriented backends."""

    system: NotRequired[str]
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]


@runtime_checkable
class PromptInferenceBackend(Protocol):
    """Interface for prompt-completion inference engines."""

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | None:
        """Generate a response from a plain prompt."""


@runtime_checkable
class ChatInferenceBackend(Protocol):
    """Interface for chat-style inference engines used by the agent."""

    def generate(self, messages: list[ChatMessage]) -> str:
        """Generate an assistant message from chat history."""


InferenceBackend = PromptInferenceBackend | ChatInferenceBackend | Any
