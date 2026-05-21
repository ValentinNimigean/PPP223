from __future__ import annotations

from typing import Dict, List, Optional

from openai import OpenAI

from model.base import ChatMessage


class LocalInferenceEngine:
    """
    Wrap a locally-running Ollama model for direct inference.

    This engine is used for batch evaluation and output sampling during data
    collection, and preserves the existing OpenAI-compatible Ollama path.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5-coder:3b",
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama-local",
        )
        self.model = model

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """Generate a single completion from a plain prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"Inference Error: {exc}")
            return None

    def batch_generate(
        self,
        prompts: List[str],
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
        show_progress: bool = True,
    ) -> List[Optional[str]]:
        """Run generate() sequentially across a list of prompts."""
        results: List[Optional[str]] = []
        total = len(prompts)
        for i, prompt in enumerate(prompts):
            results.append(self.generate(prompt, system, temperature, max_tokens))
            if show_progress and (i + 1) % 10 == 0:
                print(f"Processing {i + 1}/{total}")
        return results

    def evaluate_on_humaneval_subset(self, problems: List[Dict]) -> Dict:
        """Evaluate name-match rate on a small HumanEval-style subset."""
        total = len(problems)
        matches = 0

        print(f"Evaluating on {total} HumanEval-style problems...")

        for problem in problems:
            prompt = problem.get("prompt", "")
            canonical = problem.get("canonical_solution", "")

            func_name = "unknown"
            if "def " in canonical:
                func_name = canonical.split("def ")[1].split("(")[0].strip()

            response = self.generate(prompt)
            if response and func_name in response:
                matches += 1

        rate = matches / total if total > 0 else 0
        return {
            "total": total,
            "name_match": matches,
            "name_match_rate": rate,
        }

    def health_check(self) -> bool:
        """Return whether Ollama is reachable and the configured model is present."""
        try:
            models_response = self.client.models.list()
            available_models = [model.id for model in models_response.data]

            if self.model in available_models:
                print(f"Health Check: OK | Ollama reachable, model '{self.model}' is loaded.")
                return True

            print(
                "Health Check: WARNING | Ollama reachable, but model "
                f"'{self.model}' not found in {available_models}."
            )
            return False
        except Exception as exc:
            print(
                "Health Check: FAILED | Could not connect to Ollama at "
                f"{self.client.base_url}. Error: {exc}"
            )
            return False


class PeftAdapterInferenceEngine:
    """
    Run local generation with a Hugging Face base model plus a PEFT adapter.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The hf-peft backend requires `transformers`, `peft`, `accelerate`, "
                "and possibly `bitsandbytes` to be installed."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        model_kwargs = {"torch_dtype": "auto"}
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _fallback_prompt(messages: List[ChatMessage]) -> str:
        """Build a conservative plain-text prompt if no chat template is available."""
        lines: List[str] = []
        for message in messages:
            role = (message.get("role") or "user").upper()
            content = message.get("content") or ""
            lines.append(f"{role}:\n{content}")
        lines.append("ASSISTANT:\n")
        return "\n\n".join(lines)

    def generate(self, messages: List[ChatMessage]) -> str:
        """Generate an assistant response from chat-style messages."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = self._fallback_prompt(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.device != "auto":
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        else:
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature and self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["temperature"] = 0.0
            generation_kwargs["do_sample"] = False

        with self._torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
