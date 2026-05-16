from openai import OpenAI
from typing import List, Dict, Any
import time

class LocalInferenceEngine:
    """
    Wraps a locally-running Ollama model for direct (non-agent) inference.
    Used for batch evaluation and output sampling during DPO data collection.
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str = "qwen2.5-coder:3b"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama-local" # any string works for ollama
        )
        self.model = model

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2,
                 max_tokens: int = 512) -> str:
        """Single completion. Returns the assistant content string."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Inference Error: {e}")
            return None

    def batch_generate(self, prompts: List[str], system: str = "",
                       temperature: float = 0.2, max_tokens: int = 512,
                       show_progress: bool = True) -> List[str]:
        """
        Runs generate() for each prompt sequentially.
        If show_progress=True, prints 'Processing N/total' every 10 items.
        Returns list of response strings, with None for any failed call.
        """
        results = []
        total = len(prompts)
        for i, prompt in enumerate(prompts):
            res = self.generate(prompt, system, temperature, max_tokens)
            results.append(res)
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"Processing {i + 1}/{total}")
        
        return results

    def evaluate_on_humaneval_subset(self, problems: List[Dict]) -> Dict:
        """
        Accepts a list of dicts with keys 'prompt' and 'canonical_solution'.
        For each, generates a completion and checks if the generated code
        contains the function name from the canonical solution (weak proxy metric).
        Returns {"total": N, "name_match": K, "name_match_rate": K/N}.
        """
        total = len(problems)
        matches = 0
        
        print(f"Evaluating on {total} HumanEval-style problems...")
        
        for problem in problems:
            prompt = problem.get("prompt", "")
            canonical = problem.get("canonical_solution", "")
            
            # Extract function name from canonical solution (simple heuristic)
            # Typically 'def function_name(...):'
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
            "name_match_rate": rate
        }

    def health_check(self) -> bool:
        """
        Pings the Ollama server. Returns True if reachable and model is loaded,
        False otherwise. Prints a human-readable status line.
        """
        try:
            # Pinging models list to check connectivity
            models_response = self.client.models.list()
            available_models = [m.id for m in models_response.data]
            
            if self.model in available_models:
                print(f"Health Check: OK | Ollama reachable, model '{self.model}' is loaded.")
                return True
            else:
                print(f"Health Check: WARNING | Ollama reachable, but model '{self.model}' not found in {available_models}.")
                return False
        except Exception as e:
            print(f"Health Check: FAILED | Could not connect to Ollama at {self.client.base_url}. Error: {e}")
            return False

if __name__ == "__main__":
    # Test block
    engine = LocalInferenceEngine()
    
    if engine.health_check():
        test_prompt = "Write a Python function that reverses a linked list."
        print(f"\nPrompt: {test_prompt}\n")
        response = engine.generate(test_prompt)
        print(f"Response:\n{response}")
    else:
        print("\nExiting due to failed health check. Ensure Ollama is running.")
