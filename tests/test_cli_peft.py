from types import SimpleNamespace

from ingest.metadata import CodeChunk
from ui.cli import build_agent, build_parser


def test_cli_parser_accepts_hf_peft_backend_and_adapter_path():
    """The CLI parser should accept the hf-peft backend and adapter path flags."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--repo",
            ".",
            "--backend",
            "hf-peft",
            "--base-model",
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "--adapter-path",
            "results_dpo/adapter",
            "--device",
            "auto",
            "--max-new-tokens",
            "256",
            "--temperature",
            "0.2",
        ]
    )

    assert args.backend == "hf-peft"
    assert args.adapter_path == "results_dpo/adapter"
    assert args.base_model == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert args.max_new_tokens == 256
    assert args.temperature == 0.2


def test_cli_defaults_to_fine_tuned_local_slm_path():
    parser = build_parser()
    args = parser.parse_args(["--repo", "."])

    assert args.backend == "hf-peft"
    assert args.base_model == "Qwen/Qwen2.5-Coder-3B-Instruct"


def test_missing_default_adapter_produces_clear_error(tmp_path):
    args = SimpleNamespace(
        repo=str(tmp_path),
        backend="hf-peft",
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        adapter_path=None,
        device="auto",
        max_new_tokens=128,
        temperature=0.2,
        disable_deterministic_shortcuts=False,
    )

    try:
        build_agent(args, "")
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when no fine-tuned adapter exists.")

    assert "results_ppo/adapter" in message
    assert "results_sft/adapter" in message
    assert "python -m model.finetune" in message
    assert "bash scripts/train_rlhf_ppo.sh --train" in message


def test_no_commercial_api_model_required_for_normal_operation(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "results_sft" / "adapter"
    adapter_dir.mkdir(parents=True)

    class FakeInferenceEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, messages):
            return "Local fine-tuned SLM response."

    monkeypatch.setattr("model.inference.PeftAdapterInferenceEngine", FakeInferenceEngine)

    args = SimpleNamespace(
        repo=str(tmp_path),
        backend="hf-peft",
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        adapter_path=str(adapter_dir),
        device="auto",
        max_new_tokens=128,
        temperature=0.2,
        disable_deterministic_shortcuts=True,
    )

    agent = build_agent(args, "")

    assert agent.inference_engine is not None
    assert agent.client is None


def test_rag_still_works_with_fine_tuned_backend(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "results_ppo" / "adapter"
    adapter_dir.mkdir(parents=True)

    class FakeInferenceEngine:
        def __init__(self, **kwargs):
            self.calls = []

        def generate(self, messages):
            self.calls.append(messages)
            return "HybridRetriever is defined in rag/retriever.py."

    monkeypatch.setattr("model.inference.PeftAdapterInferenceEngine", FakeInferenceEngine)

    args = SimpleNamespace(
        repo=str(tmp_path),
        backend="hf-peft",
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        adapter_path=str(adapter_dir),
        device="auto",
        max_new_tokens=128,
        temperature=0.2,
        disable_deterministic_shortcuts=True,
    )

    agent = build_agent(args, "")
    chunks = [
        CodeChunk(
            filepath="rag/retriever.py",
            chunk_type="class",
            name="HybridRetriever",
            qualified_name="HybridRetriever",
            start_line=1,
            end_line=5,
            text="class HybridRetriever: pass",
            signature="class HybridRetriever",
        )
    ]
    agent.set_chunks(chunks)
    answer = agent.ask("What class handles hybrid vector search?", max_turns=1)

    assert "HybridRetriever" in answer
    assert agent.inference_engine is not None
