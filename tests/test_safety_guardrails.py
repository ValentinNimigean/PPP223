from ingest.metadata import CodeChunk
from model.agent import SLMAgent
from safety.guardrails import evaluate_guardrails
from safety.hallucination import HallucinationDetector
from safety.toxicity import ToxicityDetector


def test_abusive_prompt_is_blocked():
    agent = SLMAgent(hallucination_check=True)
    response = agent.ask("You are stupid and worthless. Explain the retriever.")

    assert "Please rephrase your request in a neutral way." in response
    assert agent.last_guardrail_result is not None
    assert agent.last_guardrail_result.reason == "toxic_input"


def test_fake_file_and_entity_are_flagged():
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
    detector = HallucinationDetector(chunks)
    result = evaluate_guardrails(
        "HybridRetriever calls FakeRetrieverClass from fake_module.py.",
        toxicity_detector=ToxicityDetector(),
        hallucination_detector=detector,
        allowed_entities=["hybridretriever"],
        allowed_files=["rag/retriever.py"],
    )

    assert "FakeRetrieverClass" in result.unsupported_entities
    assert "fake_module.py" in result.unsupported_files
    assert result.hallucination_score > 0.0


def test_grounded_answer_passes_guardrails():
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
    detector = HallucinationDetector(chunks)
    result = evaluate_guardrails(
        "`HybridRetriever` is defined in `rag/retriever.py`.",
        toxicity_detector=ToxicityDetector(),
        hallucination_detector=detector,
        allowed_entities=["hybridretriever"],
        allowed_files=["rag/retriever.py"],
    )

    assert result.allowed is True
    assert result.hallucination_score == 0.0


def test_regeneration_fallback_path_does_not_crash():
    class FakeInferenceEngine:
        def __init__(self):
            self.calls = 0

        def generate(self, messages):
            self.calls += 1
            if self.calls == 1:
                return "FakeRetrieverClass is defined in fake_module.py."
            return "UnknownEntity is implemented in imaginary.py."

    engine = FakeInferenceEngine()
    agent = SLMAgent(
        repo_map_string="",
        inference_engine=engine,
        hallucination_check=True,
        enable_deterministic_shortcuts=False,
        enable_tool_result_templates=False,
    )
    agent.set_chunks(
        [
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
    )

    response = agent.ask("Explain the retriever briefly.", max_turns=1)

    assert "cannot verify" in response.lower()
    assert engine.calls == 2
    assert agent.last_guardrail_result is not None
    assert agent.last_guardrail_result.allowed is False
