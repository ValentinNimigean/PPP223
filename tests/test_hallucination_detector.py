import pytest
from ingest.metadata import CodeChunk
from eval.hallucination import HallucinationDetector

def test_hallucination_detector_ignores_plain_english():
    chunks = [
        CodeChunk(
            filepath="ui/app.py",
            chunk_type="function",
            name="load_backend",
            qualified_name="load_backend",
            start_line=1,
            end_line=5,
            text="def load_backend(): pass",
        )
    ]
    detector = HallucinationDetector(chunks)
    scan = detector.scan("The Streamlit UI entrypoint is ui/app.py; it uses load_backend, and the chat flow is driven by st.chat_input.")
    assert "The" not in scan["unverified_entities"]
    assert "chat" not in scan["unverified_entities"]
    assert "flow" not in scan["unverified_entities"]
    assert "ui/app.py" in scan["verified_files"]

def test_hallucination_detector_flags_fake_code_entity():
    chunks = [
        CodeChunk(
            filepath="rag/retriever.py",
            chunk_type="class",
            name="HybridRetriever",
            qualified_name="HybridRetriever",
            start_line=1,
            end_line=5,
            text="class HybridRetriever: pass",
        )
    ]
    detector = HallucinationDetector(chunks)
    scan = detector.scan("HybridRetriever calls FakeRetrieverClass from fake_module.py.")
    assert "HybridRetriever" in scan["verified_entities"]
    assert "FakeRetrieverClass" in scan["unverified_entities"]
    assert "fake_module.py" in scan["unverified_files"]
