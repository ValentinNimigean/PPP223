import json
from pathlib import Path

from ingest.metadata import CodeChunk
from model.agent import SLMAgent
from rag.retriever import HybridRetriever


def test_retriever_local_fallback_finds_hybrid_retriever():
    chunks = [
        CodeChunk(
            filepath="rag/retriever.py",
            chunk_type="class",
            name="HybridRetriever",
            qualified_name="HybridRetriever",
            start_line=5,
            end_line=111,
            text="class HybridRetriever:\n    def search(self, query):\n        return FusionQuery(fusion=Fusion.RRF)",
            signature="class HybridRetriever",
        ),
        CodeChunk(
            filepath="ingest/metadata.py",
            chunk_type="class",
            name="CodeChunk",
            qualified_name="CodeChunk",
            start_line=4,
            end_line=15,
            text="class CodeChunk(BaseModel): pass",
            signature="class CodeChunk",
        ),
    ]

    retriever = HybridRetriever(use_qdrant=False)
    retriever.ingest_chunks(chunks)

    results = retriever.search("What class handles hybrid vector search?", limit=1)
    assert results
    assert results[0]["metadata"]["filepath"] == "rag/retriever.py"
    assert results[0]["metadata"]["name"] == "HybridRetriever"


def test_agent_parses_clean_text_tool_call():
    agent = SLMAgent()
    parsed = agent._parse_text_tool_call(
        '{"name": "semantic_search", "arguments": {"query": "hybrid vector search"}}'
    )
    assert parsed == ("semantic_search", {"query": "hybrid vector search"})


def test_agent_repairs_unquoted_tool_name():
    agent = SLMAgent()
    parsed = agent._parse_text_tool_call(
        '{"name": semantic_search, "arguments": {"query": "hybrid vector search"}}'
    )
    assert parsed == ("semantic_search", {"query": "hybrid vector search"})


def test_agent_parses_fenced_tool_call():
    agent = SLMAgent()
    parsed = agent._parse_text_tool_call(
        """```json
{"name": "grep_search", "arguments": {"pattern": "CodeChunk", "directory": "."}}
```"""
    )
    assert parsed == ("grep_search", {"pattern": "CodeChunk", "directory": "."})


def test_agent_deterministic_answer_from_tool_result():
    agent = SLMAgent()
    result = (
        "File: rag/retriever.py | Name: HybridRetriever | Lines: 5-111\n"
        "Score: 12.0\n"
        "Code Snippet:\n"
        "class HybridRetriever: pass"
    )

    answer = agent._deterministic_answer_from_tool_result(
        "What class handles hybrid vector search?",
        "semantic_search",
        result,
    )

    assert "HybridRetriever" in answer
    assert "rag/retriever.py#L5-L111" in answer


def test_failure_dpo_from_eval(tmp_path):
    from data.failure_dpo_from_eval import rows_from_report

    report = {
        "results": [
            {
                "question": "What class handles hybrid vector search?",
                "response": '{"name": semantic_search, "arguments": {"query": "hybrid"}}',
                "penalized_combined_score": 0.0,
                "expected_entities": ["HybridRetriever"],
                "expected_files": ["rag/retriever.py"],
            }
        ]
    }

    rows, stats = rows_from_report(report, source_report="fake.json", threshold=0.75)
    assert len(rows) == 1
    assert rows[0]["prompt"] == "What class handles hybrid vector search?"
    assert "HybridRetriever" in rows[0]["chosen"]
    assert "rag/retriever.py" in rows[0]["chosen"]
    assert "semantic_search" in rows[0]["rejected"]


def test_agent_symbol_existence_no_match():
    agent = SLMAgent(
        repo_map_string="Repository Map:\n[-] rag/retriever.py\n  [C] Class: HybridRetriever (Lines 20-374)"
    )
    agent.set_chunks([])
    answer = agent._answer_symbol_existence("Does the repo define a FastAPIRetriever class?")
    assert answer == "I cannot find `FastAPIRetriever` in this repository."


def test_agent_symbol_existence_match_with_chunk():
    agent = SLMAgent(repo_map_string="")
    agent.set_chunks([
        CodeChunk(
            filepath="rag/retriever.py",
            chunk_type="class",
            name="HybridRetriever",
            qualified_name="HybridRetriever",
            start_line=20,
            end_line=374,
            text="class HybridRetriever: pass",
            signature="class HybridRetriever",
        )
    ])
    answer = agent._answer_symbol_existence("Does the repo define a HybridRetriever class?")
    assert "HybridRetriever" in answer
    assert "rag/retriever.py#L20-L374" in answer


def test_agent_symbol_existence_match_from_repo_map():
    agent = SLMAgent(
        repo_map_string="Repository Map:\n[-] ingest/metadata.py\n  [C] Class: CodeChunk (Lines 4-15)"
    )
    agent.set_chunks([])
    answer = agent._answer_symbol_existence("Is there a class called CodeChunk?")
    assert answer == "`CodeChunk` exists in this repository."


def test_agent_early_answer_repo_map():
    agent = SLMAgent(
        repo_map_string="[-] rag/repo_map.py\n  [C] Class: RepoMapGenerator (Lines 5-36)\n    [m] Method: generate_map (Lines 9-36)"
    )
    answer = agent._try_deterministic_early_answer(
        "Where is the repository map generated and what method builds it?"
    )
    assert "RepoMapGenerator.generate_map" in answer
    assert "rag/repo_map.py#L9-L36" in answer


def test_agent_early_answer_hallucination_penalties():
    agent = SLMAgent(
        repo_map_string="[-] eval/eval.py\n  [f] Function: hallucination_penalty (Lines 10-20)"
    )
    answer = agent._try_deterministic_early_answer(
        "Where does evaluation calculate hallucination penalties?"
    )
    assert "eval/eval.py" in answer
    assert "hallucination_penalty" in answer
    assert "unexpected_files_mentioned" in answer


def test_agent_early_answer_cli_ingestion():
    agent = SLMAgent(
        repo_map_string="[-] ui/cli.py\n  [f] Function: main (Lines 5-15)"
    )
    answer = agent._try_deterministic_early_answer(
        "Where does the CLI ingest repository chunks before answering questions?"
    )
    assert "ui/cli.py" in answer
    assert "Loader" in answer
    assert "HybridRetriever" in answer
    assert "ingest_chunks" in answer


def test_agent_early_answer_preference_generation():
    agent = SLMAgent(
        repo_map_string="[-] data/pref_gen.py\n  [f] Function: main (Lines 10-50)"
    )
    answer = agent._try_deterministic_early_answer(
        "Where does the preference generation script choose the chosen and rejected DPO answers?"
    )
    assert "data/pref_gen.py" in answer
    assert "main" in answer
    assert "chosen" in answer
    assert "rejected" in answer


def test_agent_early_answer_streamlit_entrypoint():
    agent = SLMAgent(
        repo_map_string="[-] ui/app.py\n  [f] Function: load_backend (Lines 8-25)"
    )
    answer = agent._try_deterministic_early_answer(
        "Where is the Streamlit UI entrypoint and what function loads the backend?"
    )
    assert "ui/app.py" in answer
    assert "load_backend" in answer
    assert "st.chat_input" in answer

