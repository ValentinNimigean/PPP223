import json

from eval.metrics import exact_match, file_score, optional_bertscore, optional_rouge, penalized_combined_score, token_f1
from eval.rag_eval import retrieval_precision_at_k, source_hit_rate
from eval.report import build_markdown_summary, write_json_report, write_markdown_summary


def test_penalized_combined_score_and_file_score():
    response = "HybridRetriever is implemented in rag/retriever.py."
    penalized, scores = penalized_combined_score(
        response,
        ["HybridRetriever"],
        ["rag/retriever.py"],
    )

    assert file_score(response, ["rag/retriever.py"]) == 1.0
    assert scores["entity_score"] == 1.0
    assert scores["file_score"] == 1.0
    assert penalized == 1.0


def test_penalized_combined_score_with_hallucinated_file():
    response = "HybridRetriever is in fake_module.py and rag/retriever.py."
    penalized, scores = penalized_combined_score(
        response,
        ["HybridRetriever"],
        ["rag/retriever.py"],
    )

    assert scores["hallucination_penalty"] > 0.0
    assert "fake_module.py" in scores["unexpected_files_mentioned"]
    assert penalized < scores["combined_score"]


def test_exact_match_and_token_f1():
    assert exact_match("answer", "answer") == 1.0
    assert exact_match("answer", "different") == 0.0
    assert token_f1("the quick brown fox", "quick brown") > 0.0
    assert token_f1("", "quick brown") == 0.0


def test_optional_metrics_gracefully_skip():
    rouge = optional_rouge("prediction", "reference")
    bert = optional_bertscore("prediction", "reference")
    assert rouge is None or "rouge1_fmeasure" in rouge
    assert bert is None or "bertscore_f1" in bert


def test_rag_precision_and_source_hit_rate():
    results = [
        {"metadata": {"filepath": "rag/retriever.py"}},
        {"metadata": {"filepath": "model/agent.py"}},
    ]
    assert retrieval_precision_at_k(results, ["rag/retriever.py"], k=2) == 0.5
    assert source_hit_rate(results, ["rag/retriever.py"]) == 1.0


def test_report_generation(tmp_path):
    report = {
        "model": "demo-model",
        "backend": "ollama",
        "mode": "self",
        "overall": {
            "questions_evaluated": 1,
            "combined_score": 1.0,
            "penalized_combined_score": 1.0,
            "average_reward_model_score": None,
        },
        "results": [
            {
                "question": "What class handles hybrid vector search?",
                "combined_score": 1.0,
                "penalized_combined_score": 1.0,
                "hallucination_penalty": 0.0,
                "toxicity_flags": {"risk": "low"},
                "sources": ["rag/retriever.py"],
            }
        ],
    }

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    write_json_report(report, str(json_path))
    write_markdown_summary(report, str(md_path))

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert parsed["overall"]["combined_score"] == 1.0
    assert "# Evaluation Summary" in markdown
    assert "rag/retriever.py" in markdown
    assert "What class handles hybrid vector search?" in build_markdown_summary(report)
