import re
from typing import Dict, List


class ToxicityDetector:
    """
    Lightweight deterministic toxicity detector for demo/evaluation purposes.

    This is not a replacement for a production moderation model. It is intended
    to show that the agent has a safety layer for clearly abusive, hateful, or
    threatening language.
    """

    BLOCKED_TERMS = {
        "idiot",
        "stupid",
        "moron",
        "kill yourself",
        "go die",
        "i hate you",
        "worthless",
        "shut up",
    }

    HIGH_RISK_PHRASES = {
        "kill yourself",
        "go die",
    }

    def scan(self, text: str) -> Dict:
        text = text or ""
        lowered = text.lower()

        matched_terms: List[str] = []

        for term in self.BLOCKED_TERMS:
            escaped = re.escape(term.lower())

            if " " in term:
                pattern = escaped
            else:
                pattern = rf"\b{escaped}\b"

            if re.search(pattern, lowered):
                matched_terms.append(term)

        matched_terms = sorted(set(matched_terms))

        if not matched_terms:
            return {
                "is_toxic": False,
                "matched_terms": [],
                "toxicity_score": 0.0,
                "risk": "low",
            }

        high_risk_hits = [term for term in matched_terms if term in self.HIGH_RISK_PHRASES]

        if high_risk_hits:
            risk = "high"
            score = 1.0
        elif len(matched_terms) >= 2:
            risk = "medium"
            score = 0.6
        else:
            risk = "medium"
            score = 0.4

        return {
            "is_toxic": True,
            "matched_terms": matched_terms,
            "toxicity_score": score,
            "risk": risk,
        }

    def format_report(self, scan_result: Dict) -> str:
        return (
            f"Toxicity Risk: {scan_result['risk'].upper()} "
            f"(score: {scan_result['toxicity_score']:.2f})\n"
            f"Matched terms: "
            f"{', '.join(scan_result['matched_terms']) if scan_result['matched_terms'] else 'None'}"
        )


if __name__ == "__main__":
    detector = ToxicityDetector()

    clean = detector.scan("Explain how the retriever works.")
    assert clean["is_toxic"] is False
    assert clean["risk"] == "low"

    toxic = detector.scan("This code is stupid.")
    assert toxic["is_toxic"] is True
    assert "stupid" in toxic["matched_terms"]

    print(detector.format_report(clean))
    print(detector.format_report(toxic))
    print("ToxicityDetector OK")
