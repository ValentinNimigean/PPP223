import re
import os
import sys
from typing import List, Dict, Set, Optional

# Ensure root path is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.metadata import CodeChunk

class HallucinationDetector:
    IGNORE = {
        "def", "class", "return", "import", "from", "self", "true",
        "false", "none", "and", "or", "not", "in", "is", "if", "else",
        "for", "while", "with", "as", "try", "except", "pass", "print",
        "str", "int", "float", "list", "dict", "set", "bool", "len",
        "range", "type"
    }

    def __init__(self, chunks: List[CodeChunk]):
        """
        Builds a known-entity index from the ingested chunks.
        """
        self.known_names = set()
        self.known_qualified = set()
        self.known_files = set()
        
        for chunk in chunks:
            if chunk.name:
                self.known_names.add(chunk.name.lower())
            if chunk.qualified_name:
                self.known_qualified.add(chunk.qualified_name.lower())
            if chunk.filepath:
                self.known_files.add(chunk.filepath.lower())
                
        self.known_all = self.known_names.union(self.known_qualified).union(self.known_files)

    def scan(self, response: str) -> Dict:
        """
        Scans a response string for Python identifiers and file paths.
        """
        # 1. Extract Python identifiers
        # r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b'
        identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', response))
        
        verified_entities = []
        unverified_entities = []
        
        for ident in identifiers:
            ident_lower = ident.lower()
            if ident_lower in self.IGNORE:
                continue
            
            if ident_lower in self.known_all:
                verified_entities.append(ident)
            else:
                unverified_entities.append(ident)
                
        # 2. Extract file path candidates
        # r'[\w/\\]+\.py'
        file_candidates = set(re.findall(r'[\w/\\]+\.py', response))
        
        verified_files = []
        unverified_files = []
        
        for path in file_candidates:
            path_lower = path.lower()
            # Normalize path slashes for comparison if necessary, but here we just lowercase
            if path_lower in self.known_files:
                verified_files.append(path)
            else:
                unverified_files.append(path)
                
        # 3. Calculate score
        total_extracted = len(unverified_entities) + len(verified_entities) + len(unverified_files) + len(verified_files)
        unverified_count = len(unverified_entities) + len(unverified_files)
        
        score = unverified_count / (total_extracted + 1e-9)
        
        if score < 0.2:
            risk = "low"
        elif score < 0.5:
            risk = "medium"
        else:
            risk = "high"
            
        return {
            "verified_entities": sorted(verified_entities),
            "unverified_entities": sorted(unverified_entities),
            "verified_files": sorted(verified_files),
            "unverified_files": sorted(unverified_files),
            "hallucination_risk": risk,
            "hallucination_score": score
        }

    def format_report(self, scan_result: Dict) -> str:
        """
        Returns a human-readable string summarizing the scan result.
        """
        lines = [
            f"Hallucination Risk: {scan_result['hallucination_risk'].upper()} (score: {scan_result['hallucination_score']:.2f})",
            f"Verified entities: {', '.join(scan_result['verified_entities']) if scan_result['verified_entities'] else 'None'}",
            f"Unverified entities: {', '.join(scan_result['unverified_entities']) if scan_result['unverified_entities'] else 'None'}",
            f"Verified files: {', '.join(scan_result['verified_files']) if scan_result['verified_files'] else 'None'}",
            f"Unverified files: {', '.join(scan_result['unverified_files']) if scan_result['unverified_files'] else 'None'}"
        ]
        return "\n".join(lines)

if __name__ == "__main__":
    # Self-test
    test_chunks = [
        CodeChunk(
            filepath="rag/retriever.py",
            chunk_type="class",
            name="HybridRetriever",
            qualified_name="HybridRetriever",
            start_line=1,
            end_line=10,
            text="class HybridRetriever: pass"
        ),
        CodeChunk(
            filepath="rag/retriever.py",
            chunk_type="method",
            name="ingest_chunks",
            qualified_name="HybridRetriever.ingest_chunks",
            start_line=5,
            end_line=8,
            text="def ingest_chunks(self): pass"
        ),
        CodeChunk(
            filepath="ingest/loader.py",
            chunk_type="function",
            name="load_repo",
            qualified_name="load_repo",
            start_line=1,
            end_line=5,
            text="def load_repo(): pass"
        )
    ]
    
    detector = HallucinationDetector(test_chunks)
    
    test_response = "The HybridRetriever class uses ingest_chunks to process data from fake_module.py and calls UnknownClass."
    scan = detector.scan(test_response)
    
    assert "HybridRetriever" in scan["verified_entities"]
    assert "ingest_chunks" in scan["verified_entities"]
    assert "UnknownClass" in scan["unverified_entities"]
    assert "fake_module.py" in scan["unverified_files"]
    assert "rag/retriever.py" not in scan["unverified_files"]
    
    print(detector.format_report(scan))
    print("HallucinationDetector OK")
