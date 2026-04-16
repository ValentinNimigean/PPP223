import subprocess
import json
from typing import List, Dict, Any

class AgentTools:
    """
    A collection of tools that the SLM can call.
    """
    
    @staticmethod
    def get_tool_schemas() -> List[Dict[str, Any]]:
        """
        Returns the OpenAI format JSON schemas for the agent tools.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "grep_search",
                    "description": "Searches the codebase for exact string matches or regex patterns. Use this to find specific variables, function definitions, or imports that are not caught by semantic search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The string or regex pattern to search for."
                            },
                            "directory": {
                                "type": "string",
                                "description": "The directory to search inside (default is '.')."
                            }
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": "Queries the Qdrant Hybrid Retriever to find the most conceptually relevant functions or classes based on natural language.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The natural language query describing what logic you are looking for."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    @staticmethod
    def grep_search(pattern: str, directory: str = ".") -> str:
        """
        Executes a local grep (or equivalent python implementation) to find literal matches.
        """
        try:
            # We use git grep if available, fallback to basic python parsing or findstr on windows
            # Since this is windows, findstr is safer native. But grep is cleaner if WSL/GitBash.
            # Using python's basic traversal as a cross-platform safe 'grep':
            import os
            import re
            
            results = []
            for root, _, files in os.walk(directory):
                if '.git' in root or '__pycache__' in root or 'venv' in root:
                    continue
                for file in files:
                    if file.endswith('.py') or file.endswith('.md'):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                for i, line in enumerate(lines):
                                    if re.search(pattern, line):
                                        results.append(f"{filepath}:{i+1}:{line.strip()}")
                        except Exception:
                            pass
            
            if not results:
                return "No matches found."
            return "\n".join(results[:50]) # Limit output size
            
        except Exception as e:
            return f"Error executing search: {str(e)}"
