import json
import logging
import os
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AgentTools:
    """
    A collection of tools that the SLM can call.
    """
    SEARCHABLE_EXTENSIONS = {".py", ".md", ".toml", ".cfg", ".yaml", ".yml", ".json", ".txt"}
    
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
        # Validate regex pattern
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        try:
            results = []
            for root, _, files in os.walk(directory):
                if '.git' in root or '__pycache__' in root or 'venv' in root or 'node_modules' in root:
                    continue
                for file in files:
                    if "repomix-output" in file:
                        continue
                    ext = os.path.splitext(file)[1]
                    if ext in AgentTools.SEARCHABLE_EXTENSIONS:
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                for i, line in enumerate(lines):
                                    if re.search(pattern, line):
                                        results.append(f"{filepath}:{i+1}:{line.strip()}")
                        except Exception as e:
                            logger.debug(f"Could not read {filepath}: {e}")
            
            if not results:
                return "No matches found."
            
            if len(results) > 50:
                truncation_note = f"\n[Truncated: showing 50 of {len(results)} matches]"
            else:
                truncation_note = ""
                
            return "\n".join(results[:50]) + truncation_note
            
        except Exception as e:
            return f"Error executing search: {str(e)}"
