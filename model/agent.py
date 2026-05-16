import json
import logging
from openai import OpenAI
from model.tools import AgentTools
from eval.hallucination import HallucinationDetector
from ingest.metadata import CodeChunk
from typing import List

logger = logging.getLogger(__name__)

class SLMAgent:
    def __init__(self, repo_map_string: str = "", base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5-coder:3b", hallucination_check: bool = False):
        """
        Initializes the SLM Agent pointing to a local Ollama server by default.
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama-local" # any string works for ollama
        )
        self.model = model
        self.repo_map = repo_map_string
        self.retriever = None
        self.hallucination_check = hallucination_check
        self._chunks = []
        self._detector = None

    def set_retriever(self, retriever):
        self.retriever = retriever

    def set_chunks(self, chunks: List[CodeChunk]):
        self._chunks = chunks
        if self.hallucination_check:
            self._detector = HallucinationDetector(chunks)

    def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception:
            args = {}
            
        logger.debug(f"\n[Agent Action] Executing Tool: {name} | Args: {args}")
        
        if name == "grep_search":
            return AgentTools.grep_search(args.get("pattern", ""))
        elif name == "semantic_search":
            if not self.retriever:
                return "Error: Retriever not attached to agent."
            results = self.retriever.search(args.get("query", ""), limit=3)
            # Simplify output so SLM doesn't get overwhelmed with vectors
            formatted = []
            for res in results:
                formatted.append(f"File: {res['metadata']['filepath']} | Name: {res['metadata']['name']}\nCode Snippet:\n{res['document']}")
            return "\n\n".join(formatted)
            
        return "Tool not recognized."

    def ask(self, user_prompt: str, max_turns: int = 8) -> str:
        logger.debug(f"\n[Agent] Thinking about: {user_prompt}")
        messages = [
            {
                "role": "system",
                "content": f"You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n{self.repo_map}\n\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."
            },
            {"role": "user", "content": user_prompt}
        ]
        
        tools = AgentTools.get_tool_schemas()
        
        try:
            for _ in range(max_turns):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                
                msg = response.choices[0].message
                msg_dict = msg.model_dump()
                messages.append({k: v for k, v in msg_dict.items() if v is not None or k == "content"})
                
                # If the model didn't call any tools, we are done
                if not msg.tool_calls:
                    final_content = msg.content
                    if self._detector is not None:
                        scan = self._detector.scan(final_content)
                        logger.debug(self._detector.format_report(scan))
                        if scan["hallucination_risk"] == "high":
                            warning = f"\n\n⚠️ Hallucination Warning: {len(scan['unverified_entities'])} unverified entities detected: {scan['unverified_entities']}"
                            return final_content + warning
                    return final_content
                
                # If it did call tools, execute them
                for tool_call in msg.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(tool_result)
                    })
                    
                logger.debug(f"[Agent] Tool results gathered. Sending back for reasoning...")
            
            return f"Agent exceeded maximum turns ({max_turns}) without reaching a final answer."
            
        except Exception as e:
            return f"Agent Execution Error: {e}\n(Make sure Ollama is running: 'ollama run qwen2.5-coder:3b' at {self.client.base_url})"
