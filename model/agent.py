import json
from openai import OpenAI
from model.tools import AgentTools

class SLMAgent:
    def __init__(self, repo_map_string: str = "", base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5-coder:7b"):
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

    def set_retriever(self, retriever):
        self.retriever = retriever

    def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception:
            args = {}
            
        print(f"\n[Agent Action] Executing Tool: {name} | Args: {args}")
        
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

    def ask(self, user_prompt: str) -> str:
        print(f"\n[Agent] Thinking about: {user_prompt}")
        messages = [
            {
                "role": "system",
                "content": f"You are a Python Expert Assistant. You have tools to explore code.\nHere is the exact repository layout:\n{self.repo_map}\n\nUse semantic_search for concepts, and grep_search for exact names. Always explain what you find."
            },
            {"role": "user", "content": user_prompt}
        ]
        
        tools = AgentTools.get_tool_schemas()
        
        try:
            # First Pass: Ask the SLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # If the model decided to call tools
            if message.tool_calls:
                messages.append(message.to_dict()) # Append the assistant's tool call request
                
                for tool_call in message.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    # Tell the model what the tool found
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result
                    })
                
                print(f"[Agent] Tool results gathered. Generating final response...")
                # Second Pass: Send the results back for the final answer
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return final_response.choices[0].message.content
                
            # If no tool was needed
            return message.content
            
        except Exception as e:
            return f"Agent Execution Error: {e}\n(Make sure Ollama is running at {self.client.base_url} with model {self.model})"
