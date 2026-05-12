from pydantic import BaseModel, Field
from typing import Optional, List

class CodeChunk(BaseModel):
    filepath: str = Field(description="The path to the file containing this chunk.")
    chunk_type: str = Field(description="The type of the chunk: 'class', 'function', or 'method'.")
    name: str = Field(description="The name of the class or function.")
    parent_class: Optional[str] = Field(default=None, description="The name of the parent class, if this is a method.")
    start_line: int = Field(description="The starting line number of the chunk (1-indexed).")
    end_line: int = Field(description="The ending line number of the chunk (1-indexed).")
    text: str = Field(description="The exact text content of the chunk.")
    decorators: List[str] = Field(default=[], description="List of decorators applied to this chunk.")
    signature: str = Field(default="", description="The signature of the function or method.")
    bases: List[str] = Field(default=[], description="List of base classes for a class chunk.")
