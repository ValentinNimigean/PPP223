import ast
from typing import List

from ingest.base import DocumentRecord
from ingest.metadata import CodeChunk


class TextChunker:
    """Chunk arbitrary text records into overlapping normalized document chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split plain text into overlapping chunks."""
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks

    def chunk_records(self, records: List[DocumentRecord]) -> List[DocumentRecord]:
        """Expand document records into chunked document records."""
        chunked: List[DocumentRecord] = []

        for record in records:
            pieces = self.chunk_text(record.text)
            if not pieces:
                continue

            for index, piece in enumerate(pieces):
                metadata = dict(record.metadata)
                metadata["chunk_index"] = index
                metadata["chunk_count"] = len(pieces)
                chunked.append(
                    DocumentRecord(
                        id=record.id,
                        source=record.source,
                        text=piece,
                        metadata=metadata,
                        doc_type=record.doc_type,
                        created_at=record.created_at,
                        chunk_id=f"{record.id}#chunk-{index}",
                    )
                )

        return chunked


class ASTChunker:
    def __init__(self, max_chunk_lines: int = 300):
        self.max_chunk_lines = max_chunk_lines

    def _check_large_chunk(self, name: str, filepath: str, start: int, end: int):
        n = end - start + 1
        if n > self.max_chunk_lines:
            print(f"[Warning] Large chunk: {name} spans {n} lines in {filepath}")

    def _extract_nested_functions(self, parent_node, parent_qualified_name, filepath, source_code, chunks, depth=0):
        if depth >= 3:
            return
            
        for node in parent_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk_text = ast.get_source_segment(source_code, node)
                if not chunk_text:
                    continue
                
                self._check_large_chunk(node.name, filepath, node.lineno, node.end_lineno)
                qualified_name = f"{parent_qualified_name}.{node.name}"
                
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                signature = ast.unparse(node.args) if hasattr(node, "args") else ""
                
                chunks.append(CodeChunk(
                    filepath=filepath,
                    chunk_type="function",
                    name=node.name,
                    qualified_name=qualified_name,
                    parent_class=parent_qualified_name.split(".")[-1],
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    text=chunk_text,
                    decorators=decorators,
                    signature=signature
                ))
                self._extract_nested_functions(node, qualified_name, filepath, source_code, chunks, depth + 1)
            elif isinstance(node, ast.ClassDef):
                chunk_text = ast.get_source_segment(source_code, node)
                if not chunk_text:
                    continue
                
                self._check_large_chunk(node.name, filepath, node.lineno, node.end_lineno)
                qualified_name = f"{parent_qualified_name}.{node.name}"
                
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                bases = [ast.unparse(b) for b in getattr(node, "bases", [])]
                
                chunks.append(CodeChunk(
                    filepath=filepath,
                    chunk_type="class",
                    name=node.name,
                    qualified_name=qualified_name,
                    parent_class=parent_qualified_name.split(".")[-1],
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    text=chunk_text,
                    decorators=decorators,
                    bases=bases
                ))
                self._extract_from_class(node, qualified_name, filepath, source_code, chunks, depth + 1)

    def _extract_from_class(self, class_node, parent_qualified_name, filepath, source_code, chunks, depth=0):
        if depth >= 3:
            return
            
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk_text = ast.get_source_segment(source_code, node)
                if not chunk_text:
                    continue
                
                self._check_large_chunk(node.name, filepath, node.lineno, node.end_lineno)
                qualified_name = f"{parent_qualified_name}.{node.name}"
                
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                signature = ast.unparse(node.args) if hasattr(node, "args") else ""
                
                chunks.append(CodeChunk(
                    filepath=filepath,
                    chunk_type="method",
                    name=node.name,
                    qualified_name=qualified_name,
                    parent_class=class_node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    text=chunk_text,
                    decorators=decorators,
                    signature=signature
                ))
                # Recurse into method for nested functions
                self._extract_nested_functions(node, qualified_name, filepath, source_code, chunks, depth + 1)
            elif isinstance(node, ast.ClassDef):
                chunk_text = ast.get_source_segment(source_code, node)
                if not chunk_text:
                    continue
                
                self._check_large_chunk(node.name, filepath, node.lineno, node.end_lineno)
                qualified_name = f"{parent_qualified_name}.{node.name}"
                
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                bases = [ast.unparse(b) for b in getattr(node, "bases", [])]
                
                chunks.append(CodeChunk(
                    filepath=filepath,
                    chunk_type="class",
                    name=node.name,
                    qualified_name=qualified_name,
                    parent_class=class_node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    text=chunk_text,
                    decorators=decorators,
                    bases=bases
                ))
                self._extract_from_class(node, qualified_name, filepath, source_code, chunks, depth + 1)

    def chunk_file(self, filepath: str, source_code: str) -> List[CodeChunk]:
        """
        Parses the Python source code into an AST and extracts CodeChunks.
        """
        chunks = []
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"Syntax error in {filepath}: {e}")
            raise e # Reraise to let loader handle fallback

        for node in tree.body:
            # Check for Module level functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk_text = ast.get_source_segment(source_code, node)
                if not chunk_text:
                    continue
                    
                self._check_large_chunk(node.name, filepath, node.lineno, node.end_lineno)
                
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                signature = ast.unparse(node.args) if hasattr(node, "args") else ""
                
                chunks.append(CodeChunk(
                    filepath=filepath,
                    chunk_type="function",
                    name=node.name,
                    qualified_name=node.name,
                    parent_class=None,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    text=chunk_text,
                    decorators=decorators,
                    signature=signature
                ))
                # Recurse into nested functions/classes
                self._extract_nested_functions(node, node.name, filepath, source_code, chunks, depth=1)
            
            # Check for Classes
            elif isinstance(node, ast.ClassDef):
                chunk_text = ast.get_source_segment(source_code, node)
                if not chunk_text:
                    continue
                    
                self._check_large_chunk(node.name, filepath, node.lineno, node.end_lineno)
                
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                bases = [ast.unparse(b) for b in getattr(node, "bases", [])]
                        
                chunks.append(CodeChunk(
                    filepath=filepath,
                    chunk_type="class",
                    name=node.name,
                    qualified_name=node.name,
                    parent_class=None,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    text=chunk_text,
                    decorators=decorators,
                    bases=bases
                ))
                # Recurse into class body
                self._extract_from_class(node, node.name, filepath, source_code, chunks, depth=1)

        return chunks

if __name__ == "__main__":
    source = """
def top_level_func():
    def nested_func():
        pass
    return True

class OuterClass:
    class NestedClass:
        pass
    
    def method_one(self):
        pass
        
    def method_two(self):
        def inner_of_method():
            pass
"""
    chunker = ASTChunker()
    chunks = chunker.chunk_file("test.py", source)
    
    # Expected chunks:
    # 1. top_level_func
    # 2. top_level_func.nested_func
    # 3. OuterClass
    # 4. OuterClass.NestedClass
    # 5. OuterClass.method_one
    # 6. OuterClass.method_two
    # 7. OuterClass.method_two.inner_of_method
    
    chunk_names = {c.qualified_name for c in chunks}
    print(f"Extracted {len(chunks)} chunks: {chunk_names}")
    
    assert "top_level_func" in chunk_names
    assert "top_level_func.nested_func" in chunk_names
    assert "OuterClass" in chunk_names
    assert "OuterClass.NestedClass" in chunk_names
    assert "OuterClass.method_one" in chunk_names
    assert "OuterClass.method_two" in chunk_names
    assert "OuterClass.method_two.inner_of_method" in chunk_names
    
    assert len(chunks) == 7
    print("Chunker OK")
