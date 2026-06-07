import ast
from typing import List
from ingest.metadata import CodeChunk

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

class LineChunker:
    def __init__(self, lines_per_chunk: int = 50, overlap_lines: int = 5):
        self.lines_per_chunk = lines_per_chunk
        self.overlap_lines = overlap_lines

    def chunk_file(self, filepath: str, source_code: str) -> List[CodeChunk]:
        """
        Chunks file based on lines.
        """
        lines = source_code.splitlines()
        chunks = []
        num_lines = len(lines)
        if num_lines == 0:
            return []

        step = self.lines_per_chunk - self.overlap_lines
        if step <= 0:
            step = self.lines_per_chunk

        chunk_idx = 0
        start = 0
        while start < num_lines:
            end = min(start + self.lines_per_chunk, num_lines)
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)
            
            chunks.append(CodeChunk(
                filepath=filepath,
                chunk_type="line",
                name=f"line_chunk_{chunk_idx}",
                qualified_name=f"line_chunk_{chunk_idx}",
                parent_class=None,
                start_line=start + 1,
                end_line=end,
                text=chunk_text,
                decorators=[],
                signature=""
            ))
            chunk_idx += 1
            if end == num_lines:
                break
            start += step
        return chunks

class CharacterChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_file(self, filepath: str, source_code: str) -> List[CodeChunk]:
        """
        Chunks file based on character count.
        """
        chunks = []
        n = len(source_code)
        if n == 0:
            return []

        # Map character offsets to line numbers
        line_starts = [0]
        for i, char in enumerate(source_code):
            if char == '\n':
                line_starts.append(i + 1)
        
        def get_line_num(char_offset: int) -> int:
            import bisect
            return bisect.bisect_right(line_starts, char_offset)

        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size

        chunk_idx = 0
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            chunk_text = source_code[start:end]
            start_line = get_line_num(start)
            end_line = get_line_num(end - 1) if end > start else start_line
            
            chunks.append(CodeChunk(
                filepath=filepath,
                chunk_type="character",
                name=f"char_chunk_{chunk_idx}",
                qualified_name=f"char_chunk_{chunk_idx}",
                parent_class=None,
                start_line=start_line,
                end_line=end_line,
                text=chunk_text,
                decorators=[],
                signature=""
            ))
            chunk_idx += 1
            if end == n:
                break
            start += step
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
    # Test ASTChunker
    chunker = ASTChunker()
    chunks = chunker.chunk_file("test.py", source)
    chunk_names = {c.qualified_name for c in chunks}
    print(f"AST Chunker: Extracted {len(chunks)} chunks: {chunk_names}")
    assert len(chunks) == 7

    # Test LineChunker
    line_chunker = LineChunker(lines_per_chunk=5, overlap_lines=1)
    line_chunks = line_chunker.chunk_file("test.py", source)
    print(f"Line Chunker: Extracted {len(line_chunks)} chunks")
    assert len(line_chunks) > 0

    # Test CharacterChunker
    char_chunker = CharacterChunker(chunk_size=100, chunk_overlap=20)
    char_chunks = char_chunker.chunk_file("test.py", source)
    print(f"Char Chunker: Extracted {len(char_chunks)} chunks")
    assert len(char_chunks) > 0

    print("All Chunkers OK")

