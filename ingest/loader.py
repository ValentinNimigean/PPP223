import os
from typing import List
from ingest.chunker import ASTChunker
from ingest.metadata import CodeChunk

class Loader:
    SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules", "site-packages", "dist-packages", ".eggs", "build", "dist"})

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.chunker = ASTChunker()

    def process_directory(self) -> List[CodeChunk]:
        """
        Crawls the root directory, reads all .py files, and chunks them.
        """
        all_chunks = []
        files_scanned = 0
        files_errored = 0
        
        for dirpath, _, filenames in os.walk(self.root_dir):
            # Skip common junk/dependency directories
            if any(skip in dirpath.split(os.sep) for skip in self.SKIP_DIRS):
                continue
                
            for file in filenames:
                if file.endswith(".py"):
                    file_path = os.path.join(dirpath, file)
                    files_scanned += 1
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            source = f.read()
                        
                        relative_path = os.path.relpath(file_path, self.root_dir)
                        chunks = self.chunker.chunk_file(relative_path, source)
                        all_chunks.extend(chunks)
                    except SyntaxError as e:
                        print(f"SyntaxError in {file_path}, falling back to tree-sitter: {e}")
                        chunks = self._fallback_tree_sitter_chunk(relative_path, source)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        files_errored += 1
        
        print(f"[Loader] Scanned {files_scanned} .py files -> {len(all_chunks)} chunks extracted ({files_errored} files skipped due to errors)")
        return all_chunks

    def _fallback_tree_sitter_chunk(self, filepath: str, source: str) -> List[CodeChunk]:
        chunks = []
        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser
            
            PY_LANGUAGE = Language(tspython.language())
            parser = Parser()
            parser.language = PY_LANGUAGE
            tree = parser.parse(bytes(source, "utf8"))
            
            def traverse(node):
                if node.type in ('function_definition', 'class_definition'):
                    name_node = next((child for child in node.children if child.type == 'identifier'), None)
                    name = name_node.text.decode('utf8') if name_node else "unknown"
                    chunk_type = "function" if node.type == 'function_definition' else "class"
                    chunks.append(CodeChunk(
                        filepath=filepath,
                        chunk_type=chunk_type,
                        name=name,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        text=node.text.decode('utf8')
                    ))
                for child in node.children:
                    traverse(child)
                    
            traverse(tree.root_node)
        except Exception as e:
            print(f"Tree-sitter fallback failed for {filepath}: {e}")
            
        return chunks
