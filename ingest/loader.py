import os
from typing import List
from ingest.chunker import ASTChunker
from ingest.metadata import CodeChunk

class Loader:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.chunker = ASTChunker()

    def process_directory(self) -> List[CodeChunk]:
        """
        Crawls the root directory, reads all .py files, and chunks them.
        """
        all_chunks = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                
                # We only want to process python files 
                # (and ideally ignore hidden dirs like .vscode or venv)
                if file.endswith(".py") and ".venv" not in dirpath and ".git" not in dirpath and "venv" not in dirpath:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            source = f.read()
                        
                        relative_path = os.path.relpath(file_path, self.root_dir)
                        chunks = self.chunker.chunk_file(relative_path, source)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        
        return all_chunks
