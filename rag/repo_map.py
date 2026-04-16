from typing import List
from collections import defaultdict
from ingest.metadata import CodeChunk

class RepoMapGenerator:
    def __init__(self):
        pass

    def generate_map(self, chunks: List[CodeChunk]) -> str:
        """
        Creates a bird's eye view string representation of the repository.
        """
        # Group by file path
        file_tree = defaultdict(list)
        for chunk in chunks:
            file_tree[chunk.filepath].append(chunk)

        repo_map_lines = ["Repository Map:", "================="]

        # Sort files alphabetically
        for filepath in sorted(file_tree.keys()):
            repo_map_lines.append(f"\n[-] {filepath}")
            file_chunks = file_tree[filepath]
            
            # Sort chunks by start line to keep logical order
            file_chunks.sort(key=lambda x: x.start_line)

            for chunk in file_chunks:
                if chunk.chunk_type == "class":
                    repo_map_lines.append(f"  [C] Class: {chunk.name} (Lines {chunk.start_line}-{chunk.end_line})")
                elif chunk.chunk_type == "method":
                    repo_map_lines.append(f"    [m] Method: {chunk.name} (Lines {chunk.start_line}-{chunk.end_line})")
                elif chunk.chunk_type == "function":
                    repo_map_lines.append(f"  [F] Function: {chunk.name} (Lines {chunk.start_line}-{chunk.end_line})")

        return "\n".join(repo_map_lines)
