import ast
from typing import List
from ingest.metadata import CodeChunk

class ASTChunker:
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
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                chunk_text = ast.get_source_segment(source_code, node)
                if not chunk_text:
                    continue
                    
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                signature = ast.unparse(node.args) if hasattr(node, "args") else ""
                
                chunks.append(CodeChunk(
                    filepath=filepath,
                    chunk_type="function",
                    name=node.name,
                    parent_class=None,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    text=chunk_text,
                    decorators=decorators,
                    signature=signature
                ))
            
            # Check for Classes
            elif isinstance(node, ast.ClassDef):
                chunk_text = ast.get_source_segment(source_code, node)
                if not chunk_text:
                    continue
                    
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                bases = []
                for base in getattr(node, "bases", []):
                    try:
                        bases.append(ast.unparse(base))
                    except:
                        pass
                        
                chunks.append(CodeChunk(
                    filepath=filepath,
                    chunk_type="class",
                    name=node.name,
                    parent_class=None,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    text=chunk_text,
                    decorators=decorators,
                    bases=bases
                ))
                
                # Check for Methods inside the class
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef) or isinstance(class_node, ast.AsyncFunctionDef):
                        method_text = ast.get_source_segment(source_code, class_node)
                        if not method_text:
                            continue
                            
                        method_decorators = [ast.unparse(d) for d in getattr(class_node, "decorator_list", [])]
                        method_signature = ast.unparse(class_node.args) if hasattr(class_node, "args") else ""
                        
                        chunks.append(CodeChunk(
                            filepath=filepath,
                            chunk_type="method",
                            name=class_node.name,
                            parent_class=node.name,
                            start_line=class_node.lineno,
                            end_line=class_node.end_lineno,
                            text=method_text,
                            decorators=method_decorators,
                            signature=method_signature
                        ))

        return chunks
