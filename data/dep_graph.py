import ast
import grimp
from collections import defaultdict
import os

class IntraFileVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
        self.inherits = []
        self.defines = []
        self._stack = []

    def visit_ClassDef(self, node):
        self._stack.append(node.name)
        self.defines.append((".".join(self._stack), "class", node.lineno))
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.inherits.append((".".join(self._stack), base.id))
            elif isinstance(base, ast.Attribute):
                self.inherits.append((".".join(self._stack), ast.unparse(base)))
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node):
        self._stack.append(node.name)
        self.defines.append((".".join(self._stack), "function", node.lineno))
        self.generic_visit(node)
        self._stack.pop()
        
    def visit_AsyncFunctionDef(self, node):
        self._stack.append(node.name)
        self.defines.append((".".join(self._stack), "async_function", node.lineno))
        self.generic_visit(node)
        self._stack.pop()

    def visit_Call(self, node):
        callee = ast.unparse(node.func) if hasattr(ast, "unparse") else None
        if self._stack and callee:
            self.calls.append((".".join(self._stack), callee))
        self.generic_visit(node)


class DependencyGraph:
    def __init__(self, target_pkg: str):
        self.target_pkg = target_pkg
        self.intra_file_deps = defaultdict(lambda: {"calls": [], "inherits": [], "defines": []})
        self.module_graph = None

    def build(self):
        """Builds both the inter-module graph via grimp and the intra-module relations via AST."""
        try:
            self.module_graph = grimp.build_graph(self.target_pkg, include_external_packages=False)
        except Exception as e:
            print(f"Failed to build grimp graph for {self.target_pkg}: {e}")
            
        pkg_dir = os.path.abspath(self.target_pkg)
        for root, _, files in os.walk(pkg_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    self._parse_intra_file(filepath)
                    
    def _parse_intra_file(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            visitor = IntraFileVisitor()
            visitor.visit(tree)
            
            self.intra_file_deps[filepath] = {
                "calls": visitor.calls,
                "inherits": visitor.inherits,
                "defines": visitor.defines
            }
        except SyntaxError:
            pass # Ignore malformed files for the intra-file graph

    def get_chunk_deps(self, filepath: str, chunk_name: str):
        """Returns dependency info for a specific chunk name within a file."""
        deps = {"calls_to": [], "called_by": [], "inherits_from": [], "imported_by": []}
        
        # Intra-file edges
        file_data = self.intra_file_deps.get(filepath, {})
        for caller, callee in file_data.get("calls", []):
            if caller == chunk_name:
                deps["calls_to"].append(callee)
            if callee == chunk_name:
                deps["called_by"].append(caller)
                
        for child, base in file_data.get("inherits", []):
            if child == chunk_name:
                deps["inherits_from"].append(base)
                
        # Inter-file edges via grimp
        if self.module_graph:
            # We must map filepath to a module name for grimp
            try:
                rel_path = os.path.relpath(filepath, start=os.path.dirname(os.path.abspath(self.target_pkg)))
                module_name = rel_path.replace(os.path.sep, ".").replace(".py", "")
                if module_name.endswith(".__init__"):
                    module_name = module_name[:-9]
                
                # Check what modules import this module
                imported_by = self.module_graph.find_modules_directly_imported_by(module_name)
                # Not a perfect 1:1 chunk mapping, but gives file-level context
                deps["imported_by"] = list(imported_by)
            except Exception:
                pass
                
        return deps
