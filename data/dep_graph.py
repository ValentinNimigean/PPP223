import ast
import grimp
from collections import defaultdict
import os
import tempfile
from typing import Optional

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
    def __init__(self, pkg_name: str, pkg_dir: str):
        self.pkg_name = pkg_name
        self.pkg_dir = os.path.abspath(pkg_dir)
        self.intra_file_deps = defaultdict(lambda: {"calls": [], "inherits": [], "defines": []})
        self.module_graph = None

    def build(self):
        """Builds both the inter-module graph via grimp and the intra-module relations via AST."""
        try:
            self.module_graph = grimp.build_graph(self.pkg_name, include_external_packages=False)
        except Exception as e:
            print(f"Failed to build grimp graph for {self.pkg_name}: {e}")
            
        try:
            if not os.path.exists(self.pkg_dir):
                print(f"Warning: pkg_dir {self.pkg_dir} does not exist.")
                return

            for root, _, files in os.walk(self.pkg_dir):
                for file in files:
                    if file.endswith(".py"):
                        filepath = os.path.join(root, file)
                        self._parse_intra_file(filepath)
        except Exception as e:
            print(f"Warning: Failed to walk directory {self.pkg_dir}: {e}")
                    
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

    def get_chunk_deps(self, filepath: str, chunk_name: str, parent_class: Optional[str] = None):
        """Returns dependency info for a specific chunk name within a file."""
        qualified_name = f"{parent_class}.{chunk_name}" if parent_class else chunk_name
        deps = {"calls_to": [], "called_by": [], "inherits_from": [], "imported_by": []}
        
        # Intra-file edges
        file_data = self.intra_file_deps.get(filepath, {})
        for caller, callee in file_data.get("calls", []):
            if caller == qualified_name:
                # Strip the qualifying prefix (e.g. "self.baz" -> "baz") before returning
                deps["calls_to"].append(callee.split(".")[-1])
            if callee == qualified_name:
                deps["called_by"].append(caller)
                
        for child, base in file_data.get("inherits", []):
            if child == qualified_name:
                deps["inherits_from"].append(base)
                
        # Inter-file edges via grimp
        if self.module_graph:
            # We must map filepath to a module name for grimp
            try:
                # Calculate relative path from the parent of the package directory
                # so that the top-level package name is included in the module path.
                rel_path = os.path.relpath(filepath, start=os.path.dirname(self.pkg_dir))
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

def test_dep_graph():
    source = """
class Foo:
    def bar(self):
        self.baz()
    
    def baz(self):
        pass
"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "temp_test.py")
        with open(filepath, "w") as f:
            f.write(source)
        graph = DependencyGraph(pkg_name="non_existent_pkg_123", pkg_dir=tmpdir)
        graph.build()
        graph._parse_intra_file(filepath)
        deps = graph.get_chunk_deps(filepath, "bar", parent_class="Foo")
        calls = deps["calls_to"]
        assert "baz" in calls or "self.baz" in calls, f"Expected baz in {calls}"
        print("DepGraph OK")

if __name__ == "__main__":
    test_dep_graph()
