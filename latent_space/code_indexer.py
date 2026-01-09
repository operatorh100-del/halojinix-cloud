"""
HALO's Code Indexer - Tree-sitter based AST parser for TypeScript/Python

Extracts:
- Nodes: files, classes, functions, methods
- Edges: CALLS, IMPORTS, CONTAINS, EXTENDS, IMPLEMENTS

This is the foundation of HALO's Latent Space - treating code as topology,
not just text.
"""

import tree_sitter_typescript as ts_typescript
import tree_sitter_python as ts_python
from tree_sitter import Language, Parser
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from pathlib import Path
import re


# Initialize languages
TS_LANGUAGE = Language(ts_typescript.language_typescript())
TSX_LANGUAGE = Language(ts_typescript.language_tsx())
PY_LANGUAGE = Language(ts_python.language())


@dataclass
class CodeNode:
    """A semantic unit in the code graph."""
    id: str                              # e.g., "engine/gfx/gpu-context.ts::GpuContext"
    type: str                            # file, class, function, method, interface
    name: str
    path: str                            # File path (relative)
    language: str                        # typescript, python
    line_start: int
    line_end: int
    code: str                            # Full source text
    docstring: Optional[str] = None      # JSDoc/docstring if present
    signature: Optional[str] = None      # For functions: params and return type
    contained_by: Optional[str] = None   # Parent node ID
    contains: List[str] = field(default_factory=list)  # Child node IDs


@dataclass 
class CodeEdge:
    """A relationship between code nodes."""
    id: str
    type: str                            # CALLS, IMPORTS, CONTAINS, EXTENDS, IMPLEMENTS
    from_node: str                       # Source node ID
    to_node: str                         # Target node ID
    line: Optional[int] = None           # Line where relationship occurs
    evidence: Optional[str] = None       # Code snippet showing the relationship


class CodeIndexer:
    """
    Parses source files into code nodes and edges.
    
    Usage:
        indexer = CodeIndexer()
        nodes, edges = indexer.index_file("engine/gfx/gpu-context.ts")
    """
    
    def __init__(self):
        self.ts_parser = Parser(TS_LANGUAGE)
        self.tsx_parser = Parser(TSX_LANGUAGE)
        self.py_parser = Parser(PY_LANGUAGE)
        
        # Track discovered nodes and edges
        self.nodes: Dict[str, CodeNode] = {}
        self.edges: List[CodeEdge] = []
        
        # For resolving imports to actual files
        self.known_files: Set[str] = set()
        
    def get_parser(self, path: str) -> tuple[Parser, str]:
        """Get appropriate parser for file type."""
        if path.endswith('.tsx'):
            return self.tsx_parser, 'typescript'
        elif path.endswith('.ts'):
            return self.ts_parser, 'typescript'
        elif path.endswith('.py'):
            return self.py_parser, 'python'
        else:
            raise ValueError(f"Unsupported file type: {path}")
    
    def index_file(self, filepath: str, base_path: str = "") -> tuple[List[CodeNode], List[CodeEdge]]:
        """
        Index a single source file.
        
        Args:
            filepath: Path to the source file
            base_path: Base path to strip for relative paths
            
        Returns:
            Tuple of (nodes, edges) discovered in this file
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get relative path for IDs
        if base_path:
            rel_path = str(path.relative_to(base_path)).replace("\\", "/")
        else:
            rel_path = str(path).replace("\\", "/")
        
        # Parse the file
        source_code = path.read_bytes()
        parser, language = self.get_parser(filepath)
        tree = parser.parse(source_code)
        
        # Create file node
        file_node = CodeNode(
            id=rel_path,
            type="file",
            name=path.name,
            path=rel_path,
            language=language,
            line_start=1,
            line_end=source_code.count(b'\n') + 1,
            code=source_code.decode('utf-8', errors='replace')
        )
        self.nodes[file_node.id] = file_node
        
        # Walk the AST
        if language == 'typescript':
            self._index_typescript(tree.root_node, source_code, rel_path, file_node.id)
        elif language == 'python':
            self._index_python(tree.root_node, source_code, rel_path, file_node.id)
        
        # Return nodes and edges from this file
        file_nodes = [n for n in self.nodes.values() if n.path == rel_path or n.id == rel_path]
        file_edges = [e for e in self.edges if rel_path in e.from_node or rel_path in e.to_node]
        
        return file_nodes, file_edges
    
    def _index_typescript(self, root_node, source: bytes, file_path: str, file_id: str):
        """Extract nodes and edges from TypeScript AST."""
        
        def get_text(node) -> str:
            return source[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
        
        def get_docstring(node) -> Optional[str]:
            """Look for preceding JSDoc comment."""
            prev = node.prev_sibling
            if prev and prev.type == 'comment':
                text = get_text(prev)
                if text.startswith('/**'):
                    return text
            return None
        
        def walk(node, parent_id: str = file_id):
            """Recursively walk the AST."""
            
            # Classes
            if node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                name = get_text(name_node) if name_node else 'anonymous'
                node_id = f"{file_path}::{name}"
                
                class_node = CodeNode(
                    id=node_id,
                    type="class",
                    name=name,
                    path=file_path,
                    language="typescript",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    code=get_text(node),
                    docstring=get_docstring(node),
                    contained_by=parent_id
                )
                self.nodes[node_id] = class_node
                
                # CONTAINS edge
                self.edges.append(CodeEdge(
                    id=f"{parent_id}--CONTAINS--{node_id}",
                    type="CONTAINS",
                    from_node=parent_id,
                    to_node=node_id,
                    line=node.start_point[0] + 1
                ))
                
                # Check for extends
                heritage = node.child_by_field_name('heritage')
                if heritage:
                    for child in heritage.children:
                        if child.type == 'extends_clause':
                            extends_text = get_text(child)
                            match = re.search(r'extends\s+(\w+)', extends_text)
                            if match:
                                base_class = match.group(1)
                                self.edges.append(CodeEdge(
                                    id=f"{node_id}--EXTENDS--{base_class}",
                                    type="EXTENDS",
                                    from_node=node_id,
                                    to_node=base_class,  # Will need resolution
                                    line=child.start_point[0] + 1,
                                    evidence=extends_text
                                ))
                        elif child.type == 'implements_clause':
                            impl_text = get_text(child)
                            # Can implement multiple interfaces
                            for iface in re.findall(r'\b(\w+)\b', impl_text.replace('implements', '')):
                                self.edges.append(CodeEdge(
                                    id=f"{node_id}--IMPLEMENTS--{iface}",
                                    type="IMPLEMENTS",
                                    from_node=node_id,
                                    to_node=iface,
                                    line=child.start_point[0] + 1,
                                    evidence=impl_text
                                ))
                
                # Recurse into class body
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        walk(child, node_id)
                return
            
            # Interfaces
            if node.type == 'interface_declaration':
                name_node = node.child_by_field_name('name')
                name = get_text(name_node) if name_node else 'anonymous'
                node_id = f"{file_path}::{name}"
                
                iface_node = CodeNode(
                    id=node_id,
                    type="interface",
                    name=name,
                    path=file_path,
                    language="typescript",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    code=get_text(node),
                    docstring=get_docstring(node),
                    contained_by=parent_id
                )
                self.nodes[node_id] = iface_node
                
                self.edges.append(CodeEdge(
                    id=f"{parent_id}--CONTAINS--{node_id}",
                    type="CONTAINS",
                    from_node=parent_id,
                    to_node=node_id,
                    line=node.start_point[0] + 1
                ))
                return
            
            # Functions (top-level)
            if node.type in ('function_declaration', 'arrow_function', 'function_expression'):
                name_node = node.child_by_field_name('name')
                name = get_text(name_node) if name_node else 'anonymous'
                
                # Skip anonymous functions unless they're assigned
                if name == 'anonymous':
                    # Check if this is a variable assignment
                    if node.parent and node.parent.type == 'variable_declarator':
                        name_node = node.parent.child_by_field_name('name')
                        if name_node:
                            name = get_text(name_node)
                
                if name != 'anonymous':
                    node_id = f"{file_path}::{name}"
                    
                    # Get signature
                    params_node = node.child_by_field_name('parameters')
                    params_text = get_text(params_node) if params_node else "()"
                    return_node = node.child_by_field_name('return_type')
                    return_text = get_text(return_node) if return_node else ""
                    signature = f"{params_text}{return_text}"
                    
                    func_node = CodeNode(
                        id=node_id,
                        type="function",
                        name=name,
                        path=file_path,
                        language="typescript",
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        code=get_text(node),
                        signature=signature,
                        docstring=get_docstring(node),
                        contained_by=parent_id
                    )
                    self.nodes[node_id] = func_node
                    
                    self.edges.append(CodeEdge(
                        id=f"{parent_id}--CONTAINS--{node_id}",
                        type="CONTAINS",
                        from_node=parent_id,
                        to_node=node_id,
                        line=node.start_point[0] + 1
                    ))
                    
                    # Extract calls from function body
                    self._extract_calls_typescript(node, source, file_path, node_id)
                return
            
            # Methods (inside classes)
            if node.type == 'method_definition':
                name_node = node.child_by_field_name('name')
                name = get_text(name_node) if name_node else 'anonymous'
                node_id = f"{parent_id}::{name}"
                
                # Get signature
                params_node = node.child_by_field_name('parameters')
                params_text = get_text(params_node) if params_node else "()"
                return_node = node.child_by_field_name('return_type')
                return_text = get_text(return_node) if return_node else ""
                signature = f"{params_text}{return_text}"
                
                method_node = CodeNode(
                    id=node_id,
                    type="method",
                    name=name,
                    path=file_path,
                    language="typescript",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    code=get_text(node),
                    signature=signature,
                    docstring=get_docstring(node),
                    contained_by=parent_id
                )
                self.nodes[node_id] = method_node
                
                self.edges.append(CodeEdge(
                    id=f"{parent_id}--CONTAINS--{node_id}",
                    type="CONTAINS",
                    from_node=parent_id,
                    to_node=node_id,
                    line=node.start_point[0] + 1
                ))
                
                # Extract calls from method body
                self._extract_calls_typescript(node, source, file_path, node_id)
                return
            
            # Import statements
            if node.type == 'import_statement':
                import_text = get_text(node)
                
                # Extract what's being imported
                source_node = node.child_by_field_name('source')
                if source_node:
                    import_path = get_text(source_node).strip('"\'')
                    
                    self.edges.append(CodeEdge(
                        id=f"{file_path}--IMPORTS--{import_path}",
                        type="IMPORTS",
                        from_node=file_id,
                        to_node=import_path,  # Will need resolution
                        line=node.start_point[0] + 1,
                        evidence=import_text
                    ))
                return
            
            # Recurse into children
            for child in node.children:
                walk(child, parent_id)
        
        walk(root_node)
    
    def _extract_calls_typescript(self, node, source: bytes, file_path: str, caller_id: str):
        """Extract function/method calls from a code block."""
        
        def get_text(n) -> str:
            return source[n.start_byte:n.end_byte].decode('utf-8', errors='replace')
        
        def find_calls(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    call_text = get_text(func)
                    full_call = get_text(n)
                    
                    # Determine the callee
                    # Could be: foo(), this.foo(), obj.foo(), etc.
                    callee = call_text.split('.')[-1] if '.' in call_text else call_text
                    
                    self.edges.append(CodeEdge(
                        id=f"{caller_id}--CALLS--{callee}@{n.start_point[0]+1}",
                        type="CALLS",
                        from_node=caller_id,
                        to_node=callee,  # Will need resolution
                        line=n.start_point[0] + 1,
                        evidence=full_call[:100]  # Truncate long calls
                    ))
            
            for child in n.children:
                find_calls(child)
        
        find_calls(node)
    
    def _index_python(self, root_node, source: bytes, file_path: str, file_id: str):
        """Extract nodes and edges from Python AST."""
        
        def get_text(node) -> str:
            return source[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
        
        def get_docstring(node) -> Optional[str]:
            """Get docstring from function/class body."""
            body = node.child_by_field_name('body')
            if body and body.children:
                first = body.children[0]
                if first.type == 'expression_statement':
                    expr = first.children[0] if first.children else None
                    if expr and expr.type == 'string':
                        return get_text(expr)
            return None
        
        def walk(node, parent_id: str = file_id):
            # Class definitions
            if node.type == 'class_definition':
                name_node = node.child_by_field_name('name')
                name = get_text(name_node) if name_node else 'anonymous'
                node_id = f"{file_path}::{name}"
                
                class_node = CodeNode(
                    id=node_id,
                    type="class",
                    name=name,
                    path=file_path,
                    language="python",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    code=get_text(node),
                    docstring=get_docstring(node),
                    contained_by=parent_id
                )
                self.nodes[node_id] = class_node
                
                self.edges.append(CodeEdge(
                    id=f"{parent_id}--CONTAINS--{node_id}",
                    type="CONTAINS",
                    from_node=parent_id,
                    to_node=node_id,
                    line=node.start_point[0] + 1
                ))
                
                # Check for base classes
                superclass = node.child_by_field_name('superclasses')
                if superclass:
                    for arg in superclass.children:
                        if arg.type == 'argument_list':
                            for base in arg.children:
                                if base.type == 'identifier':
                                    base_name = get_text(base)
                                    self.edges.append(CodeEdge(
                                        id=f"{node_id}--EXTENDS--{base_name}",
                                        type="EXTENDS",
                                        from_node=node_id,
                                        to_node=base_name,
                                        line=base.start_point[0] + 1
                                    ))
                
                # Recurse into class body
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        walk(child, node_id)
                return
            
            # Function definitions
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                name = get_text(name_node) if name_node else 'anonymous'
                
                # Determine if this is a method or top-level function
                is_method = parent_id != file_id and '::' in parent_id
                node_type = "method" if is_method else "function"
                node_id = f"{parent_id}::{name}"
                
                # Get signature
                params_node = node.child_by_field_name('parameters')
                params_text = get_text(params_node) if params_node else "()"
                return_node = node.child_by_field_name('return_type')
                return_text = f" -> {get_text(return_node)}" if return_node else ""
                signature = f"{params_text}{return_text}"
                
                func_node = CodeNode(
                    id=node_id,
                    type=node_type,
                    name=name,
                    path=file_path,
                    language="python",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    code=get_text(node),
                    signature=signature,
                    docstring=get_docstring(node),
                    contained_by=parent_id
                )
                self.nodes[node_id] = func_node
                
                self.edges.append(CodeEdge(
                    id=f"{parent_id}--CONTAINS--{node_id}",
                    type="CONTAINS",
                    from_node=parent_id,
                    to_node=node_id,
                    line=node.start_point[0] + 1
                ))
                
                # Extract calls
                self._extract_calls_python(node, source, file_path, node_id)
                return
            
            # Import statements
            if node.type in ('import_statement', 'import_from_statement'):
                import_text = get_text(node)
                
                # Try to extract module name
                module_node = node.child_by_field_name('module_name')
                if module_node:
                    module_name = get_text(module_node)
                else:
                    # Fall back to parsing the text
                    match = re.search(r'(?:from|import)\s+([\w.]+)', import_text)
                    module_name = match.group(1) if match else import_text
                
                self.edges.append(CodeEdge(
                    id=f"{file_path}--IMPORTS--{module_name}",
                    type="IMPORTS",
                    from_node=file_id,
                    to_node=module_name,
                    line=node.start_point[0] + 1,
                    evidence=import_text
                ))
                return
            
            # Recurse
            for child in node.children:
                walk(child, parent_id)
        
        walk(root_node)
    
    def _extract_calls_python(self, node, source: bytes, file_path: str, caller_id: str):
        """Extract function/method calls from Python code."""
        
        def get_text(n) -> str:
            return source[n.start_byte:n.end_byte].decode('utf-8', errors='replace')
        
        def find_calls(n):
            if n.type == 'call':
                func = n.child_by_field_name('function')
                if func:
                    call_text = get_text(func)
                    full_call = get_text(n)
                    
                    # Get the actual function name
                    callee = call_text.split('.')[-1] if '.' in call_text else call_text
                    
                    self.edges.append(CodeEdge(
                        id=f"{caller_id}--CALLS--{callee}@{n.start_point[0]+1}",
                        type="CALLS",
                        from_node=caller_id,
                        to_node=callee,
                        line=n.start_point[0] + 1,
                        evidence=full_call[:100]
                    ))
            
            for child in n.children:
                find_calls(child)
        
        find_calls(node)
    
    def index_directory(self, dir_path: str, base_path: Optional[str] = None) -> tuple[List[CodeNode], List[CodeEdge]]:
        """
        Recursively index all supported files in a directory.
        
        Args:
            dir_path: Directory to index
            base_path: Base path for relative paths (defaults to dir_path)
            
        Returns:
            Tuple of (all_nodes, all_edges)
        """
        path = Path(dir_path)
        base = Path(base_path) if base_path else path
        
        all_nodes = []
        all_edges = []
        
        # Find all supported files
        patterns = ['**/*.ts', '**/*.tsx', '**/*.py']
        for pattern in patterns:
            for file_path in path.glob(pattern):
                # Skip node_modules, dist, etc.
                if any(skip in str(file_path) for skip in ['node_modules', 'dist', '__pycache__', '.git']):
                    continue
                
                try:
                    nodes, edges = self.index_file(str(file_path), str(base))
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                    print(f"? Indexed: {file_path.relative_to(base)}")
                except Exception as e:
                    print(f"? Failed: {file_path.relative_to(base)} - {e}")
        
        return all_nodes, all_edges
    
    def to_dict(self) -> dict:
        """Export indexed data as dictionary for JSON serialization."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "name": n.name,
                    "path": n.path,
                    "language": n.language,
                    "line_start": n.line_start,
                    "line_end": n.line_end,
                    "signature": n.signature,
                    "docstring": n.docstring,
                    "contained_by": n.contained_by,
                    "contains": n.contains,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "type": e.type,
                    "from": e.from_node,
                    "to": e.to_node,
                    "line": e.line,
                    "evidence": e.evidence,
                }
                for e in self.edges
            ]
        }


def main():
    """Test the indexer on engine/gfx/gpu-context.ts"""
    import json
    
    indexer = CodeIndexer()
    
    # Test single file
    print("=" * 60)
    print("Testing CodeIndexer on engine/gfx/gpu-context.ts")
    print("=" * 60)
    
    try:
        nodes, edges = indexer.index_file(
            "F:/primewave-engine/engine/gfx/gpu-context.ts",
            "F:/primewave-engine"
        )
        
        print(f"\nFound {len(nodes)} nodes:")
        for node in nodes:
            print(f"  [{node.type}] {node.id} (lines {node.line_start}-{node.line_end})")
        
        print(f"\nFound {len(edges)} edges:")
        for edge in edges:
            print(f"  {edge.from_node} --{edge.type}--> {edge.to_node}")
            if edge.evidence:
                print(f"    Evidence: {edge.evidence[:60]}...")
        
        # Export to JSON
        output_path = "F:/primewave-engine/data/code-graph-sample.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(indexer.to_dict(), f, indent=2)
        print(f"\nExported to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
