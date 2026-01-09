"""
ScornSpine Semantic Chunker - RT16600
Language-aware document chunking for better embeddings.

Based on JONAH RT16700 research:
- AST-aware chunking for code (preserves function/class boundaries)
- Header-based chunking for Markdown
- Contextualized embedding strings
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Chunk:
    """A semantic chunk of a document."""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, header, paragraph, etc.
    name: Optional[str] = None  # Function/class name, header text
    parent: Optional[str] = None  # Parent class/header for context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_embedding_string(self, file_path: str) -> str:
        """Build rich context string for embedding."""
        parts = [f"File: {file_path}"]
        
        if self.chunk_type:
            parts.append(f"Type: {self.chunk_type}")
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.parent:
            parts.append(f"Parent: {self.parent}")
        
        parts.append("")  # Blank line before content
        parts.append(self.content)
        
        return "\n".join(parts)


class BaseChunker(ABC):
    """Abstract base for language-specific chunkers."""
    
    @abstractmethod
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """Chunk document content."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """File extensions this chunker handles."""
        pass


class MarkdownChunker(BaseChunker):
    """
    Markdown chunker - splits by headers.
    Preserves header hierarchy for context.
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.md', '.markdown']
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = []
        lines = content.split('\n')
        
        current_chunk_lines = []
        current_header = None
        current_header_level = 0
        parent_headers = {}  # level -> header text
        chunk_start_line = 1
        
        for i, line in enumerate(lines, 1):
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if header_match:
                # Save previous chunk if any content
                if current_chunk_lines and any(l.strip() for l in current_chunk_lines):
                    chunks.append(Chunk(
                        content='\n'.join(current_chunk_lines).strip(),
                        start_line=chunk_start_line,
                        end_line=i - 1,
                        chunk_type='section',
                        name=current_header,
                        parent=parent_headers.get(current_header_level - 1),
                    ))
                
                # Start new chunk
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                
                # Update parent hierarchy
                parent_headers[level] = header_text
                # Clear lower level parents
                for l in list(parent_headers.keys()):
                    if l > level:
                        del parent_headers[l]
                
                current_header = header_text
                current_header_level = level
                current_chunk_lines = [line]
                chunk_start_line = i
            else:
                current_chunk_lines.append(line)
        
        # Save final chunk
        if current_chunk_lines and any(l.strip() for l in current_chunk_lines):
            chunks.append(Chunk(
                content='\n'.join(current_chunk_lines).strip(),
                start_line=chunk_start_line,
                end_line=len(lines),
                chunk_type='section',
                name=current_header or 'document',
                parent=parent_headers.get(current_header_level - 1),
            ))
        
        return chunks


class PythonChunker(BaseChunker):
    """
    Python chunker - splits by function/class definitions.
    Falls back to regex until tree-sitter is installed.
    
    TODO: Upgrade to tree-sitter for proper AST parsing
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.py']
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = []
        lines = content.split('\n')
        
        # Patterns for Python constructs
        class_pattern = re.compile(r'^class\s+(\w+)')
        func_pattern = re.compile(r'^(async\s+)?def\s+(\w+)')
        
        current_chunk_lines = []
        current_type = None
        current_name = None
        current_parent = None
        chunk_start_line = 1
        current_indent = 0
        in_class = None
        
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            # Check for class definition
            class_match = class_pattern.match(stripped)
            if class_match and indent == 0:
                # Save previous chunk
                if current_chunk_lines:
                    chunks.append(self._make_chunk(
                        current_chunk_lines, chunk_start_line, i - 1,
                        current_type, current_name, current_parent
                    ))
                
                in_class = class_match.group(1)
                current_type = 'class'
                current_name = in_class
                current_parent = None
                current_chunk_lines = [line]
                chunk_start_line = i
                current_indent = 0
                continue
            
            # Check for function definition
            func_match = func_pattern.match(stripped)
            if func_match:
                func_name = func_match.group(2)
                
                # Top-level function or new method
                if indent == 0 or (indent == 4 and in_class):
                    # Save previous chunk
                    if current_chunk_lines:
                        chunks.append(self._make_chunk(
                            current_chunk_lines, chunk_start_line, i - 1,
                            current_type, current_name, current_parent
                        ))
                    
                    current_type = 'method' if in_class and indent == 4 else 'function'
                    current_name = func_name
                    current_parent = in_class if indent == 4 else None
                    current_chunk_lines = [line]
                    chunk_start_line = i
                    current_indent = indent
                    continue
            
            # Track class exit (back to indent 0)
            if in_class and stripped and indent == 0 and not class_match:
                in_class = None
            
            current_chunk_lines.append(line)
        
        # Save final chunk
        if current_chunk_lines:
            chunks.append(self._make_chunk(
                current_chunk_lines, chunk_start_line, len(lines),
                current_type, current_name, current_parent
            ))
        
        return [c for c in chunks if c.content.strip()]
    
    def _make_chunk(
        self,
        lines: List[str],
        start: int,
        end: int,
        chunk_type: Optional[str],
        name: Optional[str],
        parent: Optional[str]
    ) -> Chunk:
        content = '\n'.join(lines).strip()
        
        # Extract docstring if present
        docstring = None
        doc_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
        if doc_match:
            docstring = doc_match.group(1).strip()[:200]
        
        return Chunk(
            content=content,
            start_line=start,
            end_line=end,
            chunk_type=chunk_type or 'module',
            name=name,
            parent=parent,
            metadata={'docstring': docstring} if docstring else {},
        )


class PowerShellChunker(BaseChunker):
    """
    PowerShell chunker - splits by function definitions.
    Falls back to regex until tree-sitter-powershell is available.
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.ps1', '.psm1']
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = []
        lines = content.split('\n')
        
        # Pattern for PowerShell functions
        func_pattern = re.compile(r'^function\s+([\w-]+)', re.IGNORECASE)
        
        current_chunk_lines = []
        current_name = None
        chunk_start_line = 1
        brace_depth = 0
        in_function = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for function definition
            func_match = func_pattern.match(stripped)
            if func_match and not in_function:
                # Save previous chunk
                if current_chunk_lines:
                    chunks.append(Chunk(
                        content='\n'.join(current_chunk_lines).strip(),
                        start_line=chunk_start_line,
                        end_line=i - 1,
                        chunk_type='module' if not current_name else 'function',
                        name=current_name,
                    ))
                
                current_name = func_match.group(1)
                current_chunk_lines = [line]
                chunk_start_line = i
                brace_depth = 0
                in_function = True
                continue
            
            # Track braces for function body
            if in_function:
                brace_depth += stripped.count('{') - stripped.count('}')
                current_chunk_lines.append(line)
                
                if brace_depth <= 0 and '{' in ''.join(current_chunk_lines):
                    # Function ended
                    chunks.append(Chunk(
                        content='\n'.join(current_chunk_lines).strip(),
                        start_line=chunk_start_line,
                        end_line=i,
                        chunk_type='function',
                        name=current_name,
                    ))
                    current_chunk_lines = []
                    chunk_start_line = i + 1
                    current_name = None
                    in_function = False
            else:
                current_chunk_lines.append(line)
        
        # Save final chunk
        if current_chunk_lines:
            chunks.append(Chunk(
                content='\n'.join(current_chunk_lines).strip(),
                start_line=chunk_start_line,
                end_line=len(lines),
                chunk_type='function' if current_name else 'module',
                name=current_name,
            ))
        
        return [c for c in chunks if c.content.strip()]


class GenericChunker(BaseChunker):
    """
    Generic chunker for unknown file types.
    Uses fixed-size chunks with overlap.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['*']  # Fallback for all
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = []
        lines = content.split('\n')
        
        if not lines:
            return []
        
        # Split into roughly equal chunks by lines
        total_chars = len(content)
        if total_chars <= self.chunk_size:
            return [Chunk(
                content=content,
                start_line=1,
                end_line=len(lines),
                chunk_type='document',
                name=Path(file_path).name,
            )]
        
        current_chunk = []
        current_chars = 0
        chunk_start = 1
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            current_chars += len(line) + 1  # +1 for newline
            
            if current_chars >= self.chunk_size:
                chunks.append(Chunk(
                    content='\n'.join(current_chunk),
                    start_line=chunk_start,
                    end_line=i,
                    chunk_type='segment',
                ))
                
                # Start new chunk with overlap
                overlap_lines = max(1, self.overlap // 50)  # Roughly estimate lines
                current_chunk = current_chunk[-overlap_lines:]
                chunk_start = i - overlap_lines + 1
                current_chars = sum(len(l) for l in current_chunk)
        
        # Save final chunk
        if current_chunk:
            chunks.append(Chunk(
                content='\n'.join(current_chunk),
                start_line=chunk_start,
                end_line=len(lines),
                chunk_type='segment',
            ))
        
        return chunks


class SemanticChunker:
    """
    Main chunker that delegates to language-specific implementations.
    """
    
    def __init__(self):
        self.chunkers: Dict[str, BaseChunker] = {}
        self.generic_chunker = GenericChunker()
        
        # Register built-in chunkers
        self.register(MarkdownChunker())
        self.register(PythonChunker())
        self.register(PowerShellChunker())
    
    def register(self, chunker: BaseChunker):
        """Register a language-specific chunker."""
        for ext in chunker.supported_extensions:
            self.chunkers[ext] = chunker
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """Chunk document using appropriate chunker."""
        ext = Path(file_path).suffix.lower()
        
        chunker = self.chunkers.get(ext, self.generic_chunker)
        return chunker.chunk(content, file_path)
    
    def chunk_with_context(self, content: str, file_path: str) -> List[str]:
        """Get chunks as embedding-ready strings with context."""
        chunks = self.chunk(content, file_path)
        return [chunk.to_embedding_string(file_path) for chunk in chunks]


# Future tree-sitter integration
def get_tree_sitter_chunker(language: str) -> Optional[BaseChunker]:
    """
    Factory for tree-sitter based chunkers.
    TODO: Install py-tree-sitter and language grammars
    
    Usage after implementation:
        pip install tree-sitter
        pip install tree-sitter-python tree-sitter-languages
    """
    try:
        # Check if tree-sitter is available
        import tree_sitter
        from tree_sitter_languages import get_language, get_parser
        
        # Return tree-sitter based implementation
        # ... implementation TBD
        return None
    except ImportError:
        return None


if __name__ == "__main__":
    chunker = SemanticChunker()
    
    # Test Python chunking
    python_code = '''
"""Test module."""

def hello():
    """Say hello."""
    print("Hello!")

class Greeter:
    """A greeter class."""
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        """Greet by name."""
        print(f"Hello, {self.name}!")

def goodbye():
    print("Goodbye!")
'''
    
    print("=== Python Chunks ===")
    for chunk in chunker.chunk(python_code, "test.py"):
        print(f"[{chunk.chunk_type}] {chunk.name} (L{chunk.start_line}-{chunk.end_line})")
        print(f"  Parent: {chunk.parent}")
        print()
    
    # Test Markdown chunking
    markdown = '''
# Introduction

This is the intro.

## Getting Started

How to get started.

### Prerequisites

What you need.

## Advanced Topics

More complex stuff.
'''
    
    print("=== Markdown Chunks ===")
    for chunk in chunker.chunk(markdown, "README.md"):
        print(f"[{chunk.chunk_type}] {chunk.name} (L{chunk.start_line}-{chunk.end_line})")
        print(f"  Parent: {chunk.parent}")
        print()
