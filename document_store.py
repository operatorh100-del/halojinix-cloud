"""
ScornSpine Document Store - RT16500
SQLite-based document versioning and chunk tracking.

Based on JONAH RT16700 research:
- ACID transactions for document tracking
- Content hash for change detection
- Chunk-level tracking for FAISS vector IDs
"""

import sqlite3
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Document metadata."""
    id: int
    path: str
    content_hash: str
    version: int
    last_indexed: datetime
    chunk_count: int


@dataclass  
class Chunk:
    """Chunk metadata with FAISS vector reference."""
    id: int
    doc_id: int
    chunk_index: int
    vector_id: int  # FAISS internal ID
    start_line: Optional[int]
    end_line: Optional[int]
    content_preview: str


class DocumentStore:
    """
    SQLite document version tracking for ScornSpine.
    
    Why SQLite over JSON/Redis:
    - ACID transactions (critical for consistency)
    - Full SQL queries for debugging/analysis
    - Low memory footprint (disk-based)
    - Survives crashes without RDB config
    """
    
    DB_VERSION = 1  # Bump when schema changes
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("F:/primewave-engine/data/spine-documents.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self._init_schema()
        print(f"[DocumentStore] Initialized: {self.db_path}")
    
    def _init_schema(self):
        """Create tables if not exist."""
        cursor = self.conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunk_count INTEGER DEFAULT 0,
                file_size INTEGER DEFAULT 0,
                language TEXT,
                deleted INTEGER DEFAULT 0
            )
        """)
        
        # Chunks table - tracks individual vectors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                vector_id INTEGER NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                chunk_type TEXT,
                content_preview TEXT,
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        
        # Schema version tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Compaction history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                docs_before INTEGER,
                docs_after INTEGER,
                vectors_removed INTEGER,
                duration_seconds REAL
            )
        """)
        
        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_path ON documents(path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_vector ON chunks(vector_id)")
        
        # Check schema version
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        current_version = row[0] if row[0] else 0
        
        if current_version < self.DB_VERSION:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (self.DB_VERSION,))
            print(f"[DocumentStore] Schema upgraded to v{self.DB_VERSION}")
        
        self.conn.commit()
    
    def hash_content(self, content: str) -> str:
        """Generate content hash for change detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def has_changed(self, path: str, content: str) -> Tuple[bool, Optional[Document]]:
        """
        Check if content has changed since last indexing.
        
        Returns:
            (changed: bool, existing_doc: Optional[Document])
        """
        new_hash = self.hash_content(content)
        
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, path, content_hash, version, last_indexed, chunk_count FROM documents WHERE path = ? AND deleted = 0",
            (path,)
        )
        row = cursor.fetchone()
        
        if not row:
            return (True, None)  # New document
        
        existing = Document(
            id=row['id'],
            path=row['path'],
            content_hash=row['content_hash'],
            version=row['version'],
            last_indexed=row['last_indexed'],
            chunk_count=row['chunk_count']
        )
        
        if existing.content_hash == new_hash:
            return (False, existing)  # No change
        
        return (True, existing)  # Content changed
    
    def register_document(
        self,
        path: str,
        content: str,
        language: Optional[str] = None,
        file_size: Optional[int] = None
    ) -> Tuple[int, bool]:
        """
        Register a document for indexing.
        
        Returns:
            (doc_id, is_update)
        """
        content_hash = self.hash_content(content)
        cursor = self.conn.cursor()
        
        # Check existing
        cursor.execute("SELECT id, version FROM documents WHERE path = ?", (path,))
        row = cursor.fetchone()
        
        if row:
            # Update existing
            doc_id = row['id']
            new_version = row['version'] + 1
            cursor.execute("""
                UPDATE documents 
                SET content_hash = ?, version = ?, last_indexed = CURRENT_TIMESTAMP, 
                    file_size = ?, language = ?, deleted = 0
                WHERE id = ?
            """, (content_hash, new_version, file_size or len(content), language, doc_id))
            self.conn.commit()
            return (doc_id, True)
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO documents (path, content_hash, file_size, language)
                VALUES (?, ?, ?, ?)
            """, (path, content_hash, file_size or len(content), language))
            self.conn.commit()
            return (cursor.lastrowid, False)
    
    def add_chunk(
        self,
        doc_id: int,
        chunk_index: int,
        vector_id: int,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        chunk_type: Optional[str] = None,
        content_preview: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> int:
        """Register a chunk with its FAISS vector ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chunks (doc_id, chunk_index, vector_id, start_line, end_line, chunk_type, content_preview, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, chunk_index, vector_id, start_line, end_line, chunk_type, content_preview[:200] if content_preview else None, embedding_model))
        
        # Update chunk count
        cursor.execute("UPDATE documents SET chunk_count = chunk_count + 1 WHERE id = ?", (doc_id,))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_chunks(self, doc_id: int) -> List[Chunk]:
        """Get all chunks for a document."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, doc_id, chunk_index, vector_id, start_line, end_line, chunk_type, content_preview
            FROM chunks WHERE doc_id = ? ORDER BY chunk_index
        """, (doc_id,))
        
        return [
            Chunk(
                id=row['id'],
                doc_id=row['doc_id'],
                chunk_index=row['chunk_index'],
                vector_id=row['vector_id'],
                start_line=row['start_line'],
                end_line=row['end_line'],
                content_preview=row['content_preview']
            )
            for row in cursor.fetchall()
        ]
    
    def get_vector_ids(self, doc_id: int) -> List[int]:
        """Get FAISS vector IDs for a document (for soft deletion)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT vector_id FROM chunks WHERE doc_id = ?", (doc_id,))
        return [row['vector_id'] for row in cursor.fetchall()]
    
    def delete_chunks(self, doc_id: int) -> int:
        """Delete chunks for a document (call after soft-deleting vectors)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        cursor.execute("UPDATE documents SET chunk_count = 0 WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount
    
    def soft_delete_document(self, path: str) -> bool:
        """Mark document as deleted (for eventual cleanup)."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE documents SET deleted = 1 WHERE path = ?", (path,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def hard_delete_document(self, path: str) -> bool:
        """Permanently delete document and chunks."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE path = ?", (path,))
        row = cursor.fetchone()
        if not row:
            return False
        
        doc_id = row['id']
        cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return True
    
    def get_document_by_path(self, path: str) -> Optional[Document]:
        """Get document by path."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, path, content_hash, version, last_indexed, chunk_count 
            FROM documents WHERE path = ? AND deleted = 0
        """, (path,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return Document(
            id=row['id'],
            path=row['path'],
            content_hash=row['content_hash'],
            version=row['version'],
            last_indexed=row['last_indexed'],
            chunk_count=row['chunk_count']
        )
    
    def get_all_paths(self) -> List[str]:
        """Get all indexed document paths."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT path FROM documents WHERE deleted = 0")
        return [row['path'] for row in cursor.fetchall()]
    
    def log_compaction(
        self,
        started_at: datetime,
        completed_at: datetime,
        docs_before: int,
        docs_after: int,
        vectors_removed: int
    ):
        """Log a compaction run."""
        cursor = self.conn.cursor()
        duration = (completed_at - started_at).total_seconds()
        cursor.execute("""
            INSERT INTO compaction_log (started_at, completed_at, docs_before, docs_after, vectors_removed, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (started_at, completed_at, docs_before, docs_after, vectors_removed, duration))
        self.conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM documents WHERE deleted = 0")
        total_docs = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM chunks")
        total_chunks = cursor.fetchone()['total']
        
        cursor.execute("SELECT SUM(file_size) as total FROM documents WHERE deleted = 0")
        row = cursor.fetchone()
        total_size = row['total'] or 0
        
        cursor.execute("SELECT COUNT(*) as deleted FROM documents WHERE deleted = 1")
        deleted_docs = cursor.fetchone()['deleted']
        
        cursor.execute("""
            SELECT language, COUNT(*) as count 
            FROM documents WHERE deleted = 0 AND language IS NOT NULL 
            GROUP BY language
        """)
        by_language = {row['language']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute("SELECT MAX(completed_at) as last FROM compaction_log")
        row = cursor.fetchone()
        last_compaction = row['last'] if row else None
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "deleted_documents": deleted_docs,
            "by_language": by_language,
            "last_compaction": last_compaction,
            "db_path": str(self.db_path),
        }
    
    def cleanup_deleted(self) -> int:
        """Hard delete all soft-deleted documents. Call after FAISS compaction."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE deleted = 1")
        deleted_ids = [row['id'] for row in cursor.fetchall()]
        
        for doc_id in deleted_ids:
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        
        self.conn.commit()
        return len(deleted_ids)
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# Convenience function for integration
def detect_language(path: str) -> Optional[str]:
    """Detect language from file extension."""
    ext_map = {
        '.py': 'python',
        '.ps1': 'powershell',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.md': 'markdown',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.txt': 'text',
        '.sh': 'shell',
        '.bat': 'batch',
        '.css': 'css',
        '.html': 'html',
    }
    ext = Path(path).suffix.lower()
    return ext_map.get(ext)


if __name__ == "__main__":
    # Test
    store = DocumentStore()
    
    # Register document
    doc_id, is_update = store.register_document(
        path="test/example.py",
        content="def hello(): print('world')",
        language="python"
    )
    print(f"Registered doc {doc_id}, update={is_update}")
    
    # Add chunks
    store.add_chunk(doc_id, 0, vector_id=100, start_line=1, end_line=1, 
                    chunk_type="function", content_preview="def hello():")
    
    # Check change
    changed, existing = store.has_changed("test/example.py", "def hello(): print('world')")
    print(f"Changed: {changed}")
    
    changed, existing = store.has_changed("test/example.py", "def hello(): print('new')")
    print(f"Changed after edit: {changed}")
    
    # Stats
    print(f"Stats: {store.get_stats()}")
    
    store.close()
