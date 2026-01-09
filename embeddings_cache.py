"""
RT937: Persistent Embedding Cache for ScornSpine
Caches embeddings on disk to speed up reindexing.
"""

import sqlite3
import hashlib
import pickle
from pathlib import Path
from typing import Optional, List

CACHE_DB = Path("f:/primewave-engine/haloscorn/scornspine/data/embeddings_cache.db")

def init_cache():
    """Initialize the cache database."""
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            file_path TEXT PRIMARY KEY,
            file_hash TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_file_hash(text: str) -> str:
    """Compute hash of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_cached_embedding(path: str, text: str) -> Optional[List[float]]:
    """Retrieve cached embedding if file hash matches."""
    if not CACHE_DB.exists():
        return None

    try:
        conn = sqlite3.connect(str(CACHE_DB))
        cursor = conn.execute(
            "SELECT embedding, file_hash FROM embeddings WHERE file_path = ?",
            (path,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            cached_embedding_blob, cached_hash = row
            current_hash = get_file_hash(text)
            if cached_hash == current_hash:
                return pickle.loads(cached_embedding_blob)
    except Exception as e:
        print(f"[Cache] Error reading cache: {e}")

    return None

def cache_embedding(path: str, text: str, embedding: List[float]):
    """Store embedding in cache."""
    try:
        conn = sqlite3.connect(str(CACHE_DB))
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (file_path, file_hash, embedding) VALUES (?, ?, ?)",
            (path, get_file_hash(text), pickle.dumps(embedding))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Cache] Error writing cache: {e}")

def cache_embeddings_batch(items: List[tuple]):
    """
    Store multiple embeddings in cache.
    items: List of (path, text, embedding)
    """
    try:
        conn = sqlite3.connect(str(CACHE_DB))
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (file_path, file_hash, embedding) VALUES (?, ?, ?)",
            [(path, get_file_hash(text), pickle.dumps(embedding)) for path, text, embedding in items]
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Cache] Error writing batch cache: {e}")
