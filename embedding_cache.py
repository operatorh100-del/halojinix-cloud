"""
Embedding Cache for ScornSpine
==============================

RT982/A2: SQLite-backed cache for document embeddings.

Caches embeddings by content hash to avoid re-embedding unchanged documents.
Expected speedup: 5-10x for incremental reindex operations.

Usage:
    from haloscorn.scornspine.embedding_cache import EmbeddingCache, CachedEmbedding

    cache = EmbeddingCache()
    cached_model = CachedEmbedding(base_model, cache)

    # Use cached_model in place of base embedding model
    Settings.embed_model = cached_model

RT993: Fixed BaseEmbedding inheritance per JONAH's research.
"""

import sqlite3
import hashlib
import logging
import pickle
import threading
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime

# RT993: Import BaseEmbedding and PrivateAttr for proper llama-index integration
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "haloscorn" / "scornspine" / "cache"
CACHE_DB = CACHE_DIR / "embeddings.db"


class EmbeddingCache:
    """
    SQLite-backed embedding cache.

    Stores embeddings keyed by content hash to avoid re-computing
    embeddings for unchanged documents.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or CACHE_DB
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._stats = {"hits": 0, "misses": 0}
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self._conn.cursor()

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL
                )
            """)

            # Create index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash
                ON embeddings(content_hash)
            """)

            self._conn.commit()

            # Get stats
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            logger.info(f"[EmbeddingCache] Initialized with {count} cached embeddings")

        except Exception as e:
            logger.error(f"[EmbeddingCache] Failed to init DB: {e}")
            self._conn = None

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA256 hash of text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, content_hash: str, model_name: str) -> Optional[List[float]]:
        """
        Get cached embedding by content hash.

        Args:
            content_hash: SHA256 hash of the text content
            model_name: Name of embedding model (for validation)

        Returns:
            List of floats (embedding) or None if not cached
        """
        if self._conn is None:
            return None

        try:
            with self._lock:
                cursor = self._conn.cursor()
                cursor.execute(
                    "SELECT embedding FROM embeddings WHERE content_hash = ? AND model_name = ?",
                    (content_hash, model_name)
                )
                row = cursor.fetchone()

                if row:
                    # Update last_used timestamp
                    cursor.execute(
                        "UPDATE embeddings SET last_used = ? WHERE content_hash = ?",
                        (datetime.now().isoformat(), content_hash)
                    )
                    self._conn.commit()

                    self._stats["hits"] += 1
                    return pickle.loads(row[0])
                else:
                    self._stats["misses"] += 1
                    return None

        except Exception as e:
            logger.warning(f"[EmbeddingCache] Get failed: {e}")
            return None

    def put(self, content_hash: str, embedding: List[float], model_name: str):
        """
        Store embedding in cache.

        Args:
            content_hash: SHA256 hash of the text content
            embedding: The embedding vector
            model_name: Name of embedding model
        """
        if self._conn is None:
            return

        try:
            with self._lock:
                cursor = self._conn.cursor()
                now = datetime.now().isoformat()

                cursor.execute(
                    """INSERT OR REPLACE INTO embeddings
                       (content_hash, embedding, model_name, dim, created_at, last_used)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        content_hash,
                        pickle.dumps(embedding),
                        model_name,
                        len(embedding),
                        now,
                        now
                    )
                )
                self._conn.commit()

        except Exception as e:
            logger.warning(f"[EmbeddingCache] Put failed: {e}")

    def get_batch(self, content_hashes: List[str], model_name: str) -> dict:
        """
        Get multiple cached embeddings at once.

        Args:
            content_hashes: List of SHA256 hashes
            model_name: Name of embedding model

        Returns:
            Dict mapping hash -> embedding (only for cached items)
        """
        if self._conn is None or not content_hashes:
            return {}

        try:
            with self._lock:
                cursor = self._conn.cursor()
                placeholders = ",".join("?" * len(content_hashes))
                cursor.execute(
                    f"SELECT content_hash, embedding FROM embeddings WHERE content_hash IN ({placeholders}) AND model_name = ?",
                    (*content_hashes, model_name)
                )

                results = {}
                for row in cursor.fetchall():
                    results[row[0]] = pickle.loads(row[1])
                    self._stats["hits"] += 1

                self._stats["misses"] += len(content_hashes) - len(results)
                return results

        except Exception as e:
            logger.warning(f"[EmbeddingCache] Batch get failed: {e}")
            return {}

    def put_batch(self, items: List[tuple], model_name: str):
        """
        Store multiple embeddings at once.

        Args:
            items: List of (content_hash, embedding) tuples
            model_name: Name of embedding model
        """
        if self._conn is None or not items:
            return

        try:
            with self._lock:
                cursor = self._conn.cursor()
                now = datetime.now().isoformat()

                cursor.executemany(
                    """INSERT OR REPLACE INTO embeddings
                       (content_hash, embedding, model_name, dim, created_at, last_used)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        (h, pickle.dumps(e), model_name, len(e), now, now)
                        for h, e in items
                    ]
                )
                self._conn.commit()

        except Exception as e:
            logger.warning(f"[EmbeddingCache] Batch put failed: {e}")

    def clear(self):
        """Clear all cached embeddings."""
        if self._conn is None:
            return

        try:
            with self._lock:
                cursor = self._conn.cursor()
                cursor.execute("DELETE FROM embeddings")
                self._conn.commit()
                logger.info("[EmbeddingCache] Cache cleared")
        except Exception as e:
            logger.error(f"[EmbeddingCache] Clear failed: {e}")

    def prune(self, max_age_days: int = 30):
        """Remove embeddings not used in the last N days."""
        if self._conn is None:
            return

        try:
            with self._lock:
                cursor = self._conn.cursor()
                cutoff = datetime.now().isoformat()[:10]  # Just date part
                cursor.execute(
                    "DELETE FROM embeddings WHERE last_used < date(?, '-' || ? || ' days')",
                    (cutoff, max_age_days)
                )
                deleted = cursor.rowcount
                self._conn.commit()
                if deleted > 0:
                    logger.info(f"[EmbeddingCache] Pruned {deleted} stale embeddings")
        except Exception as e:
            logger.warning(f"[EmbeddingCache] Prune failed: {e}")

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0

        size = 0
        count = 0
        if self._conn:
            try:
                cursor = self._conn.cursor()
                cursor.execute("SELECT COUNT(*), SUM(LENGTH(embedding)) FROM embeddings")
                row = cursor.fetchone()
                count = row[0] or 0
                size = row[1] or 0
            except (sqlite3.Error, AttributeError):
                pass

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate_percent": round(hit_rate, 1),
            "cached_embeddings": count,
            "cache_size_mb": round(size / 1024 / 1024, 2) if size else 0
        }

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class CachedEmbedding(BaseEmbedding):
    """
    RT993: Cached embedding wrapper that inherits from BaseEmbedding.

    Intercepts embedding calls, checks cache first, and only
    computes new embeddings for uncached content.

    Fixed per JONAH's research: Must inherit from BaseEmbedding
    for llama-index Settings.embed_model compatibility.
    """

    # RT993: Use PrivateAttr for non-serialized fields (Pydantic requirement)
    _base_model: Any = PrivateAttr()
    _cache: EmbeddingCache = PrivateAttr()
    _model_name_internal: str = PrivateAttr()

    def __init__(
        self,
        base_model: Any,
        cache: EmbeddingCache,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            base_model: The underlying embedding model (HuggingFaceEmbedding, etc)
            cache: EmbeddingCache instance
            model_name: Name to use for cache key (default: base_model.model_name)
        """
        # RT1086: Get model name BEFORE super().__init__() but don't assign to self yet
        # Pydantic v2 requires super().__init__() before setting PrivateAttrs
        resolved_model_name = model_name or getattr(base_model, 'model_name', 'unknown')

        # RT993: Initialize base class with required parameters FIRST
        super().__init__(
            model_name=resolved_model_name,
            embed_batch_size=getattr(base_model, 'embed_batch_size', 10),
            **kwargs
        )

        # RT1086: NOW we can safely set PrivateAttrs (after super().__init__)
        self._model_name_internal = resolved_model_name
        self._base_model = base_model
        self._cache = cache

        logger.info(f"[CachedEmbedding] Initialized wrapping {self._model_name_internal}")

    @classmethod
    def class_name(cls) -> str:
        """RT993: Required by llama-index for model identification."""
        return "CachedEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding, using cache."""
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text, using cache."""
        content_hash = EmbeddingCache.compute_hash(text)

        # Check cache
        cached = self._cache.get(content_hash, self._model_name_internal)
        if cached is not None:
            return cached

        # Compute and cache
        embedding = self._base_model._get_text_embedding(text)
        self._cache.put(content_hash, embedding, self._model_name_internal)
        return embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts, using cache."""
        if not texts:
            return []

        # Compute hashes
        hashes = [EmbeddingCache.compute_hash(t) for t in texts]

        # Batch cache lookup
        cached = self._cache.get_batch(hashes, self._model_name_internal)

        # Identify uncached texts
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, (text, h) in enumerate(zip(texts, hashes)):
            if h in cached:
                results[i] = cached[h]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute uncached embeddings
        if uncached_texts:
            new_embeddings = self._base_model._get_text_embeddings(uncached_texts)

            # Store in cache and results
            to_cache = []
            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                to_cache.append((hashes[idx], emb))

            self._cache.put_batch(to_cache, self._model_name_internal)

            logger.info(f"[CachedEmbedding] {len(texts)} texts: {len(cached)} cached, {len(uncached_texts)} computed")
        else:
            logger.info(f"[CachedEmbedding] {len(texts)} texts: ALL CACHED")

        return results  # type: ignore

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version - delegates to sync for now."""
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version - delegates to sync for now."""
        return self._get_query_embedding(query)


# -------------------------------------------------------------------------------
# CLI / TESTING
# -------------------------------------------------------------------------------

def main():
    """Test embedding cache."""
    import argparse

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

    parser = argparse.ArgumentParser(description="Embedding Cache")
    parser.add_argument("command", choices=["stats", "clear", "prune"])
    parser.add_argument("--days", type=int, default=30, help="Max age for prune")

    args = parser.parse_args()

    cache = EmbeddingCache()

    if args.command == "stats":
        import json
        print(json.dumps(cache.stats, indent=2))
    elif args.command == "clear":
        cache.clear()
        print("Cache cleared")
    elif args.command == "prune":
        cache.prune(args.days)
        print(f"Pruned embeddings older than {args.days} days")

    cache.close()


if __name__ == "__main__":
    main()
