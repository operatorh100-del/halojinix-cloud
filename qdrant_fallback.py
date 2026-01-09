"""
QdrantCloud Fallback - RT31701
When local FAISS returns low scores, fall back to 703K cloud vectors.

Usage:
    from haloscorn.scornspine.qdrant_fallback import qdrant_fallback

    # In your search flow:
    local_results = spine.query(query, top_k)
    if needs_fallback(local_results):
        cloud_results = qdrant_fallback.search(embedding_vector, top_k)
        results = merge_results(local_results, cloud_results)

Architecture:
    - Local FAISS: 22,973 docs (fast, ~50ms)
    - Qdrant Cloud: 703,382 vectors (30x more, ~200ms)
    - Fallback triggers when: top_score < 0.5 OR result_count < 3

Human directive: "go baby go!" - Build infrastructure for persistence.
"""

import os
import time
import httpx
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Qdrant Cloud configuration
QDRANT_URL = os.getenv("QDRANT_URL", "").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "halojinix-spine")

# Fallback thresholds (RT31701)
MIN_SCORE_THRESHOLD = 0.45  # Below this score, fallback kicks in
MIN_RESULTS_THRESHOLD = 3   # Below this count, fallback kicks in
FALLBACK_TIMEOUT = 10.0     # Cloud query timeout in seconds


class QdrantFallback:
    """
    RT31701: Qdrant Cloud fallback for low-confidence local searches.

    The 703K cloud vectors catch what local 22K misses.
    """

    def __init__(self):
        self._client: Optional[httpx.Client] = None
        self._available = bool(QDRANT_URL and QDRANT_API_KEY)
        self._stats = {
            "calls": 0,
            "hits": 0,
            "total_ms": 0.0,
            "errors": 0
        }

        if self._available:
            print(f"[RT31701] QdrantCloud fallback ENABLED - Collection: {QDRANT_COLLECTION}")
        else:
            print("[RT31701] QdrantCloud fallback DISABLED - No credentials in .env")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats.copy()

    def _get_client(self) -> httpx.Client:
        """Lazy-init httpx client with auth headers."""
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "api-key": QDRANT_API_KEY,
                    "Content-Type": "application/json"
                },
                timeout=FALLBACK_TIMEOUT
            )
        return self._client

    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        collection: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search Qdrant Cloud with pre-computed embedding vector.

        Args:
            vector: 768-dim embedding from ScornSpine embedder
            top_k: Number of results to return
            collection: Override collection name (default: halojinix-spine)

        Returns:
            List of results with: path, text, score, category, source='qdrant-cloud'
        """
        if not self._available:
            return []

        collection = collection or QDRANT_COLLECTION
        url = f"{QDRANT_URL}/collections/{collection}/points/search"

        payload = {
            "vector": vector,
            "limit": top_k,
            "with_payload": True
        }

        start = time.time()
        self._stats["calls"] += 1

        try:
            client = self._get_client()
            response = client.post(url, json=payload)
            response.raise_for_status()

            elapsed_ms = (time.time() - start) * 1000
            self._stats["total_ms"] += elapsed_ms

            data = response.json()
            points = data.get("result", [])

            if points:
                self._stats["hits"] += 1

            # Convert Qdrant format to ScornSpine format
            results = []
            for point in points:
                payload = point.get("payload", {})

                # Handle various payload field names from Qdrant
                path = (
                    payload.get("filepath") or
                    payload.get("path") or
                    payload.get("file") or
                    "unknown"
                )
                text = payload.get("text") or payload.get("content") or ""
                category = payload.get("category") or "cloud"

                results.append({
                    "path": path,
                    "text": text,
                    "score": round(point.get("score", 0.0), 4),
                    "category": category,
                    "source": "qdrant-cloud",  # Mark as cloud result
                    "node_id": payload.get("node_id"),
                    "latency_ms": round(elapsed_ms, 2)
                })

            print(f"[RT31701] Cloud search: {len(results)} results in {elapsed_ms:.0f}ms")
            return results

        except Exception as e:
            self._stats["errors"] += 1
            elapsed_ms = (time.time() - start) * 1000
            print(f"[RT31701] Cloud search ERROR: {e} ({elapsed_ms:.0f}ms)")
            return []

    def close(self):
        """Close httpx client."""
        if self._client:
            self._client.close()
            self._client = None


def needs_fallback(results: List[Dict[str, Any]]) -> bool:
    """
    RT31701: Determine if cloud fallback should be triggered.

    Triggers when:
    - No results
    - Too few results (< MIN_RESULTS_THRESHOLD)
    - Low confidence (top score < MIN_SCORE_THRESHOLD)
    """
    if not results:
        return True

    if len(results) < MIN_RESULTS_THRESHOLD:
        return True

    # Check top score (higher is better in cosine similarity)
    top_score = results[0].get("score", 0.0)
    if top_score < MIN_SCORE_THRESHOLD:
        return True

    return False


def merge_results(
    local_results: List[Dict[str, Any]],
    cloud_results: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    RT31701: Merge local FAISS + cloud Qdrant results.

    Strategy:
    - Deduplicate by path
    - Sort by score (descending)
    - Return top_k
    """
    # Create path-based dedup map (cloud results have source='qdrant-cloud')
    seen_paths = set()
    merged = []

    # Local results first (they have fresher content)
    for r in local_results:
        path = r.get("path", "")
        if path not in seen_paths:
            seen_paths.add(path)
            if "source" not in r:
                r["source"] = "local-faiss"
            merged.append(r)

    # Then cloud results (fills gaps)
    for r in cloud_results:
        path = r.get("path", "")
        if path not in seen_paths:
            seen_paths.add(path)
            merged.append(r)

    # Sort by score descending
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return merged[:top_k]


# Module-level singleton
qdrant_fallback = QdrantFallback()


# Cleanup on module unload
import atexit
atexit.register(qdrant_fallback.close)
