"""
ScornSpine Cross-Encoder Reranker
=================================

RT1369: Implements cross-encoder reranking to improve retrieval relevance.

A cross-encoder takes (query, document) pairs and produces relevance scores,
unlike bi-encoders which embed query and docs separately. Cross-encoders are
more accurate but slower, so we use them to rerank the top-k results from
the hybrid search.

Model: ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking
- 6 layers, 22M parameters
- Good balance of speed and accuracy
- ~30-50ms per batch of 20 documents

Usage:
    from haloscorn.scornspine.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, results, top_k=10)
"""

import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("scornspine.reranker")

# RT1369: Cross-encoder reranking
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    logger.warning("[Reranker] sentence-transformers not installed - reranking disabled")


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval relevance.

    Uses ms-marco-MiniLM-L-6-v2 which is optimized for passage ranking.
    The model was trained on the MS MARCO dataset with 500k+ query-passage pairs.
    """

    # Default model - good balance of speed/accuracy
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name. Default: ms-marco-MiniLM-L-6-v2
            device: 'cuda' or 'cpu'. None for auto-detect.
            max_length: Maximum input length (query + document)
            batch_size: Batch size for inference
        """
        self._model: Optional[Any] = None
        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._load_time_ms = 0
        self._is_loaded = False

        if not CROSSENCODER_AVAILABLE:
            logger.warning("[Reranker] CrossEncoder not available")
            return

    def _ensure_loaded(self) -> bool:
        """Lazy load the model on first use."""
        if self._is_loaded:
            return True

        if not CROSSENCODER_AVAILABLE:
            return False

        try:
            start = time.time()

            # Detect device
            if self._device is None:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"[Reranker] Loading {self._model_name} on {self._device}...")

            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
                device=self._device
            )

            self._load_time_ms = (time.time() - start) * 1000
            self._is_loaded = True

            logger.info(f"[Reranker] Loaded in {self._load_time_ms:.0f}ms")
            return True

        except Exception as e:
            logger.error(f"[Reranker] Failed to load model: {e}")
            return False

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_key: str = "text",
        preserve_keys: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder scoring.

        Args:
            query: The search query
            results: List of result dicts from hybrid search
            top_k: Number of results to return. None returns all reranked.
            text_key: Key in result dict containing document text
            preserve_keys: Keys to copy from original results.
                          Default: all keys except scores

        Returns:
            List of results reranked by cross-encoder score, with
            'rerank_score' added to each result.
        """
        if not results:
            return []

        if not self._ensure_loaded():
            logger.warning("[Reranker] Model not loaded - returning original results")
            return results[:top_k] if top_k else results

        start = time.time()

        # Build (query, document) pairs
        pairs = []
        for result in results:
            # Get text from result, trying multiple keys
            doc_text = result.get(text_key, "")
            if not doc_text:
                doc_text = result.get("text_preview", "")
            if not doc_text:
                doc_text = str(result.get("metadata", {}).get("text", ""))[:self._max_length]

            pairs.append([query, doc_text])

        try:
            # Score all pairs
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False
            )

            # Add scores and sort
            reranked = []
            for i, result in enumerate(results):
                new_result = result.copy()
                new_result["rerank_score"] = float(scores[i])
                # Preserve original scores
                if "score" in result:
                    new_result["pre_rerank_score"] = result["score"]
                reranked.append(new_result)

            # Sort by rerank score (descending)
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Update primary score to be rerank score
            for r in reranked:
                r["score"] = r["rerank_score"]

            elapsed_ms = (time.time() - start) * 1000
            logger.debug(f"[Reranker] Reranked {len(results)} results in {elapsed_ms:.0f}ms")

            if top_k:
                return reranked[:top_k]
            return reranked

        except Exception as e:
            logger.error(f"[Reranker] Reranking failed: {e}")
            return results[:top_k] if top_k else results

    @property
    def is_available(self) -> bool:
        """Check if reranking is available."""
        return CROSSENCODER_AVAILABLE

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self._model_name

    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "available": self.is_available,
            "loaded": self.is_loaded,
            "model": self._model_name,
            "device": self._device,
            "load_time_ms": self._load_time_ms,
            "max_length": self._max_length,
            "batch_size": self._batch_size
        }


# Singleton instance for shared use
_reranker_instance: Optional[CrossEncoderReranker] = None

def get_reranker() -> CrossEncoderReranker:
    """Get or create the singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
