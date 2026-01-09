"""
BM25 Index for ScornSpine Hybrid Search
=======================================

RT263: Phase 1 - Adds BM25 keyword search alongside vector search.

BM25 excels at exact keyword matching where vector embeddings may miss:
- Technical terms (ADR-0050, RT263)
- Proper nouns (HALO, VERA, JONAH)
- Specific file paths
- Version numbers

Uses rank-bm25 library (already installed in ChatRTX venv).

Usage:
    from haloscorn.scornspine.bm25_index import BM25Index

    bm25 = BM25Index()
    bm25.build_index(documents)  # List of {"id": str, "text": str, "metadata": dict}
    results = bm25.search("particle count standards", top_k=5)
"""

import json
import pickle
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check if rank-bm25 is available
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("[BM25Index] rank-bm25 not installed - BM25 search disabled")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BM25_INDEX_FILE = PROJECT_ROOT / "haloscorn" / "scornspine" / "index" / "bm25_index.pkl"


# -------------------------------------------------------------------------------
# TOKENIZATION
# -------------------------------------------------------------------------------

# Common English stopwords (minimal set - keep technical terms)
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'although', 'though', 'this', 'that',
    'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing.

    Preserves:
    - Technical terms (ADR-0050, RT263)
    - Agent names (HALO, VERA, JONAH)
    - File paths (docs/decisions/ADR-0050.md)
    - Version numbers (v85, 1.0.0)

    Removes:
    - Common stopwords
    - Very short tokens (< 2 chars)
    """
    # Lowercase but preserve structure
    text_lower = text.lower()

    # Split on whitespace and punctuation, but preserve some patterns
    # Pattern: alphanumeric + underscores + hyphens + dots (for versions/paths)
    tokens = re.findall(r'[a-z0-9_\-\.]+', text_lower)

    # Filter
    filtered = []
    for token in tokens:
        # Skip stopwords
        if token in STOPWORDS:
            continue
        # Skip very short tokens (but keep version numbers like "v1")
        if len(token) < 2 and not token.isdigit():
            continue
        # Skip pure punctuation
        if re.match(r'^[\.\-_]+$', token):
            continue
        filtered.append(token)

    return filtered


# -------------------------------------------------------------------------------
# BM25 INDEX
# -------------------------------------------------------------------------------

@dataclass
class BM25Document:
    """A document in the BM25 index."""
    id: str
    tokens: List[str]
    metadata: Dict[str, Any]
    text: str  # Original text for retrieval


class BM25Index:
    """
    BM25 index for keyword-based retrieval.

    Complements vector search by handling:
    - Exact keyword matches
    - Technical terms and identifiers
    - Proper nouns and agent names

    RT8800: Tunable k1/b parameters for BM25Okapi.
    - k1 controls term frequency saturation (higher = more weight to repeated terms)
    - b controls document length normalization (higher = more penalty for long docs)

    Defaults (k1=1.5, b=0.75) are tuned for web documents.
    For technical docs, k1=1.2, b=0.5 often works better.
    """

    # RT8800: BM25 tuning parameters
    # For technical documentation with consistent formatting:
    DEFAULT_K1 = 1.2  # Lower than 1.5 - less boost for term repetition
    DEFAULT_B = 0.5   # Lower than 0.75 - less length normalization (our docs vary in length)

    def __init__(self, k1: float = None, b: float = None):
        self._bm25 = None
        self._documents: List[BM25Document] = []
        self._is_built = False
        self._k1 = k1 if k1 is not None else self.DEFAULT_K1
        self._b = b if b is not None else self.DEFAULT_B

    def build_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Build BM25 index from documents.

        Args:
            documents: List of {"id": str, "text": str, "metadata": dict}

        Returns:
            True if successful
        """
        try:
            if not BM25_AVAILABLE:
                logger.warning("[BM25] rank-bm25 not available - cannot build index")
                return False

            logger.info(f"[BM25] Building index from {len(documents)} documents...")

            # Tokenize all documents
            self._documents = []
            corpus = []

            for doc in documents:
                tokens = tokenize(doc.get("text", ""))
                if tokens:  # Skip empty documents
                    bm25_doc = BM25Document(
                        id=doc.get("id", ""),
                        tokens=tokens,
                        metadata=doc.get("metadata", {}),
                        text=doc.get("text", "")[:1000]  # Store first 1000 chars for preview
                    )
                    self._documents.append(bm25_doc)
                    corpus.append(tokens)

            if not corpus:
                logger.warning("[BM25] No documents to index!")
                return False

            # Build BM25 index with tuned parameters
            # RT8800: k1 and b tuned for technical documentation
            self._bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b)
            self._is_built = True

            logger.info(f"[BM25] [OK] Index built: {len(self._documents)} documents, avg {sum(len(d.tokens) for d in self._documents) / len(self._documents):.0f} tokens/doc (k1={self._k1}, b={self._b})")

            return True

        except ImportError:
            logger.error("[BM25] rank_bm25 not installed!")
            return False
        except Exception as e:
            logger.error(f"[BM25] Build failed: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the BM25 index.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of {"id": str, "score": float, "metadata": dict, "text_preview": str}
        """
        if not self._is_built or self._bm25 is None:
            logger.warning("[BM25] Index not built!")
            return []

        try:
            # Tokenize query
            query_tokens = tokenize(query)
            if not query_tokens:
                return []

            # Get BM25 scores
            scores = self._bm25.get_scores(query_tokens)

            # Get top-k indices
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            # Build results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include positive scores
                    doc = self._documents[idx]
                    results.append({
                        "id": doc.id,
                        "score": float(scores[idx]),
                        "metadata": doc.metadata,
                        "text_preview": doc.text[:200]
                    })

            return results

        except Exception as e:
            logger.error(f"[BM25] Search failed: {e}")
            return []

    def save(self, path: Optional[Path] = None) -> bool:
        """Save BM25 index to disk."""
        save_path = path or BM25_INDEX_FILE
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'bm25': self._bm25,
                    'documents': self._documents,
                    'is_built': self._is_built,
                }, f)
            logger.info(f"[BM25] Index saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"[BM25] Save failed: {e}")
            return False

    def load(self, path: Optional[Path] = None) -> bool:
        """Load BM25 index from disk."""
        load_path = path or BM25_INDEX_FILE
        try:
            if not load_path.exists():
                logger.warning(f"[BM25] No index file at {load_path}")
                return False

            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            self._bm25 = data.get('bm25')
            self._documents = data.get('documents', [])
            self._is_built = data.get('is_built', False)

            # RT12071: Validate loaded data isn't corrupt/empty
            if not self._is_built or self._bm25 is None or len(self._documents) == 0:
                logger.warning(f"[BM25] Loaded file is corrupt or empty (is_built={self._is_built}, docs={len(self._documents)}, bm25={self._bm25 is not None})")
                self._bm25 = None
                self._documents = []
                self._is_built = False
                return False

            logger.info(f"[BM25] Index loaded: {len(self._documents)} documents")
            return True

        except Exception as e:
            logger.error(f"[BM25] Load failed: {e}")
            return False

    @property
    def document_count(self) -> int:
        """Number of documents in the index."""
        return len(self._documents)

    @property
    def is_ready(self) -> bool:
        """Whether the index is ready for queries."""
        return self._is_built and self._bm25 is not None


# -------------------------------------------------------------------------------
# RANK FUSION
# -------------------------------------------------------------------------------

def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF is more robust than linear combination because it works on ranks
    rather than raw scores (which may have different scales).

    Formula: RRF(d) = ? (weight / (k + rank(d)))

    Args:
        vector_results: Results from vector search [{"id": ..., "score": ...}]
        bm25_results: Results from BM25 search [{"id": ..., "score": ...}]
        k: Ranking constant (default 60, standard in literature)
        vector_weight: Weight for vector results (default 0.7)
        bm25_weight: Weight for BM25 results (default 0.3)

    Returns:
        Fused results sorted by combined score
    """
    # Build ID -> result maps and compute RRF scores
    fused_scores: Dict[str, float] = {}
    result_data: Dict[str, Dict[str, Any]] = {}

    # Process vector results
    for rank, result in enumerate(vector_results, start=1):
        doc_id = result.get("id") or result.get("filepath", "")
        if doc_id:
            rrf_score = vector_weight / (k + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
            if doc_id not in result_data:
                result_data[doc_id] = result.copy()
                result_data[doc_id]["vector_rank"] = rank
                result_data[doc_id]["vector_score"] = result.get("score", 0)

    # Process BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result.get("id") or result.get("filepath", "")
        if doc_id:
            rrf_score = bm25_weight / (k + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
            if doc_id not in result_data:
                result_data[doc_id] = result.copy()
            result_data[doc_id]["bm25_rank"] = rank
            result_data[doc_id]["bm25_score"] = result.get("score", 0)

    # Sort by fused score and build final results
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        result = result_data[doc_id]
        result["fused_score"] = fused_scores[doc_id]
        result["retrieval_method"] = "hybrid"
        fused_results.append(result)

    return fused_results


# -------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------

def main():
    """Test BM25 index."""
    import argparse

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

    parser = argparse.ArgumentParser(description="BM25 Index Test")
    parser.add_argument("command", choices=["build", "search", "test"])
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=5)

    args = parser.parse_args()

    bm25 = BM25Index()

    if args.command == "build":
        # Test with some sample documents
        sample_docs = [
            {"id": "doc1", "text": "HALO is the implementation agent for GPU rendering", "metadata": {"agent": "halo"}},
            {"id": "doc2", "text": "VERA handles coordination and routing between agents", "metadata": {"agent": "vera"}},
            {"id": "doc3", "text": "JONAH does research and architecture decisions", "metadata": {"agent": "jonah"}},
            {"id": "doc4", "text": "Particle count minimum is 120 million for production", "metadata": {"category": "standards"}},
            {"id": "doc5", "text": "ADR-0050 defines the particle count standards", "metadata": {"category": "decisions"}},
        ]
        bm25.build_index(sample_docs)
        bm25.save()

    elif args.command == "search":
        if not args.query:
            print("Error: --query required")
            return
        bm25.load()
        results = bm25.search(args.query, top_k=args.top_k)
        print(json.dumps(results, indent=2))

    elif args.command == "test":
        # Quick self-test
        sample_docs = [
            {"id": "doc1", "text": "HALO renders particles on GPU", "metadata": {}},
            {"id": "doc2", "text": "VERA coordinates agent communication", "metadata": {}},
            {"id": "doc3", "text": "particle count is 120 million minimum", "metadata": {}},
        ]
        bm25.build_index(sample_docs)

        print("Query: 'particle'")
        results = bm25.search("particle", top_k=3)
        for r in results:
            print(f"  {r['id']}: {r['score']:.2f}")

        print("\nQuery: 'HALO GPU'")
        results = bm25.search("HALO GPU", top_k=3)
        for r in results:
            print(f"  {r['id']}: {r['score']:.2f}")


if __name__ == "__main__":
    main()
