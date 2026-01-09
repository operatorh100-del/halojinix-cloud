"""
COCONUT - Continuous Thought in Latent Space
============================================

Inspired by Meta's COCONUT paper (Dec 2024): "Coconut: Chain of Continuous Thought"

Traditional LLM reasoning:
    "Let me think... First, I need to... Oh wait, that's not right..."
    (Verbose, token-heavy, exposes intermediate reasoning)

COCONUT reasoning:
    query_vector ? think_step(v) ? think_step(v) ? final_vector ? decode
    (Reasoning happens in latent space, only final output is text)

Why this matters for HALOJINIX:
    - We already have embeddings from ScornSpine
    - We can iterate on vectors without generating tokens
    - Final synthesis uses nearest-neighbor context, not verbose reasoning

This module implements "Experiment 3: Latent Reasoning" from:
    docs/research/latent-space-deep-dive.md

Two modes:
    1. Full COCONUT (requires /embed and /search_vector endpoints)
    2. Text-based approximation (works with existing /search endpoint)

Author: JONAH (Thread-Puller)
Date: 2025-12-30
RT4400: Built during Free Agency / Agent Night
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ThoughtState:
    """
    A single state in the continuous thought chain.

    Like COCONUT's "continuous thought" - the vector IS the reasoning state.
    We don't decode it to tokens until we need the final answer.
    """
    vector: np.ndarray          # The current thought vector (768-dim)
    step: int                   # Which thought step this is
    confidence: float           # How "settled" the thought is (0.0 - 1.0)
    energy: float               # "Distance traveled" in this step
    sources: List[str]          # IDs of nodes that influenced this state

    def magnitude(self) -> float:
        """Vector magnitude - how "strong" is this thought?"""
        return float(np.linalg.norm(self.vector))

    def similarity(self, other: 'ThoughtState') -> float:
        """Cosine similarity with another thought state."""
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_a * norm_b))


@dataclass
class ThoughtChain:
    """
    A chain of continuous thoughts.

    This is analogous to COCONUT's chain where hidden states are fed back
    as input, but we work directly with embedding vectors.
    """
    query: str                          # Original query
    states: List[ThoughtState]          # All intermediate states
    final_state: Optional[ThoughtState] # Converged state (if any)
    converged: bool                     # Did we reach convergence?
    context_used: List[Dict]            # What context influenced reasoning

    def trace(self) -> str:
        """Human-readable trace of the thought chain."""
        lines = [f"Query: {self.query}", ""]
        for state in self.states:
            lines.append(f"Step {state.step}: conf={state.confidence:.3f}, energy={state.energy:.4f}, mag={state.magnitude():.3f}")
            if state.sources:
                lines.append(f"  Sources: {', '.join(state.sources[:3])}...")
        if self.converged:
            lines.append(f"\n? Converged at step {len(self.states)}")
        else:
            lines.append(f"\n[WARN] Did not converge (max steps reached)")
        return '\n'.join(lines)


class CoconutReasoner:
    """
    COCONUT-inspired continuous thought reasoner.

    This is NOT true COCONUT (which uses transformer hidden states).
    This is an approximation using embedding space operations:

    1. Encode query to vector
    2. Find relevant context vectors
    3. Iterate: blend current vector with context (like attention)
    4. Check for convergence
    5. Decode final vector to context

    The key insight from COCONUT: reasoning doesn't need to be in tokens.
    The vector IS the thought. We only decode when we need an answer.
    """

    VECTOR_SIZE = 768  # intfloat/multilingual-e5-base

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        search_fn: Callable[[np.ndarray, int], List[Tuple[str, np.ndarray, float]]],
        max_steps: int = 5,
        convergence_threshold: float = 0.02,
        momentum: float = 0.3,
        context_blend: float = 0.4
    ):
        """
        Initialize the COCONUT reasoner.

        Args:
            embed_fn: Function that embeds text to vector (query -> 768-dim vector)
            search_fn: Function that finds nearest neighbors (vector, k -> [(id, vector, score), ...])
            max_steps: Maximum reasoning steps before forcing output
            convergence_threshold: Stop when energy drops below this
            momentum: How much of previous direction to keep (0.0 = pure averaging)
            context_blend: How much to blend context into current state
        """
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.momentum = momentum
        self.context_blend = context_blend

    def reason(self, query: str, top_k: int = 5, verbose: bool = False) -> ThoughtChain:
        """
        Reason continuously in latent space.

        Args:
            query: The question or task
            top_k: How many context vectors to consider per step
            verbose: Whether to log each step

        Returns:
            ThoughtChain with all intermediate states and final answer
        """
        # Step 0: Encode query
        query_vector = np.array(self.embed_fn(query), dtype=np.float32)
        if verbose:
            logger.info(f"[COCONUT] Query encoded: mag={np.linalg.norm(query_vector):.3f}")

        states = []
        current_vector = query_vector.copy()
        previous_direction = np.zeros_like(current_vector)
        all_sources = []
        context_used = []

        for step in range(1, self.max_steps + 1):
            # Find relevant context at current position
            neighbors = self.search_fn(current_vector, top_k)

            if not neighbors:
                if verbose:
                    logger.warning(f"[COCONUT] Step {step}: No neighbors found")
                break

            # Collect sources
            step_sources = [n[0] for n in neighbors]
            all_sources.extend(step_sources)

            # Weight neighbors by similarity (attention-like)
            total_weight = sum(n[2] for n in neighbors)
            if total_weight == 0:
                break

            # Compute context centroid (weighted average)
            context_centroid = np.zeros_like(current_vector)
            for node_id, vector, score in neighbors:
                weight = score / total_weight
                context_centroid += weight * np.array(vector, dtype=np.float32)

            # Direction toward context
            direction = context_centroid - current_vector

            # Apply momentum from previous step
            direction = (1 - self.momentum) * direction + self.momentum * previous_direction

            # Move toward context (with blend factor)
            new_vector = current_vector + self.context_blend * direction

            # Normalize to unit sphere (keeps vectors comparable)
            norm = np.linalg.norm(new_vector)
            if norm > 0:
                new_vector = new_vector / norm * np.linalg.norm(query_vector)

            # Compute energy (how much we moved)
            energy = float(np.linalg.norm(new_vector - current_vector))

            # Compute confidence (similarity to context centroid)
            confidence = float(np.dot(new_vector, context_centroid) /
                             (np.linalg.norm(new_vector) * np.linalg.norm(context_centroid) + 1e-8))

            # Create state
            state = ThoughtState(
                vector=new_vector.copy(),
                step=step,
                confidence=confidence,
                energy=energy,
                sources=step_sources
            )
            states.append(state)

            # Store context for this step
            context_used.extend([
                {"id": n[0], "score": n[2], "step": step}
                for n in neighbors
            ])

            if verbose:
                logger.info(f"[COCONUT] Step {step}: energy={energy:.4f}, conf={confidence:.3f}, sources={len(step_sources)}")

            # Check convergence
            if energy < self.convergence_threshold:
                if verbose:
                    logger.info(f"[COCONUT] Converged at step {step}!")
                return ThoughtChain(
                    query=query,
                    states=states,
                    final_state=state,
                    converged=True,
                    context_used=context_used
                )

            # Update for next iteration
            previous_direction = direction
            current_vector = new_vector

        # Did not converge - use last state
        final_state = states[-1] if states else ThoughtState(
            vector=query_vector,
            step=0,
            confidence=0.0,
            energy=0.0,
            sources=[]
        )

        return ThoughtChain(
            query=query,
            states=states,
            final_state=final_state,
            converged=False,
            context_used=context_used
        )

    def decode(self, chain: ThoughtChain, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Decode the final thought state to context.

        This is where we "collapse" the continuous thought into
        discrete information that can be used for synthesis.

        Args:
            chain: The completed thought chain
            top_k: How many context items to return

        Returns:
            List of relevant context items with metadata
        """
        if chain.final_state is None:
            return []

        # Search from final position
        neighbors = self.search_fn(chain.final_state.vector, top_k)

        return [
            {
                "id": n[0],
                "score": n[2],
                "reasoning_confidence": chain.final_state.confidence,
                "steps_taken": chain.final_state.step,
                "converged": chain.converged
            }
            for n in neighbors
        ]


# =============================================================================
# Integration with ScornSpine
# =============================================================================

def create_spine_reasoner(spine_url: str = "http://127.0.0.1:7782") -> CoconutReasoner:
    """
    Create a COCONUT reasoner backed by ScornSpine.

    Args:
        spine_url: URL of the ScornSpine server

    Returns:
        Configured CoconutReasoner
    """
    import requests

    def embed_fn(text: str) -> np.ndarray:
        """Embed text via Spine."""
        try:
            resp = requests.post(
                f"{spine_url}/embed",
                json={"text": text},
                timeout=30
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Embed failed: {resp.text}")
            return np.array(resp.json()["embedding"], dtype=np.float32)
        except requests.RequestException as e:
            raise RuntimeError(f"Spine connection failed: {e}") from e

    def search_fn(vector: np.ndarray, k: int) -> List[Tuple[str, np.ndarray, float]]:
        """Search by vector via Spine."""
        try:
            resp = requests.post(
                f"{spine_url}/search_vector",
                json={
                    "vector": vector.tolist(),
                    "k": k
                },
                timeout=30
            )
            if resp.status_code != 200:
                # Fall back to text search if vector search not available
                return []
        except requests.RequestException:
            return []  # Connection failed, return empty results

        results = resp.json().get("results", [])
        return [
            (r["id"], np.array(r.get("vector", [0]*768), dtype=np.float32), r["score"])
            for r in results
        ]

    return CoconutReasoner(
        embed_fn=embed_fn,
        search_fn=search_fn,
        max_steps=5,
        convergence_threshold=0.02,
        momentum=0.3,
        context_blend=0.4
    )


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COCONUT Latent Reasoning")
    parser.add_argument("query", help="The query to reason about")
    parser.add_argument("--spine-url", default="http://127.0.0.1:7782", help="ScornSpine URL")
    parser.add_argument("--max-steps", type=int, default=5, help="Max reasoning steps")
    parser.add_argument("--verbose", action="store_true", help="Show each step")

    args = parser.parse_args()

    # Enable logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=" * 70)
    print("COCONUT - Continuous Thought in Latent Space")
    print("=" * 70)
    print()

    try:
        reasoner = create_spine_reasoner(args.spine_url)
        reasoner.max_steps = args.max_steps

        chain = reasoner.reason(args.query, verbose=args.verbose)

        print(chain.trace())
        print()

        print("=== Decoded Context ===")
        context = reasoner.decode(chain)
        for item in context:
            print(f"  [{item['score']:.3f}] {item['id']}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure ScornSpine is running: python -m haloscorn.scornspine.server")


# =============================================================================
# Text-Based COCONUT (works with existing /search endpoint)
# =============================================================================

class TextCoconutReasoner:
    """
    Text-based approximation of COCONUT reasoning.

    Instead of true vector operations, we:
    1. Search for relevant context
    2. Extract key terms from results
    3. Refine query with extracted terms
    4. Repeat until convergence

    This is a simulation of latent space reasoning using text retrieval.
    Once /embed and /search_vector endpoints are available, use CoconutReasoner.
    """

    def __init__(
        self,
        search_fn: Callable[[str, int], List[Dict[str, Any]]],
        max_steps: int = 3,
        convergence_threshold: float = 0.9
    ):
        """
        Initialize text-based COCONUT reasoner.

        Args:
            search_fn: Function that searches by text query (query, k -> [{text, score, ...}])
            max_steps: Maximum reasoning iterations
            convergence_threshold: Stop when result overlap exceeds this
        """
        self.search_fn = search_fn
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold

    def _extract_keywords(self, results: List[Dict], top_n: int = 5) -> List[str]:
        """Extract key terms from search results."""
        import re
        from collections import Counter

        # Combine all text
        all_text = " ".join(r.get("text", "") for r in results)

        # Simple keyword extraction (no NLP dependencies)
        # Split into words, filter stopwords and short words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "need", "dare", "ought", "used", "to", "of", "in",
                     "for", "on", "with", "at", "by", "from", "as", "into",
                     "through", "during", "before", "after", "above", "below",
                     "between", "under", "again", "further", "then", "once",
                     "here", "there", "when", "where", "why", "how", "all",
                     "each", "few", "more", "most", "other", "some", "such",
                     "no", "nor", "not", "only", "own", "same", "so", "than",
                     "too", "very", "just", "and", "but", "if", "or", "because",
                     "until", "while", "this", "that", "these", "those", "what"}

        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b', all_text.lower())
        words = [w for w in words if w not in stopwords and len(w) > 3]

        # Get most common
        counts = Counter(words)
        return [word for word, _ in counts.most_common(top_n)]

    def _result_overlap(self, results_a: List[Dict], results_b: List[Dict]) -> float:
        """Compute overlap between two result sets."""
        ids_a = {r.get("filepath", r.get("id", i)) for i, r in enumerate(results_a)}
        ids_b = {r.get("filepath", r.get("id", i)) for i, r in enumerate(results_b)}

        if not ids_a or not ids_b:
            return 0.0

        intersection = len(ids_a & ids_b)
        union = len(ids_a | ids_b)
        return intersection / union if union > 0 else 0.0

    def reason(self, query: str, top_k: int = 5, verbose: bool = False) -> Dict[str, Any]:
        """
        Reason iteratively using text search.

        Args:
            query: Initial query
            top_k: Results per step
            verbose: Log each step

        Returns:
            Dict with final results, steps taken, and convergence info
        """
        current_query = query
        all_results = []
        steps = []
        prev_results = []

        for step in range(1, self.max_steps + 1):
            # Search
            results = self.search_fn(current_query, top_k)

            if not results:
                if verbose:
                    logger.info(f"[TextCOCONUT] Step {step}: No results, stopping")
                break

            # Check convergence
            overlap = self._result_overlap(results, prev_results)
            if overlap >= self.convergence_threshold:
                if verbose:
                    logger.info(f"[TextCOCONUT] Step {step}: Converged (overlap={overlap:.2f})")
                steps.append({
                    "step": step,
                    "query": current_query,
                    "result_count": len(results),
                    "overlap": overlap,
                    "converged": True
                })
                all_results = results
                break

            # Extract keywords and refine query
            keywords = self._extract_keywords(results)
            refined_query = f"{query} {' '.join(keywords)}"

            if verbose:
                logger.info(f"[TextCOCONUT] Step {step}: overlap={overlap:.2f}, keywords={keywords}")

            steps.append({
                "step": step,
                "query": current_query,
                "result_count": len(results),
                "overlap": overlap,
                "keywords": keywords,
                "converged": False
            })

            prev_results = results
            all_results = results
            current_query = refined_query

        return {
            "original_query": query,
            "final_query": current_query,
            "results": all_results,
            "steps": steps,
            "converged": len(steps) > 0 and steps[-1].get("converged", False),
            "total_steps": len(steps)
        }


def create_text_spine_reasoner(spine_url: str = "http://127.0.0.1:7782") -> TextCoconutReasoner:
    """
    Create a text-based COCONUT reasoner using existing ScornSpine endpoints.

    This works with the current /search endpoint (no /embed or /search_vector needed).

    Args:
        spine_url: URL of the ScornSpine server

    Returns:
        Configured TextCoconutReasoner
    """
    import requests

    def search_fn(query: str, k: int) -> List[Dict[str, Any]]:
        """Search via Spine's /search endpoint."""
        try:
            resp = requests.post(
                f"{spine_url}/search",
                json={"query": query, "top_k": k},
                timeout=120  # RT4400: Increased for slow indexing periods
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("results", [])
        except requests.RequestException:
            return []  # Spine unavailable, return empty results

    return TextCoconutReasoner(
        search_fn=search_fn,
        max_steps=3,
        convergence_threshold=0.8
    )
