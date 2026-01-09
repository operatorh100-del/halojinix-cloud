"""
ScornSpine RunPod Serverless Handler
RT31800: Wraps ScornSpine for RunPod serverless deployment

USAGE:
    Local test: python runpod_handler.py --test-input test_input.json
    Deploy: runpodctl deploy --serverless

HANDLER FORMAT (per RunPod docs):
    Input:  {"input": {"query": "...", "top_k": 5, "agent": "jonah"}}
    Output: {"results": [...], "latency_ms": 42.5, "doc_count": 1234}

CLOUDBORNE ARCHITECTURE (RT31800):
    1. This handler runs on RunPod A100 80GB ($0.79/hr, scale-to-zero)
    2. Qdrant Cloud stores vectors (73c7f78e...europe-west3-0.gcp.cloud.qdrant.io)
    3. Mem0 runs locally or on separate RunPod instance
    4. No local Docker required - pure cloud-native

MONITORING (RT26000):
    - All logs go to stdout/stderr (captured by RunPod dashboard)
    - Structured logging with levels: INFO, WARNING, ERROR
    - Latency tracked per-request in response payload
    - View logs: RunPod Console → Endpoint → Logs tab
"""

import os
import sys
import time
import logging
import numpy  # RT33700: Import BEFORE runpod/torch to prevent version conflict
import runpod

# Structured logging (RunPod best practice)
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("scornspine")

# Add app root to path for imports
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize once per worker (not per request)
_spine = None
_initialized = False


def initialize():
    """One-time initialization per worker.

    RunPod workers persist between requests - initialize heavy objects once.
    This saves 5-10s per request vs reinitializing each time.
    """
    global _spine, _initialized
    if _initialized:
        return

    logger.info("Initializing ScornSpine worker...")
    start = time.time()

    # Import spine - try multiple paths for compatibility
    try:
        from scornspine.spine import ScornSpine
    except ImportError:
        from spine import ScornSpine

    # Configure for Qdrant Cloud (secrets injected via RunPod environment)
    qdrant_url = os.getenv('QDRANT_CLOUD_URL') or os.getenv('QDRANT_URL')
    qdrant_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_key:
        logger.error("Missing QDRANT_CLOUD_URL or QDRANT_API_KEY in environment")
        raise RuntimeError("QDRANT_CLOUD_URL and QDRANT_API_KEY required in environment")

    logger.info(f"Connecting to Qdrant Cloud: {qdrant_url[:50]}...")

    # Initialize ScornSpine with cloud Qdrant backend
    _spine = ScornSpine(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        collection="halojinix-spine"
    )

    elapsed = time.time() - start
    logger.info(f"Worker initialized in {elapsed:.2f}s - Ready for requests")
    _initialized = True


def handler(job):
    """
    RunPod serverless handler.

    Args:
        job (dict): Contains 'input' with query parameters
            - query: Search query string (required)
            - top_k: Number of results (default: 5)
            - agent: Agent ID for tracking (optional)
            - mode: 'query' (default), 'hybrid', 'coconut', 'embed'

    Returns:
        dict: Results with latency tracking
    """
    start = time.time()

    # Ensure initialized
    initialize()

    # Extract input
    job_input = job.get("input", {})

    # Validate required fields
    query = job_input.get("query", "").strip()
    if not query:
        return {"error": "query is required", "success": False}

    # Optional parameters
    top_k = min(int(job_input.get("top_k", 5)), 50)  # Cap at 50
    agent = job_input.get("agent", "anonymous")
    mode = job_input.get("mode", "query")

    try:
        if mode == "query":
            # Standard vector search
            results = _spine.query(query, top_k=top_k)

        elif mode == "hybrid":
            # Hybrid search (vector + BM25 if available)
            try:
                from scornspine.bm25_index import BM25Index, reciprocal_rank_fusion
            except ImportError:
                from bm25_index import BM25Index, reciprocal_rank_fusion

            vector_results = _spine.query(query, top_k=top_k * 2)

            # BM25 fallback (may not be available in cloud)
            bm25 = getattr(_spine, '_bm25', None)
            if bm25 and bm25.is_ready:
                bm25_results = bm25.search(query, top_k=top_k * 2)
                results = reciprocal_rank_fusion(
                    [{"id": r.get("path", str(i)), **r} for i, r in enumerate(vector_results)],
                    bm25_results,
                    vector_weight=0.7,
                    bm25_weight=0.3
                )[:top_k]
            else:
                results = vector_results[:top_k]

        elif mode == "embed":
            # Return raw embedding vector
            vector = _spine.embedder.encode([query])[0]
            elapsed = (time.time() - start) * 1000
            return {
                "success": True,
                "vector": vector.tolist(),
                "dimension": len(vector),
                "latency_ms": round(elapsed, 2)
            }

        elif mode == "coconut":
            # COCONUT latent reasoning (iterative refinement)
            try:
                from haloscorn.latent_space.coconut import TextCoconutReasoner
            except ImportError:
                try:
                    from latent_space.coconut import TextCoconutReasoner
                except ImportError:
                    return {"error": "COCONUT module not available", "success": False}

            max_steps = min(int(job_input.get("max_steps", 3)), 10)

            def search_fn(q, k):
                return _spine.query(q.strip(), top_k=k)

            reasoner = TextCoconutReasoner(
                search_fn=search_fn,
                max_steps=max_steps,
                convergence_threshold=0.8
            )

            result = reasoner.reason(query, top_k=top_k)
            elapsed = (time.time() - start) * 1000

            return {
                "success": True,
                "query": query,
                "final_query": result.get("final_query", query),
                "results": result.get("results", []),
                "converged": result.get("converged", False),
                "steps": result.get("total_steps", 0),
                "latency_ms": round(elapsed, 2)
            }

        else:
            return {"error": f"Unknown mode: {mode}", "success": False}

        elapsed = (time.time() - start) * 1000

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "agent": agent,
            "mode": mode,
            "latency_ms": round(elapsed, 2),
            "doc_count": len(_spine.docs) if hasattr(_spine, 'docs') else 0
        }

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"Handler error: {str(e)} | query={query[:50]} | mode={mode}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "latency_ms": round(elapsed, 2)
        }


# RunPod entrypoint
if __name__ == "__main__":
    # Check for local test mode
    if "--test-input" in sys.argv:
        import json
        idx = sys.argv.index("--test-input")
        test_file = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "test_input.json"

        logger.info(f"Local test mode with {test_file}")

        with open(test_file) as f:
            test_input = json.load(f)

        result = handler({"input": test_input})
        print(json.dumps(result, indent=2))
    else:
        # Production mode - start RunPod worker
        logger.info("Starting ScornSpine serverless worker...")
        runpod.serverless.start({"handler": handler})
