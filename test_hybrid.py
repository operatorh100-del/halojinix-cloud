#!/usr/bin/env python3
"""
RT263: Quick test for hybrid BM25+Vector search.
Run from ChatRTX venv: C:\ChatRTX\venv\Scripts\python.exe test_hybrid.py
"""

import sys
import os

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_hybrid_search():
    """Test hybrid search functionality."""
    from haloscorn.scornspine.spine import ScornSpine, SpineConfig

    # Initialize with hybrid enabled
    config = SpineConfig(enable_hybrid_search=True)
    spine = ScornSpine(config=config)

    print("=" * 60)
    print("RT263: Hybrid BM25+Vector Search Test")
    print("=" * 60)

    # Check BM25 availability
    from haloscorn.scornspine.bm25_index import BM25_AVAILABLE
    print(f"\n? BM25 library available: {BM25_AVAILABLE}")
    print(f"? Hybrid search enabled: {config.enable_hybrid_search}")
    print(f"? Vector weight: {config.hybrid_vector_weight}")
    print(f"? BM25 weight: {config.hybrid_bm25_weight}")

    # Test queries that should benefit from BM25
    test_queries = [
        # Exact match queries (BM25 should excel)
        "ADR-0050 background processes",
        "HALO agent implementation",
        "RT263 ScornSpine",

        # Semantic queries (Vector should excel)
        "how to handle memory persistence",
        "particle rendering optimization",
    ]

    print("\n" + "-" * 60)
    print("Running test queries...")
    print("-" * 60)

    for query in test_queries:
        print(f"\n[NOTE] Query: {query}")

        # Test with hybrid (default)
        result_hybrid = spine.query(query, top_k=3, use_hybrid=True)

        # Test with vector-only
        result_vector = spine.query(query, top_k=3, use_hybrid=False)

        print(f"   Hybrid method: {result_hybrid.get('retrieval_method', 'unknown')}")
        print(f"   Hybrid latency: {result_hybrid.get('latency_ms', 0):.1f}ms")
        print(f"   Vector latency: {result_vector.get('latency_ms', 0):.1f}ms")

        if result_hybrid.get('sources'):
            top_source = result_hybrid['sources'][0]
            print(f"   Top result: {top_source.get('filepath', 'unknown')[:60]}")
            if 'fused_score' in top_source:
                print(f"   Scores - Fused: {top_source['fused_score']:.4f}, "
                      f"Vector: {top_source.get('vector_score', 0):.4f}, "
                      f"BM25: {top_source.get('bm25_score', 0):.4f}")

    print("\n" + "=" * 60)
    print("Hybrid search test complete!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_hybrid_search()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
