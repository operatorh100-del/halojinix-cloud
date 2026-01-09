"""
Test suite for qdrant_fallback.py - RT31701
Run: python -m pytest haloscorn/scornspine/test_fallback.py -v
"""

import sys
sys.path.insert(0, "F:/primewave-engine")

from haloscorn.scornspine.qdrant_fallback import (
    qdrant_fallback,
    needs_fallback,
    merge_results,
    MIN_SCORE_THRESHOLD,
    MIN_RESULTS_THRESHOLD
)


def test_qdrant_fallback_available():
    """Test that QdrantFallback is properly initialized."""
    assert qdrant_fallback is not None
    # Should be available if .env has credentials
    print(f"  QdrantFallback available: {qdrant_fallback.is_available}")


def test_needs_fallback_empty():
    """Empty results should trigger fallback."""
    assert needs_fallback([]) == True
    print("  ✓ Empty results triggers fallback")


def test_needs_fallback_low_score():
    """Low scores should trigger fallback."""
    results = [{"score": 0.3, "path": "test.md"}]
    assert needs_fallback(results) == True
    print("  ✓ Low score (0.3) triggers fallback")


def test_needs_fallback_few_results():
    """Too few results should trigger fallback even with good scores."""
    results = [
        {"score": 0.8, "path": "a.md"},
        {"score": 0.7, "path": "b.md"}
    ]
    assert needs_fallback(results) == True
    print("  ✓ Few results (2) triggers fallback")


def test_needs_fallback_good_results():
    """Good scores with enough results should NOT trigger fallback."""
    results = [
        {"score": 0.8, "path": "a.md"},
        {"score": 0.7, "path": "b.md"},
        {"score": 0.6, "path": "c.md"}
    ]
    assert needs_fallback(results) == False
    print("  ✓ Good results (3 with high scores) does NOT trigger fallback")


def test_merge_results_dedup():
    """Merge should deduplicate by path."""
    local = [
        {"path": "a.md", "score": 0.7, "text": "local"},
        {"path": "b.md", "score": 0.5, "text": "local"}
    ]
    cloud = [
        {"path": "c.md", "score": 0.8, "text": "cloud", "source": "qdrant-cloud"},
        {"path": "a.md", "score": 0.75, "text": "cloud-dupe", "source": "qdrant-cloud"}  # Duplicate
    ]

    merged = merge_results(local, cloud, top_k=10)
    paths = [r["path"] for r in merged]

    assert len(merged) == 3, f"Expected 3 results, got {len(merged)}"
    assert paths.count("a.md") == 1, "a.md should appear only once"
    print("  ✓ Deduplication works")


def test_merge_results_sorting():
    """Merged results should be sorted by score descending."""
    local = [{"path": "low.md", "score": 0.3}]
    cloud = [{"path": "high.md", "score": 0.9, "source": "qdrant-cloud"}]

    merged = merge_results(local, cloud, top_k=10)

    assert merged[0]["path"] == "high.md", "Highest score should be first"
    assert merged[0]["score"] > merged[1]["score"], "Results should be sorted"
    print("  ✓ Sorting by score works")


def test_merge_results_top_k():
    """Merge should respect top_k limit."""
    local = [{"path": f"l{i}.md", "score": 0.5} for i in range(10)]
    cloud = [{"path": f"c{i}.md", "score": 0.5, "source": "qdrant-cloud"} for i in range(10)]

    merged = merge_results(local, cloud, top_k=5)

    assert len(merged) == 5, f"Expected 5 results, got {len(merged)}"
    print("  ✓ top_k limit works")


def test_merge_results_source_marking():
    """Local results should get source='local-faiss' if not already set."""
    local = [{"path": "local.md", "score": 0.5}]
    cloud = []

    merged = merge_results(local, cloud, top_k=10)

    assert merged[0].get("source") == "local-faiss"
    print("  ✓ Source marking works")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RT31701 TEST SUITE: qdrant_fallback.py")
    print("="*60 + "\n")

    tests = [
        test_qdrant_fallback_available,
        test_needs_fallback_empty,
        test_needs_fallback_low_score,
        test_needs_fallback_few_results,
        test_needs_fallback_good_results,
        test_merge_results_dedup,
        test_merge_results_sorting,
        test_merge_results_top_k,
        test_merge_results_source_marking,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: ERROR - {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {failed} TESTS FAILED!")
        sys.exit(1)
