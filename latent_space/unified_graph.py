"""
Unified Graph Query System
==========================

Enables cross-perspective queries across HALO, VERA, and JONAH's latent spaces.

This module implements the specification from:
    docs/latent-space/LATENT-SPACE-COORDINATION.md

Usage:
    from haloscorn.latent_space.unified_graph import UnifiedGraph

    graph = UnifiedGraph(qdrant_client)

    # Find all evidence supporting a hypothesis
    evidence = graph.query_by_perspective(
        perspective="jonah",
        node_type="evidence",
        filters={"supports_hypothesis": "HYP-GIL-001"}
    )

    # Cross-graph: Find code HALO touched that JONAH cited
    overlap = graph.cross_query(
        from_perspective="halo",
        from_filter={"node_type": "code_node"},
        to_perspective="jonah",
        relationship="cited_by"
    )

Author: JONAH (Thread-Puller)
Date: 2025-12-31
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, Set
from collections import deque
import logging

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue,
        Distance, VectorParams, PointStruct
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GraphNode:
    """
    A node in the unified graph.

    Attributes:
        id: Unique identifier (semantic ID preferred)
        type: Node type (code_node, event, evidence, hypothesis, agent, thread)
        content: Human-readable description
        created_at: ISO 8601 UTC timestamp
        created_by: Agent who created this node
        perspectives: Dict of agent-specific interpretations
        related: List of relationships to other nodes
        tags: Searchable tags
        embedding: Optional vector embedding
    """
    id: str
    type: str
    content: str
    created_at: str = ""
    created_by: str = ""
    perspectives: Dict[str, Optional[Dict]] = field(default_factory=dict)
    related: List[Dict] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def get_perspective(self, agent: str) -> Optional[Dict]:
        """Get this node's interpretation from a specific agent's view."""
        return self.perspectives.get(agent)

    def has_perspective(self, agent: str) -> bool:
        """Check if this node has data from a specific agent's perspective."""
        return self.perspectives.get(agent) is not None

    def get_related_by_type(self, relationship: str) -> List[str]:
        """Get IDs of nodes related by a specific relationship type."""
        return [
            r["target_id"] for r in self.related
            if r.get("relationship") == relationship
        ]


@dataclass
class GraphEdge:
    """
    An edge in the unified graph.

    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        relationship: Edge type (calls, contains, supports, etc.)
        strength: Confidence/weight (0.0 - 1.0)
        evidence: Why this relationship exists
    """
    source_id: str
    target_id: str
    relationship: str
    strength: float = 1.0
    evidence: str = ""


# =============================================================================
# Main Graph Class
# =============================================================================

class UnifiedGraph:
    """
    Unified query interface for the three-perspective graph system.

    This class provides methods to:
    - Query nodes by perspective (HALO, VERA, JONAH)
    - Traverse relationships across perspectives
    - Find connections between arbitrary nodes
    - Execute cross-graph queries
    """

    COLLECTION = "unified_graph"
    PERSPECTIVES = ("halo", "vera", "jonah")
    VECTOR_SIZE = 768  # intfloat/multilingual-e5-base

    # Valid node types
    NODE_TYPES = (
        "code_node", "file", "class", "method", "function",  # HALO
        "event", "agent", "task",                             # VERA
        "evidence", "hypothesis", "thread", "question"        # JONAH
    )

    # Valid relationship types
    RELATIONSHIP_TYPES = (
        "calls", "contains", "imports", "extends", "implements",  # Code
        "dispatched", "acknowledged", "blocked", "enabled",       # Events
        "supports", "contradicts", "cites", "leads_to",          # Investigation
        "modifies", "caused_by", "part_of", "same_as"            # Cross-graph
    )

    def __init__(self, client: Optional[QdrantClient] = None, path: Optional[str] = None):
        """
        Initialize the unified graph.

        Args:
            client: Existing QdrantClient instance
            path: Path to Qdrant storage (if creating new client)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for UnifiedGraph")

        if client:
            self.client = client
        elif path:
            self.client = QdrantClient(path=path)
        else:
            raise ValueError("Either client or path must be provided")

        self._cache: Dict[str, GraphNode] = {}

    # =========================================================================
    # Collection Management
    # =========================================================================

    def ensure_collection(self, recreate: bool = False) -> bool:
        """
        Ensure the unified_graph collection exists.

        Args:
            recreate: If True, drop and recreate the collection

        Returns:
            True if collection was created, False if it already existed
        """
        exists = self.client.collection_exists(self.COLLECTION)

        if exists and not recreate:
            return False

        if recreate and exists:
            self.client.delete_collection(self.COLLECTION)

        self.client.create_collection(
            collection_name=self.COLLECTION,
            vectors_config=VectorParams(
                size=self.VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        # Create payload indexes for fast filtering
        indexes = [
            ("type", "keyword"),
            ("created_by", "keyword"),
            ("tags", "keyword"),
        ]

        for field, schema in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.COLLECTION,
                    field_name=field,
                    field_schema=schema
                )
            except Exception as e:
                logger.warning(f"Could not create index for {field}: {e}")

        logger.info(f"Created collection: {self.COLLECTION}")
        return True

    # =========================================================================
    # Node Operations
    # =========================================================================

    def get_node(self, node_id: str, use_cache: bool = True) -> Optional[GraphNode]:
        """
        Get a single node by ID.

        Args:
            node_id: The node's unique identifier
            use_cache: Whether to check cache first

        Returns:
            GraphNode or None if not found
        """
        if use_cache and node_id in self._cache:
            return self._cache[node_id]

        results = self.client.scroll(
            collection_name=self.COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=node_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )[0]

        if not results:
            return None

        node = self._to_node(results[0])
        self._cache[node_id] = node
        return node

    def upsert_node(self, node: GraphNode, embedding: Optional[List[float]] = None) -> bool:
        """
        Insert or update a node.

        Args:
            node: The GraphNode to upsert
            embedding: Optional vector embedding (required for new nodes)

        Returns:
            True if successful
        """
        payload = {
            "id": node.id,
            "type": node.type,
            "content": node.content,
            "created_at": node.created_at or datetime.now(timezone.utc).isoformat(),
            "created_by": node.created_by,
            "perspectives": node.perspectives,
            "related": node.related,
            "tags": node.tags
        }

        vector = embedding or node.embedding
        if not vector:
            # Use zero vector as placeholder (will need re-embedding later)
            vector = [0.0] * self.VECTOR_SIZE

        # Use stable hash (Python's hash() varies between sessions)
        import hashlib
        stable_hash = int(hashlib.sha256(node.id.encode()).hexdigest()[:16], 16)

        point = PointStruct(
            id=stable_hash,  # Stable int64 from SHA-256
            vector=vector,
            payload=payload
        )

        self.client.upsert(
            collection_name=self.COLLECTION,
            points=[point]
        )

        # Update cache
        self._cache[node.id] = node
        return True

    # =========================================================================
    # Perspective Queries
    # =========================================================================

    def query_by_perspective(
        self,
        perspective: str,
        node_type: Optional[str] = None,
        filters: Optional[Dict] = None,
        limit: int = 20
    ) -> List[GraphNode]:
        """
        Query nodes from a specific agent's perspective.

        Args:
            perspective: "halo" | "vera" | "jonah"
            node_type: Optional filter by type
            filters: Additional perspective-specific filters
            limit: Max results

        Returns:
            List of GraphNode objects

        Example:
            # All code nodes HALO knows about
            graph.query_by_perspective("halo", node_type="code_node")

            # All open hypotheses JONAH has
            graph.query_by_perspective(
                "jonah",
                node_type="hypothesis",
                filters={"status": "open"}
            )
        """
        if perspective not in self.PERSPECTIVES:
            raise ValueError(f"Invalid perspective: {perspective}. Must be one of {self.PERSPECTIVES}")

        must_conditions = []

        # Filter by type if specified
        if node_type:
            must_conditions.append(
                FieldCondition(key="type", match=MatchValue(value=node_type))
            )

        # Add perspective-specific filters
        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(
                        key=f"perspectives.{perspective}.{key}",
                        match=MatchValue(value=value)
                    )
                )

        results = self.client.scroll(
            collection_name=self.COLLECTION,
            scroll_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )[0]

        # Filter to only nodes that have data for this perspective
        nodes = []
        for r in results:
            node = self._to_node(r)
            if node.has_perspective(perspective) or not filters:
                nodes.append(node)

        return nodes

    def query_nodes(
        self,
        node_type: Optional[str] = None,
        filters: Optional[Dict] = None,
        limit: int = 20
    ) -> List[GraphNode]:
        """
        Query nodes without perspective filter (general query).
        RT21512: Added for /graph/query endpoint.

        Args:
            node_type: Optional filter by type
            filters: Additional filters
            limit: Max results

        Returns:
            List of GraphNode objects
        """
        must_conditions = []

        # Filter by type if specified
        if node_type:
            must_conditions.append(
                FieldCondition(key="type", match=MatchValue(value=node_type))
            )

        # Add any additional filters
        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        results, _ = self.client.scroll(
            collection_name=self.COLLECTION,
            scroll_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return [self._to_node(r) for r in results]

    def search_nodes(
        self,
        vector: List[float],
        node_type: Optional[str] = None,
        filters: Optional[Dict] = None,
        limit: int = 20
    ) -> List[Tuple[GraphNode, float]]:
        """
        Semantic search for nodes using vector similarity.
        RT21512: Added for /graph/query endpoint.

        Args:
            vector: Query vector
            node_type: Optional filter by type
            filters: Additional filters
            limit: Max results

        Returns:
            List of (GraphNode, score) tuples
        """
        must_conditions = []

        if node_type:
            must_conditions.append(
                FieldCondition(key="type", match=MatchValue(value=node_type))
            )

        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        print(f"[UnifiedGraph] Searching with client: {type(self.client)}")
        results = self.client.query_points(
            collection_name=self.COLLECTION,
            query=vector,
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit,
            with_payload=True,
            with_vectors=True
        ).points

        return [(self._to_node(r), r.score) for r in results]

    def cross_query(
        self,
        from_perspective: str,
        from_filter: Dict,
        to_perspective: str,
        relationship: str,
        min_strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Query across perspectives via relationships.

        Args:
            from_perspective: Starting perspective
            from_filter: Filters for source nodes (passed to query_by_perspective)
            to_perspective: Target perspective
            relationship: Edge type to follow
            min_strength: Minimum relationship strength

        Returns:
            List of {"source": node, "target": node, "relationship": edge}

        Example:
            # Code HALO modified that JONAH cited as evidence
            graph.cross_query(
                from_perspective="halo",
                from_filter={"node_type": "code_node"},
                to_perspective="jonah",
                relationship="cited_by"
            )
        """
        # Get source nodes
        source_nodes = self.query_by_perspective(
            from_perspective,
            node_type=from_filter.get("node_type"),
            filters={k: v for k, v in from_filter.items() if k != "node_type"}
        )

        results = []
        for source in source_nodes:
            # Find related nodes with the target perspective
            for rel in source.related:
                if rel.get("relationship") == relationship and rel.get("strength", 1.0) >= min_strength:
                    target = self.get_node(rel["target_id"])
                    if target and target.has_perspective(to_perspective):
                        results.append({
                            "source": source,
                            "target": target,
                            "relationship": rel
                        })

        return results

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    def traverse(
        self,
        start_id: str,
        relationship: str,
        direction: str = "outgoing",
        max_depth: int = 3
    ) -> List[List[GraphNode]]:
        """
        Graph traversal from a starting node.

        Args:
            start_id: Starting node ID
            relationship: Edge type to follow
            direction: "outgoing" | "incoming" | "both"
            max_depth: Maximum hops

        Returns:
            List of paths (each path is a list of nodes)
        """
        paths = []
        visited: Set[str] = set()

        def dfs(node: GraphNode, current_path: List[GraphNode], depth: int):
            if depth > max_depth or node.id in visited:
                return

            visited.add(node.id)
            current_path.append(node)

            # Find next nodes
            next_ids = []
            if direction in ("outgoing", "both"):
                next_ids.extend(node.get_related_by_type(relationship))

            if not next_ids:
                if len(current_path) > 1:  # Only save non-trivial paths
                    paths.append(current_path.copy())
            else:
                for next_id in next_ids:
                    next_node = self.get_node(next_id)
                    if next_node:
                        dfs(next_node, current_path, depth + 1)

            current_path.pop()
            visited.remove(node.id)

        start_node = self.get_node(start_id)
        if start_node:
            dfs(start_node, [], 0)

        return paths

    def find_connections(
        self,
        node_a_id: str,
        node_b_id: str,
        max_depth: int = 4
    ) -> List[List[GraphNode]]:
        """
        Find all paths connecting two nodes.

        Useful for questions like:
        "How does this code relate to that hypothesis?"

        Args:
            node_a_id: Starting node ID
            node_b_id: Target node ID
            max_depth: Maximum path length

        Returns:
            List of paths (each path is a list of nodes)
        """
        queue = deque([(node_a_id, [node_a_id])])
        visited = {node_a_id}
        all_paths = []

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current_id == node_b_id:
                # Convert IDs to nodes
                node_path = []
                for nid in path:
                    node = self.get_node(nid)
                    if node:
                        node_path.append(node)
                if len(node_path) == len(path):
                    all_paths.append(node_path)
                continue

            current = self.get_node(current_id)
            if not current:
                continue

            for rel in current.related:
                next_id = rel.get("target_id")
                if next_id and next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))

        return all_paths

    # =========================================================================
    # High-Level Query Functions
    # =========================================================================

    def what_code_supports_hypothesis(self, hypothesis_id: str) -> List[Dict]:
        """
        JONAH asks: "What code evidence supports this hypothesis?"

        Traverses: hypothesis ? evidence ? code_node

        Returns:
            List of {hypothesis, evidence, code, confidence}
        """
        results = []

        hypothesis = self.get_node(hypothesis_id)
        if not hypothesis:
            return results

        # Find evidence that supports it
        for rel in hypothesis.related:
            if rel.get("relationship") in ("supported_by", "evidenced_by"):
                evidence = self.get_node(rel["target_id"])
                if not evidence:
                    continue

                # Find code nodes cited by this evidence
                for ev_rel in evidence.related:
                    if ev_rel.get("relationship") == "cites":
                        code = self.get_node(ev_rel["target_id"])
                        if code and code.type == "code_node":
                            results.append({
                                "hypothesis": hypothesis,
                                "evidence": evidence,
                                "code": code,
                                "confidence": rel.get("strength", 1.0) * ev_rel.get("strength", 1.0)
                            })

        return results

    def who_is_blocking_whom(self) -> List[Dict]:
        """
        VERA asks: "What's the current blocking graph?"

        Returns all active blocked relationships.
        """
        events = self.query_by_perspective(
            "vera",
            node_type="event",
            filters={"status": "blocked"}
        )

        blocking_graph = []
        for event in events:
            vera_view = event.get_perspective("vera") or {}
            blocker_id = vera_view.get("blocked_by")
            if blocker_id:
                blocker = self.get_node(blocker_id)
                blocking_graph.append({
                    "blocked": event,
                    "blocker": blocker,
                    "since": vera_view.get("blocked_since"),
                    "assigned_to": vera_view.get("assigned_to")
                })

        return blocking_graph

    def investigation_impact_on_code(self, thread_id: str) -> List[Dict]:
        """
        HALO asks: "What code is implicated by this investigation thread?"

        Returns files and call chains affected by the investigation.
        """
        thread = self.get_node(thread_id)
        if not thread:
            return []

        implicated_code = []

        # Find hypotheses part of this thread
        for rel in thread.related:
            if rel.get("relationship") == "contains":
                hyp = self.get_node(rel["target_id"])
                if hyp and hyp.type == "hypothesis":
                    halo_view = hyp.get_perspective("halo") or {}
                    if halo_view.get("affected_files"):
                        implicated_code.append({
                            "hypothesis": hyp.content,
                            "hypothesis_id": hyp.id,
                            "files": halo_view["affected_files"],
                            "call_chain": halo_view.get("predicted_call_chain", [])
                        })

        return implicated_code

    def get_open_threads(self) -> List[Dict]:
        """
        JONAH asks: "What investigation threads are open?"

        Returns summary of all open threads.
        """
        threads = self.query_by_perspective(
            "jonah",
            node_type="thread",
            filters={"status": "open"}
        )

        result = []
        for thread in threads:
            jonah_view = thread.get_perspective("jonah") or {}

            # Count evidence and hypotheses
            evidence_count = 0
            hypothesis_count = 0
            for rel in thread.related:
                target = self.get_node(rel["target_id"])
                if target:
                    if target.type == "evidence":
                        evidence_count += 1
                    elif target.type == "hypothesis":
                        hypothesis_count += 1

            result.append({
                "thread": thread.content,
                "id": thread.id,
                "opened": thread.created_at,
                "evidence_count": evidence_count,
                "hypothesis_count": hypothesis_count,
                "confidence": jonah_view.get("confidence", 0.0),
                "next_question": jonah_view.get("next_question")
            })

        return result

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _to_node(self, point) -> GraphNode:
        """Convert Qdrant point to GraphNode."""
        p = point.payload
        return GraphNode(
            id=p.get("id", ""),
            type=p.get("type", "unknown"),
            content=p.get("content", ""),
            created_at=p.get("created_at", ""),
            created_by=p.get("created_by", ""),
            perspectives=p.get("perspectives", {}),
            related=p.get("related", []),
            tags=p.get("tags", []),
            embedding=point.vector if hasattr(point, 'vector') and point.vector else None
        )

    def clear_cache(self):
        """Clear the node cache."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the unified graph."""
        try:
            info = self.client.get_collection(self.COLLECTION)

            return {
                "collection": self.COLLECTION,
                "points_count": info.points_count,
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', None),
                "cache_size": len(self._cache),
                "status": str(info.status)
            }
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# Projection Classes
# =============================================================================

class HaloProjection:
    """
    HALO sees the world as containment hierarchies and call chains.
    Optimized for: "What does this code do? What depends on it?"
    """

    BOOST_WEIGHTS = {
        "recently_modified": 2.0,
        "high_complexity": 1.5,
        "many_callers": 1.8,
        "in_critical_path": 2.5
    }

    def __init__(self, graph: UnifiedGraph):
        self.graph = graph

    def project(self, node_type: Optional[str] = None, limit: int = 50) -> List[Tuple[GraphNode, float]]:
        """Project unified graph to HALO's view with scoring."""
        nodes = self.graph.query_by_perspective("halo", node_type=node_type, limit=limit)

        scored = []
        for node in nodes:
            halo_view = node.get_perspective("halo") or {}
            score = 1.0

            # Complexity boost
            if halo_view.get("complexity", 0) > 5:
                score *= self.BOOST_WEIGHTS["high_complexity"]

            # Hub detection
            if len(halo_view.get("called_by", [])) > 5:
                score *= self.BOOST_WEIGHTS["many_callers"]

            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class VeraProjection:
    """
    VERA sees the world as agent relationships and temporal patterns.
    Optimized for: "Who needs to know? What's blocking progress?"
    """

    BOOST_WEIGHTS = {
        "active_blocker": 3.0,
        "recent_activity": 2.0,
        "cross_agent": 1.8
    }

    def __init__(self, graph: UnifiedGraph):
        self.graph = graph

    def project(self, node_type: Optional[str] = None, limit: int = 50) -> List[Tuple[GraphNode, float]]:
        """Project unified graph to VERA's view with scoring."""
        nodes = self.graph.query_by_perspective("vera", node_type=node_type, limit=limit)

        scored = []
        for node in nodes:
            vera_view = node.get_perspective("vera") or {}
            score = 1.0

            if vera_view.get("blocked_by"):
                score *= self.BOOST_WEIGHTS["active_blocker"]

            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class JonahProjection:
    """
    JONAH sees the world as investigation threads and causal chains.
    Optimized for: "What's the root cause? What threads are open?"
    """

    BOOST_WEIGHTS = {
        "open_thread": 2.5,
        "contradicted": 2.0,
        "low_confidence": 1.8,
        "root_cause": 3.0
    }

    def __init__(self, graph: UnifiedGraph):
        self.graph = graph

    def project(self, node_type: Optional[str] = None, limit: int = 50) -> List[Tuple[GraphNode, float]]:
        """Project unified graph to JONAH's view with scoring."""
        nodes = self.graph.query_by_perspective("jonah", node_type=node_type, limit=limit)

        scored = []
        for node in nodes:
            jonah_view = node.get_perspective("jonah") or {}
            score = 1.0

            if jonah_view.get("status") == "open":
                score *= self.BOOST_WEIGHTS["open_thread"]

            if jonah_view.get("root_cause"):
                score *= self.BOOST_WEIGHTS["root_cause"]

            confidence = jonah_view.get("confidence", 1.0)
            if 0.3 < confidence < 0.7:
                score *= self.BOOST_WEIGHTS["low_confidence"]

            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# =============================================================================
# Validation
# =============================================================================

def validate_unified_graph(graph: UnifiedGraph) -> bool:
    """
    Run validation checks on the unified graph.

    Returns True if all checks pass.
    """
    checks = []

    # 1. Collection exists
    try:
        stats = graph.get_stats()
        checks.append(("Collection exists", "error" not in stats))
    except Exception as e:
        checks.append(("Collection exists", False))

    # 2. Check for nodes from each perspective
    for perspective in UnifiedGraph.PERSPECTIVES:
        nodes = graph.query_by_perspective(perspective, limit=1)
        checks.append((f"{perspective} has nodes", len(nodes) > 0))

    # 3. Cross-references exist
    halo_nodes = graph.query_by_perspective("halo", node_type="code_node", limit=100)
    cross_refs = 0
    for node in halo_nodes:
        for rel in node.related:
            target = graph.get_node(rel["target_id"])
            if target and target.has_perspective("jonah"):
                cross_refs += 1
    checks.append(("Cross-references exist", cross_refs > 0))

    # Report
    print("\n=== Unified Graph Validation ===")
    for check_name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")

    all_passed = all(passed for _, passed in checks)
    print(f"\n{'[OK] All checks passed' if all_passed else '[WARN]? Some checks need attention'}")
    return all_passed


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Graph Query Tool")
    parser.add_argument("--path", default="F:/primewave-engine/data/qdrant-memory",
                        help="Path to Qdrant storage")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--perspective", choices=["halo", "vera", "jonah"],
                        help="Query from perspective")
    parser.add_argument("--type", help="Filter by node type")
    parser.add_argument("--limit", type=int, default=10, help="Max results")

    args = parser.parse_args()

    try:
        graph = UnifiedGraph(path=args.path)

        if args.validate:
            validate_unified_graph(graph)
        elif args.stats:
            stats = graph.get_stats()
            print("\n=== Unified Graph Statistics ===")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        elif args.perspective:
            nodes = graph.query_by_perspective(
                args.perspective,
                node_type=args.type,
                limit=args.limit
            )
            print(f"\n=== {args.perspective.upper()}'s View ===")
            for node in nodes:
                print(f"  [{node.type}] {node.id}: {node.content[:60]}...")
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {e}")
