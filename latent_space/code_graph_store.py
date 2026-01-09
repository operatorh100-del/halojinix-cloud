"""
HALO Latent Space - Code Graph Storage

Stores the indexed code graph in Qdrant:
- Each code node gets a vector embedding of its code + docstring
- Graph structure stored in payloads (contains, calls, etc.)
- Enables hybrid query: semantic + structural

This is Phase 1: Qdrant-only, graph structure in payloads.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny
)

from .code_indexer import CodeNode, CodeEdge, CodeIndexer

logger = logging.getLogger(__name__)


class CodeGraphStore:
    """
    Stores and queries the code graph using Qdrant.

    Usage:
        store = CodeGraphStore(qdrant_client, embed_model)
        store.index_directory("engine/", "F:/primewave-engine")

        # Semantic search: "GPU dispatch"
        results = store.search_semantic("GPU dispatch", limit=10)

        # Structural query: What methods does GpuContext contain?
        results = store.query_contains("engine/gfx/gpu-context.ts::GpuContext")

        # Call chain: What calls schedule()?
        results = store.query_callers("schedule")
    """

    COLLECTION_NAME = "halo-code-graph"
    VECTOR_SIZE = 768  # multilingual-e5-base

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embed_model: Any,  # HuggingFaceEmbedding
    ):
        """
        Initialize code graph store.

        Args:
            qdrant_client: Connected Qdrant client
            embed_model: Embedding model for vectorization
        """
        self.qdrant = qdrant_client
        self.embed_model = embed_model
        self.indexer = CodeIndexer()

        # Ensure collection exists
        self._ensure_collection()

        logger.info("CodeGraphStore initialized")

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = [c.name for c in self.qdrant.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                self.qdrant.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.COLLECTION_NAME}")
            else:
                logger.info(f"Collection exists: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.embed_model.get_text_embedding(text)

    def _build_embed_text(self, node: CodeNode, edges: List[CodeEdge]) -> str:
        """
        Build the text to embed for a code node.

        Includes:
        - Name and type
        - Signature (for functions/methods)
        - Docstring
        - First 500 chars of code
        - Call relationships
        """
        parts = [f"{node.type}: {node.name}"]

        if node.signature:
            parts.append(f"signature: {node.signature}")

        if node.docstring:
            # Clean up docstring
            doc = node.docstring.strip()
            if doc.startswith('/**'):
                doc = doc[3:]
            if doc.endswith('*/'):
                doc = doc[:-2]
            parts.append(doc.strip())

        # Add abbreviated code
        code = node.code[:500] if len(node.code) > 500 else node.code
        parts.append(f"code: {code}")

        # Add call relationships for context
        calls = [e.to_node for e in edges if e.from_node == node.id and e.type == "CALLS"]
        if calls:
            parts.append(f"calls: {', '.join(calls[:10])}")  # Limit to 10

        return '\n'.join(parts)

    def index_directory(
        self,
        dir_path: str,
        base_path: str,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Index all files in a directory and store in Qdrant.

        Args:
            dir_path: Directory to index
            base_path: Base path for relative paths
            batch_size: Number of points to upsert at once

        Returns:
            Statistics dict with node_count, edge_count, etc.
        """
        logger.info(f"Indexing directory: {dir_path}")

        # Index with tree-sitter
        nodes, edges = self.indexer.index_directory(dir_path, base_path)

        # Build edge lookup
        edges_by_node: Dict[str, List[CodeEdge]] = {}
        for edge in edges:
            if edge.from_node not in edges_by_node:
                edges_by_node[edge.from_node] = []
            edges_by_node[edge.from_node].append(edge)

        # Prepare points for Qdrant
        points = []
        skip_count = 0

        for node in nodes:
            # Skip file nodes (too generic)
            if node.type == "file":
                skip_count += 1
                continue

            # Build embedding text
            node_edges = edges_by_node.get(node.id, [])
            embed_text = self._build_embed_text(node, node_edges)

            try:
                embedding = self._embed(embed_text)
            except Exception as e:
                logger.warning(f"Failed to embed {node.id}: {e}")
                skip_count += 1
                continue

            # Build payload with graph structure
            calls = [e.to_node for e in node_edges if e.type == "CALLS"]
            imports = [e.to_node for e in node_edges if e.type == "IMPORTS"]
            contains = [e.to_node for e in node_edges if e.type == "CONTAINS"]

            # Find who calls this node
            called_by = [
                e.from_node for e in edges
                if e.to_node == node.name or e.to_node.endswith(f"::{node.name}")
                and e.type == "CALLS"
            ]

            payload = {
                "id": node.id,
                "type": node.type,
                "name": node.name,
                "path": node.path,
                "language": node.language,
                "line_start": node.line_start,
                "line_end": node.line_end,
                "signature": node.signature,
                "docstring": node.docstring[:500] if node.docstring else None,
                "contained_by": node.contained_by,
                # Graph structure
                "calls": calls[:50],  # Limit to prevent huge payloads
                "called_by": called_by[:50],
                "contains": contains[:50],
                "imports": imports[:20],
                # Metadata
                "embed_text_length": len(embed_text),
                "indexed_at": str(Path(dir_path).stat().st_mtime),
            }

            point = PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=payload
            )
            points.append(point)

            # Batch upsert
            if len(points) >= batch_size:
                self.qdrant.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=points
                )
                logger.info(f"Upserted batch of {len(points)} points")
                points = []

        # Final batch
        if points:
            self.qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )
            logger.info(f"Upserted final batch of {len(points)} points")

        stats = {
            "nodes_indexed": len(nodes) - skip_count,
            "nodes_skipped": skip_count,
            "edges_found": len(edges),
            "collection": self.COLLECTION_NAME
        }

        logger.info(f"Indexing complete: {stats}")
        return stats

    def search_semantic(
        self,
        query: str,
        limit: int = 10,
        type_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic search for code elements.

        Args:
            query: Natural language query
            limit: Max results
            type_filter: Filter by node type ("function", "class", etc.)

        Returns:
            List of matching nodes with scores
        """
        query_vector = self._embed(query)

        filter_cond = None
        if type_filter:
            filter_cond = Filter(
                must=[FieldCondition(key="type", match=MatchValue(value=type_filter))]
            )

        results = self.qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            query_filter=filter_cond
        )

        return [
            {
                "id": r.payload.get("id"),
                "name": r.payload.get("name"),
                "type": r.payload.get("type"),
                "path": r.payload.get("path"),
                "signature": r.payload.get("signature"),
                "score": r.score,
                "lines": f"{r.payload.get('line_start')}-{r.payload.get('line_end')}",
                "calls": r.payload.get("calls", []),
                "called_by": r.payload.get("called_by", []),
            }
            for r in results.points
        ]

    def query_contains(self, parent_id: str) -> List[Dict]:
        """
        Query what a node contains (structural query).

        Args:
            parent_id: Node ID like "engine/gfx/gpu-context.ts::GpuContext"

        Returns:
            List of child nodes
        """
        results = self.qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="contained_by", match=MatchValue(value=parent_id))]
            ),
            limit=100
        )

        return [
            {
                "id": r.payload.get("id"),
                "name": r.payload.get("name"),
                "type": r.payload.get("type"),
                "signature": r.payload.get("signature"),
            }
            for r in results[0]  # scroll returns (points, next_page_offset)
        ]

    def query_callers(self, function_name: str) -> List[Dict]:
        """
        Find all functions that call a given function.

        Args:
            function_name: Function name (e.g., "dispatch", "schedule")

        Returns:
            List of caller nodes
        """
        # Search for nodes that have this function in their "calls" array
        # This is tricky with Qdrant's filter model - we use MatchAny
        results = self.qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="calls", match=MatchAny(any=[function_name]))]
            ),
            limit=100
        )

        return [
            {
                "id": r.payload.get("id"),
                "name": r.payload.get("name"),
                "type": r.payload.get("type"),
                "path": r.payload.get("path"),
                "signature": r.payload.get("signature"),
            }
            for r in results[0]
        ]

    def query_callees(self, function_id: str) -> List[str]:
        """
        Find all functions called by a given function.

        Args:
            function_id: Full node ID

        Returns:
            List of called function names
        """
        results = self.qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=function_id))]
            ),
            limit=1
        )

        if results[0]:
            return results[0][0].payload.get("calls", [])
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.qdrant.get_collection(self.COLLECTION_NAME)
        return {
            "collection": self.COLLECTION_NAME,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value
        }


def main():
    """Test the code graph store."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from qdrant_client import QdrantClient

    print("=" * 70)
    print("HALO Latent Space - Code Graph Store Test")
    print("=" * 70)

    # Connect to Qdrant
    qdrant = QdrantClient(
        url="https://abc123.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key="..."  # Would need real key
    )

    # Load embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base",
        device="cuda"
    )

    # Create store
    store = CodeGraphStore(qdrant, embed_model)

    # Index engine
    stats = store.index_directory(
        "F:/primewave-engine/engine",
        "F:/primewave-engine"
    )
    print(f"\nIndexing stats: {stats}")

    # Test queries
    print("\n--- Semantic Search: 'GPU dispatch' ---")
    results = store.search_semantic("GPU dispatch", limit=5)
    for r in results:
        print(f"  [{r['type']}] {r['name']} ({r['path']}) - score: {r['score']:.3f}")

    print("\n--- Structural: GpuContext contains ---")
    results = store.query_contains("engine/gfx/gpu-context.ts::GpuContext")
    for r in results:
        print(f"  [{r['type']}] {r['name']}")


if __name__ == "__main__":
    main()
