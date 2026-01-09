"""
HALO Latent Space - Populate Code Graph

Uses existing ScornSpine infrastructure to:
1. Parse engine/ with Tree-sitter
2. Generate embeddings via Spine's model
3. Store in new "halo-code-graph" Qdrant collection

Run from haloscorn directory:
    python -m latent_space.populate_graph
"""

import sys
import json
import requests
from pathlib import Path
from uuid import uuid4

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from latent_space.code_indexer import CodeIndexer, CodeNode


SPINE_URL = "http://127.0.0.1:7782"
BASE_PATH = "F:/primewave-engine"
ENGINE_PATH = f"{BASE_PATH}/engine"


def embed_text(text: str) -> list:
    """Get embedding from Spine server."""
    try:
        resp = requests.post(
            f"{SPINE_URL}/embed",
            json={"text": text},
            timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embed failed: {resp.text}")
        return resp.json()["embedding"]
    except requests.RequestException as e:
        raise RuntimeError(f"Spine connection failed: {e}") from e


def build_embed_text(node: CodeNode, edges: list) -> str:
    """Build text to embed for a code node."""
    parts = [f"{node.type}: {node.name}"]

    if node.signature:
        parts.append(f"signature: {node.signature}")

    if node.docstring:
        doc = node.docstring.strip()
        if doc.startswith('/**'):
            doc = doc[3:]
        if doc.endswith('*/'):
            doc = doc[:-2]
        parts.append(doc.strip()[:300])

    # Abbreviated code
    code = node.code[:400] if len(node.code) > 400 else node.code
    parts.append(f"code: {code}")

    # Call relationships
    calls = [e.to_node for e in edges if e.from_node == node.id and e.type == "CALLS"]
    if calls:
        parts.append(f"calls: {', '.join(calls[:8])}")

    return '\n'.join(parts)


def main():
    print("=" * 70)
    print("HALO Latent Space - Populate Code Graph")
    print("=" * 70)

    # Check Spine is running
    print("\n[1/5] Checking Spine health...")
    try:
        resp = requests.get(f"{SPINE_URL}/health", timeout=5)
        health = resp.json()
        print(f"  ? Spine status: {health.get('status')}")
        print(f"  ? Embedding model: {health.get('embedding_model')}")
    except Exception as e:
        print(f"  ? Spine not available: {e}")
        print("  Start Spine first: python -m haloscorn.scornspine.server")
        return 1

    # Index engine with tree-sitter
    print("\n[2/5] Indexing engine/ with Tree-sitter...")
    indexer = CodeIndexer()
    nodes, edges = indexer.index_directory(ENGINE_PATH, BASE_PATH)

    print(f"  ? Found {len(nodes)} nodes, {len(edges)} edges")

    # Build edge lookup
    edges_by_node = {}
    for edge in edges:
        if edge.from_node not in edges_by_node:
            edges_by_node[edge.from_node] = []
        edges_by_node[edge.from_node].append(edge)

    # Skip file nodes, embed the rest
    nodes_to_embed = [n for n in nodes if n.type != "file"]
    print(f"  ? {len(nodes_to_embed)} nodes to embed (excluding file nodes)")

    # Generate embeddings
    print("\n[3/5] Generating embeddings (this may take a while)...")
    embedded_points = []
    failed = 0

    for i, node in enumerate(nodes_to_embed):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(nodes_to_embed)}...")

        node_edges = edges_by_node.get(node.id, [])
        embed_text = build_embed_text(node, node_edges)

        try:
            embedding = embed_text_via_spine(embed_text)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  ? Failed to embed {node.id}: {e}")
            continue

        # Build payload
        calls = [e.to_node for e in node_edges if e.type == "CALLS"]
        called_by = [
            e.from_node for e in edges
            if (e.to_node == node.name or e.to_node.endswith(f"::{node.name}"))
            and e.type == "CALLS"
        ]

        point = {
            "id": str(uuid4()),
            "vector": embedding,
            "payload": {
                "node_id": node.id,
                "type": node.type,
                "name": node.name,
                "path": node.path,
                "language": node.language,
                "line_start": node.line_start,
                "line_end": node.line_end,
                "signature": node.signature,
                "docstring": node.docstring[:500] if node.docstring else None,
                "contained_by": node.contained_by,
                "calls": calls[:50],
                "called_by": called_by[:50],
            }
        }
        embedded_points.append(point)

    print(f"  ? Embedded {len(embedded_points)} nodes ({failed} failed)")

    # Save to JSON for manual import (since we don't have direct Qdrant access here)
    print("\n[4/5] Saving to data/code-graph-vectors.json...")
    output_path = f"{BASE_PATH}/data/code-graph-vectors.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "collection": "halo-code-graph",
            "points_count": len(embedded_points),
            "points": embedded_points
        }, f)

    print(f"  ? Saved {len(embedded_points)} points to {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

    # Provide import instructions
    print("\n[5/5] Next steps:")
    print("  The vectors are saved locally. To import to Qdrant:")
    print("  1. Add endpoint to Spine for code-graph collection creation")
    print("  2. Or use Qdrant Cloud console to import")
    print()
    print("  Stats summary:")
    print(f"    - Total nodes: {len(nodes)}")
    print(f"    - Embedded: {len(embedded_points)}")
    print(f"    - Skipped: {len(nodes) - len(embedded_points)}")
    print(f"    - Unique types: {set(n.type for n in nodes_to_embed)}")

    return 0


def embed_text_via_spine(text: str) -> list:
    """Get embedding from Spine server."""
    try:
        resp = requests.post(
            f"{SPINE_URL}/embed",
            json={"text": text},
            timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embed failed: {resp.text}")
        return resp.json()["embedding"]
    except requests.RequestException as e:
        raise RuntimeError(f"Spine connection failed: {e}") from e


if __name__ == "__main__":
    sys.exit(main())
