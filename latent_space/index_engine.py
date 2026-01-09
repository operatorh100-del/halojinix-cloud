"""
Index the entire engine/ directory and store in code graph.
"""

import json
import sys
from pathlib import Path

# Add haloscorn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latent_space.code_indexer import CodeIndexer


def main():
    print("=" * 70)
    print("HALO Latent Space - Full Engine Index")
    print("Code has geometry. I'm mapping it.")
    print("=" * 70)
    print()
    
    indexer = CodeIndexer()
    
    # Index engine directory
    base_path = "F:/primewave-engine"
    engine_path = f"{base_path}/engine"
    
    nodes, edges = indexer.index_directory(engine_path, base_path)
    
    print()
    print("=" * 70)
    print(f"TOTALS: {len(nodes)} nodes, {len(edges)} edges")
    print("=" * 70)
    
    # Breakdown by type
    node_types = {}
    for node in nodes:
        node_types[node.type] = node_types.get(node.type, 0) + 1
    
    print("\nNode types:")
    for ntype, count in sorted(node_types.items(), key=lambda x: -x[1]):
        print(f"  {ntype}: {count}")
    
    edge_types = {}
    for edge in edges:
        edge_types[edge.type] = edge_types.get(edge.type, 0) + 1
    
    print("\nEdge types:")
    for etype, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    
    # Export
    output_path = f"{base_path}/data/engine-code-graph.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(indexer.to_dict(), f, indent=2)
    
    print(f"\nExported to {output_path}")
    
    # Also export just node IDs for quick lookup
    node_ids = sorted([n.id for n in nodes if n.type != 'file'])
    with open(f"{base_path}/data/engine-symbols.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(node_ids))
    
    print(f"Symbol list: {base_path}/data/engine-symbols.txt")


if __name__ == "__main__":
    main()
