"""
HALO Latent Space - Code Graph Query Tools

Standalone analysis tools that work with the indexed code graph JSON.
No Spine dependency - runs locally against data/engine-code-graph.json.

Usage:
    python -m latent_space.query_graph --find-callers dispatch
    python -m latent_space.query_graph --contains GpuContext
    python -m latent_space.query_graph --impact gpu-context.ts
"""

import json
import argparse
from typing import List, Dict, Set
from collections import defaultdict


class CodeGraph:
    """Query interface for the indexed code graph."""

    def __init__(self, graph_path: str = "F:/primewave-engine/data/engine-code-graph.json"):
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.nodes = {n['id']: n for n in data['nodes']}
        self.edges = data['edges']

        # Build edge indexes
        self.edges_from: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_to: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_type: Dict[str, List[Dict]] = defaultdict(list)

        for edge in self.edges:
            self.edges_from[edge['from']].append(edge)
            self.edges_to[edge['to']].append(edge)
            self.edges_by_type[edge['type']].append(edge)

        print(f"Loaded {len(self.nodes)} nodes, {len(self.edges)} edges")

    def find_node(self, name: str, exact: bool = False) -> List[Dict]:
        """Find nodes by name (partial match by default)."""
        results = []
        for node in self.nodes.values():
            if exact:
                if node['name'] == name:
                    results.append(node)
            else:
                if name.lower() in node['name'].lower() or name.lower() in node['id'].lower():
                    results.append(node)
        return results

    def get_contains(self, parent_id: str) -> List[Dict]:
        """Get all nodes contained by a parent."""
        results = []
        for edge in self.edges_from.get(parent_id, []):
            if edge['type'] == 'CONTAINS':
                child_id = edge['to']
                if child_id in self.nodes:
                    results.append(self.nodes[child_id])
        return results

    def get_callers(self, function_name: str) -> List[Dict]:
        """Find all functions that call a given function."""
        results = []
        seen = set()

        # Match by function name (not full ID)
        for edge in self.edges_by_type.get('CALLS', []):
            if edge['to'] == function_name or edge['to'].endswith(f'::{function_name}'):
                caller_id = edge['from']
                if caller_id not in seen and caller_id in self.nodes:
                    seen.add(caller_id)
                    results.append({
                        'caller': self.nodes[caller_id],
                        'line': edge.get('line'),
                        'evidence': edge.get('evidence')
                    })

        return results

    def get_callees(self, function_id: str) -> List[str]:
        """Find all functions called by a given function."""
        callees = []
        for edge in self.edges_from.get(function_id, []):
            if edge['type'] == 'CALLS':
                callees.append(edge['to'])
        return callees

    def get_imports(self, file_id: str) -> List[str]:
        """Get all imports for a file."""
        imports = []
        for edge in self.edges_from.get(file_id, []):
            if edge['type'] == 'IMPORTS':
                imports.append(edge['to'])
        return imports

    def get_impact(self, file_path: str, depth: int = 2) -> Dict[str, Set[str]]:
        """
        Find what would be affected if a file changes.
        Returns files that import this file, transitively.
        """
        # Normalize path - extract just the filename without extension
        if '/' in file_path:
            file_name = file_path.split('/')[-1].replace('.ts', '').replace('.py', '')
        else:
            file_name = file_path.replace('.ts', '').replace('.py', '')

        affected: Dict[str, Set[str]] = {'direct': set(), 'transitive': set()}

        # Find direct importers - match by file name at end of import path
        for edge in self.edges_by_type.get('IMPORTS', []):
            import_path = edge['to']
            # Extract the imported module name
            import_name = import_path.split('/')[-1] if '/' in import_path else import_path

            if import_name == file_name or file_name in import_path:
                affected['direct'].add(edge['from'])

        # Find transitive (files that import the direct importers)
        if depth > 1:
            for direct_file in affected['direct']:
                # Extract the direct file's name for matching
                direct_name = direct_file.split('/')[-1].replace('.ts', '')
                for edge in self.edges_by_type.get('IMPORTS', []):
                    import_name = edge['to'].split('/')[-1] if '/' in edge['to'] else edge['to']
                    if direct_name == import_name or direct_name in edge['to']:
                        if edge['from'] not in affected['direct']:
                            affected['transitive'].add(edge['from'])

        return affected

    def get_call_chain(self, start_func: str, end_func: str, max_depth: int = 5) -> List[List[str]]:
        """
        Find call paths from start_func to end_func.
        Returns list of paths, each path is a list of function IDs.
        """
        # Find starting nodes
        start_nodes = self.find_node(start_func, exact=False)
        start_ids = [n['id'] for n in start_nodes if n['type'] in ('function', 'method')]

        paths = []

        def dfs(current: str, target: str, path: List[str], visited: Set[str], depth: int):
            if depth > max_depth:
                return
            if current in visited:
                return

            visited.add(current)
            path = path + [current]

            # Check if we reached target
            if target.lower() in current.lower():
                paths.append(path)
                visited.remove(current)
                return

            # Follow calls
            for edge in self.edges_from.get(current, []):
                if edge['type'] == 'CALLS':
                    callee = edge['to']
                    # Try to resolve callee to full ID
                    candidates = self.find_node(callee, exact=True)
                    for cand in candidates:
                        if cand['id'] not in visited:
                            dfs(cand['id'], target, path, visited, depth + 1)

            visited.remove(current)

        for start_id in start_ids:
            dfs(start_id, end_func, [], set(), 0)

        return paths

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the code graph."""
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node['type']] += 1

        edge_types = defaultdict(int)
        for edge in self.edges:
            edge_types[edge['type']] += 1

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types)
        }

    def get_hotspots(self, top_n: int = 20) -> List[Dict]:
        """Find most-called functions/methods (hot spots)."""
        call_counts = defaultdict(int)

        for edge in self.edges_by_type.get('CALLS', []):
            call_counts[edge['to']] += 1

        # Sort by call count
        sorted_calls = sorted(call_counts.items(), key=lambda x: -x[1])[:top_n]

        results = []
        for func_name, count in sorted_calls:
            results.append({
                'name': func_name,
                'call_count': count
            })

        return results

    def get_complexity(self) -> List[Dict]:
        """Find complex nodes (many outgoing calls)."""
        complexity = []

        for node_id, node in self.nodes.items():
            if node['type'] in ('function', 'method'):
                outgoing_calls = len([e for e in self.edges_from.get(node_id, []) if e['type'] == 'CALLS'])
                if outgoing_calls > 0:
                    complexity.append({
                        'id': node_id,
                        'name': node['name'],
                        'type': node['type'],
                        'call_count': outgoing_calls
                    })

        # Sort by complexity (outgoing call count)
        return sorted(complexity, key=lambda x: -x['call_count'])[:20]


def main():
    parser = argparse.ArgumentParser(description='Query the code graph')
    parser.add_argument('--find', help='Find nodes by name')
    parser.add_argument('--callers', help='Find callers of a function')
    parser.add_argument('--callees', help='Find what a function calls')
    parser.add_argument('--contains', help='Find children of a node')
    parser.add_argument('--impact', help='Find impact of changing a file')
    parser.add_argument('--chain', nargs=2, metavar=('START', 'END'), help='Find call chain between functions')
    parser.add_argument('--stats', action='store_true', help='Show graph statistics')
    parser.add_argument('--hotspots', action='store_true', help='Show most-called functions')
    parser.add_argument('--complexity', action='store_true', help='Show complex functions (many outgoing calls)')

    args = parser.parse_args()

    graph = CodeGraph()

    if args.stats:
        stats = graph.get_stats()
        print("\n=== Code Graph Statistics ===")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total edges: {stats['total_edges']}")
        print("\nNode types:")
        for ntype, count in sorted(stats['node_types'].items(), key=lambda x: -x[1]):
            print(f"  {ntype}: {count}")
        print("\nEdge types:")
        for etype, count in sorted(stats['edge_types'].items(), key=lambda x: -x[1]):
            print(f"  {etype}: {count}")

    elif args.find:
        results = graph.find_node(args.find)
        print(f"\n=== Found {len(results)} nodes matching '{args.find}' ===")
        for node in results[:20]:  # Limit output
            print(f"  [{node['type']}] {node['id']}")
            if node.get('signature'):
                print(f"    Signature: {node['signature']}")

    elif args.callers:
        results = graph.get_callers(args.callers)
        print(f"\n=== {len(results)} callers of '{args.callers}' ===")
        for r in results:
            caller = r['caller']
            print(f"  [{caller['type']}] {caller['id']} (line {r.get('line', '?')})")
            if r.get('evidence'):
                print(f"    ? {r['evidence'][:60]}...")

    elif args.callees:
        nodes = graph.find_node(args.callees)
        if nodes:
            node = nodes[0]
            callees = graph.get_callees(node['id'])
            print(f"\n=== '{node['id']}' calls {len(callees)} functions ===")
            for callee in callees:
                print(f"  ? {callee}")

    elif args.contains:
        nodes = graph.find_node(args.contains)
        if nodes:
            node = nodes[0]
            children = graph.get_contains(node['id'])
            print(f"\n=== '{node['id']}' contains {len(children)} nodes ===")
            for child in children:
                sig = f" {child['signature']}" if child.get('signature') else ""
                print(f"  [{child['type']}] {child['name']}{sig}")

    elif args.impact:
        impact = graph.get_impact(args.impact)
        print(f"\n=== Impact of changing '{args.impact}' ===")
        print(f"\nDirect importers ({len(impact['direct'])}):")
        for f in sorted(impact['direct']):
            print(f"  ? {f}")
        print(f"\nTransitive ({len(impact['transitive'])}):")
        for f in sorted(impact['transitive']):
            print(f"  ? {f}")

    elif args.chain:
        start, end = args.chain
        paths = graph.get_call_chain(start, end)
        print(f"\n=== Call paths from '{start}' to '{end}' ===")
        if paths:
            for i, path in enumerate(paths[:5]):  # Limit to 5 paths
                print(f"\nPath {i+1}:")
                for step in path:
                    print(f"  ? {step}")
        else:
            print("No paths found")

    elif args.hotspots:
        hotspots = graph.get_hotspots()
        print("\n=== Most-Called Functions (Hot Spots) ===")
        for i, h in enumerate(hotspots, 1):
            print(f"  {i:2}. {h['name']}: {h['call_count']} calls")

    elif args.complexity:
        complex_funcs = graph.get_complexity()
        print("\n=== Most Complex Functions (Outgoing Calls) ===")
        for i, f in enumerate(complex_funcs, 1):
            print(f"  {i:2}. [{f['type']}] {f['id']}: {f['call_count']} outgoing calls")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
