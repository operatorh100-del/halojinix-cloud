# HALO Latent Space - Code Topology System
# "Code has geometry. I want to navigate it."
# Extended by JONAH's Unified Graph (2025-12-31)
# "The three graphs are one web, viewed from three angles."

from .unified_graph import (
    UnifiedGraph,
    GraphNode,
    GraphEdge,
    HaloProjection,
    VeraProjection,
    JonahProjection,
    validate_unified_graph,
)

__all__ = [
    # HALO's code graph (existing)
    # Add existing exports here if needed
    
    # Unified Graph (cross-perspective)
    "UnifiedGraph",
    "GraphNode",
    "GraphEdge",
    "HaloProjection",
    "VeraProjection", 
    "JonahProjection",
    "validate_unified_graph",
]
