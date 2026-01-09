"""
ScornSpine - The Memory Backbone of HaloScorn
==============================================

ScornSpine is the persistent memory system that gives Halojinix agents
continuity across sessions. It indexes all project knowledge and conversation
history, enabling full context restoration at any time.

Architecture:
    spine.py      - Core RAG service (vector store + retrieval)
    vertebrae.py  - Document categories and collection
    marrow.py     - Real-time conversation logger
    server.py     - HTTP API server (port 7780)

Usage:
    from haloscorn.scornspine import ScornSpine

    spine = ScornSpine()
    spine.index_all()  # Build the index
    result = spine.query("What are the particle count standards?")

The naming convention for HaloScorn submodules:
    Scorn[Function] - e.g., ScornSpine, ScornVoice, ScornBridge

"Without a spine, there is no persistence.
 Without persistence, there is no continuity.
 Without continuity, there is no intelligence."
"""

from .spine import ScornSpine
from .marrow import ConversationLogger
from .vertebrae import DOCUMENT_CATEGORIES

__all__ = ["ScornSpine", "ConversationLogger", "DOCUMENT_CATEGORIES", "__version__"]
__version__ = "2.0.0"  # RT8800: Centralized version
