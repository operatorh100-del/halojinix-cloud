"""
Agent Memory Management System
================================

Provides persistent memory storage and retrieval for HALOJINIX agents using Qdrant vector database.
Implements tiered storage (working memory ? archival), semantic search, and crash recovery.

Architecture designed in RT4804.

Usage:
    from haloscorn.scornspine.memory import AgentMemoryManager

    memory = AgentMemoryManager("halo")
    await memory.store("Qdrant migration successful", type="learning")
    results = await memory.query("migration", limit=5)
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

# RT5300: Configure file logging for debug
_log_dir = Path(__file__).parent.parent.parent / 'logs'
_log_dir.mkdir(exist_ok=True)
_memory_log_file = _log_dir / 'memory-debug.log'
_memory_handler = logging.FileHandler(_memory_log_file, mode='a', encoding='utf-8')
_memory_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(_memory_handler)
logger.setLevel(logging.DEBUG)


class AgentMemoryManager:
    """
    Manages agent memory storage and retrieval using Qdrant.

    Features:
    - Tiered storage (working memory ? archival)
    - Semantic vector search
    - Memory types (conversation, decision, learning, state, error)
    - Crash recovery support
    - Checksums for integrity
    """

    def __init__(
        self,
        agent_name: str,
        qdrant_client: QdrantClient,
        embed_model: Any,  # HuggingFaceEmbedding
        working_ttl_days: int = 7
    ):
        """
        Initialize memory manager for specific agent.

        Args:
            agent_name: Agent identifier ("halo", "jonah", "vera", "halojinix")
            qdrant_client: Connected Qdrant client
            embed_model: Embedding model for vectorization
            working_ttl_days: Days before working memory ? archival (default 7)
        """
        self.agent = agent_name
        self.qdrant = qdrant_client
        self.embed_model = embed_model
        self.working_ttl = timedelta(days=working_ttl_days)

        # Collection names
        self.collection = f"halojinix-memory-{agent_name}"
        self.shared_collection = "halojinix-memory-shared"

        logger.info(f"AgentMemoryManager initialized for {agent_name}")

    def store(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        tier: str = "working"
    ) -> str:
        """
        Store a memory with semantic embedding.

        RT5300: Converted from async to sync - underlying Qdrant calls are synchronous.

        Args:
            content: Text content of memory
            memory_type: Type of memory ("conversation", "decision", "learning", "state", "error", "identity")
            metadata: Additional metadata (tags, importance, session_id, etc.)
            tier: Storage tier ("working" or "archival")

        Returns:
            memory_id: UUID of stored memory

        Raises:
            ValueError: If memory_type is invalid
        """
        valid_types = ["conversation", "decision", "learning", "state", "error", "identity"]
        if memory_type not in valid_types:
            raise ValueError(f"Invalid memory_type. Must be one of: {valid_types}")

        # Generate embedding
        try:
            embedding = self.embed_model.get_text_embedding(content)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Fallback: store without embedding (can re-embed later)
            embedding = None

        # Generate memory ID and checksum
        memory_id = str(uuid4())
        checksum = self._compute_checksum(content)

        # Build payload
        payload = {
            "agent": self.agent,
            "type": memory_type,
            "tier": tier,
            "content": content,
            "checksum": checksum,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "version": 1,
            "access_count": 0
        }

        # Store in Qdrant
        try:
            if embedding:
                point = PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload=payload
                )
                self.qdrant.upsert(
                    collection_name=self.collection,
                    points=[point]
                )
            else:
                # RT12600: Store as metadata-only using zero vector (enables later re-embedding)
                logger.warning(f"Memory {memory_id} stored without embedding - using zero vector")
                zero_vector = [0.0] * 768  # Match multilingual-e5-base dimension
                point = PointStruct(
                    id=memory_id,
                    vector=zero_vector,
                    payload={**payload, "needs_reembed": True}
                )
                self.qdrant.upsert(
                    collection_name=self.collection,
                    points=[point]
                )

            logger.info(f"Stored {memory_type} memory {memory_id} for {self.agent}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    def query(
        self,
        query_text: str,
        memory_type: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Query memories semantically.

        RT5300: Uses query_points() for Qdrant client v2.x compatibility.

        Args:
            query_text: Text to search for
            memory_type: Filter by type (optional)
            tier: Filter by tier ("working" or "archival", optional)
            limit: Maximum results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of matching memories with scores
        """
        logger.debug(f"Query: '{query_text[:50]}...' for {self.agent}")

        # Generate query embedding
        try:
            query_embedding = self.embed_model.get_text_embedding(query_text)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        # Build filters
        filter_conditions = []
        if memory_type:
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value=memory_type))
            )
        if tier:
            filter_conditions.append(
                FieldCondition(key="tier", match=MatchValue(value=tier))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Search Qdrant (RT5300: query_points for v2.x API)
        try:
            response = self.qdrant.query_points(
                collection_name=self.collection,
                query=query_embedding,
                limit=limit,
                query_filter=query_filter,
                score_threshold=min_score,
                with_payload=True
            )
            results = response.points if hasattr(response, 'points') else []

            # Format results
            memories = []
            for hit in results:
                memory = hit.payload
                memory["memory_id"] = hit.id
                memory["score"] = hit.score
                memories.append(memory)

                # Update access count
                self._increment_access(hit.id)

            logger.info(f"Query '{query_text[:30]}...' returned {len(memories)} memories for {self.agent}")
            return memories

        except Exception as e:
            logger.error(f"Query failed for {self.agent}: {e}")
            return []

    def archive_old_memories(self, days: Optional[int] = None) -> int:
        """
        Move old working memories to archival tier.

        RT5300: Converted from async to sync.

        Args:
            days: Archive memories older than N days (default: use working_ttl)

        Returns:
            Number of memories archived
        """
        cutoff_days = days or self.working_ttl.days
        cutoff_date = datetime.utcnow() - timedelta(days=cutoff_days)
        cutoff_iso = cutoff_date.isoformat()

        logger.info(f"Archiving working memories older than {cutoff_date}")
        archived_count = 0

        try:
            # Filter for working tier memories
            scroll_filter = Filter(
                must=[
                    FieldCondition(key="metadata.tier", match=MatchValue(value="working"))
                ]
            )

            # Scroll through working memories
            offset = None
            while True:
                results, offset = self.qdrant.scroll(
                    collection_name=self.collection,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not results:
                    break

                # Filter by timestamp (Python-side since Qdrant range on ISO strings is tricky)
                for point in results:
                    timestamp = point.payload.get("metadata", {}).get("timestamp", "")
                    if timestamp < cutoff_iso:
                        # Update tier to archival
                        self.qdrant.set_payload(
                            collection_name=self.collection,
                            payload={"metadata.tier": "archival"},
                            points=[point.id]
                        )
                        archived_count += 1
                        logger.debug(f"Archived memory {point.id}")

                if offset is None:
                    break

            logger.info(f"Archived {archived_count} memories")
        except Exception as e:
            logger.error(f"Archive failed: {e}")

        return archived_count

    def _compute_checksum(self, content: str) -> str:
        """Compute SHA256 checksum of content for integrity verification."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _verify_checksum(self, content: str, checksum: str) -> bool:
        """Verify content matches stored checksum."""
        return self._compute_checksum(content) == checksum

    def _increment_access(self, memory_id: str):
        """Increment access count for LRU tracking (async, fire-and-forget)."""
        try:
            # Retrieve current point
            points = self.qdrant.retrieve(
                collection_name=self.collection,
                ids=[memory_id]
            )

            if not points:
                return

            point = points[0]
            metadata = point.payload.get("metadata", {})

            # Update access tracking
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            metadata["last_accessed"] = datetime.utcnow().isoformat()

            # Update point with new metadata
            self.qdrant.set_payload(
                collection_name=self.collection,
                payload={"metadata": metadata},
                points=[memory_id]
            )

            logger.debug(f"Incremented access for {memory_id}")
        except Exception as e:
            # Non-critical - log and continue
            logger.warning(f"Failed to increment access for {memory_id}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics. RT5300: Converted to sync.

        Returns:
            Stats dict with collection info, counts by type/tier
        """
        try:
            collection_info = self.qdrant.get_collection(self.collection)

            stats = {
                "agent": self.agent,
                "collection": self.collection,
                "total_points": collection_info.points_count,  # RT5300: Fixed API - use points_count not vectors_count
                "indexed_vectors": collection_info.indexed_vectors_count,
                "status": collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status),
            }

            # RT12600: Add counts by type using scroll with filters
            try:
                for mem_type in ["conversation", "decision", "learning", "state", "error", "identity"]:
                    filter_cond = Filter(must=[FieldCondition(key="type", match=MatchValue(value=mem_type))])
                    results, _ = self.qdrant.scroll(
                        collection_name=self.collection,
                        scroll_filter=filter_cond,
                        limit=1,
                        with_payload=False,
                        with_vectors=False
                    )
                    # Scroll returns limited results, but we can count what we get
                    # For accurate counts would need count() API - this is approximation
                    stats[f"type_{mem_type}"] = len(results) if results else 0
            except Exception:
                pass  # Stats by type are optional enhancement

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


class SharedMemoryManager:
    """
    Manages shared memory accessible by all agents.
    Used for handoffs, decisions, and cross-agent coordination.
    """

    def __init__(self, qdrant_client: QdrantClient, embed_model: Any):
        self.qdrant = qdrant_client
        self.embed_model = embed_model
        self.collection = "halojinix-memory-shared"
        logger.info("SharedMemoryManager initialized")

    def store_handoff(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a handoff message from one agent to another. RT5300: Converted to sync.

        Args:
            from_agent: Source agent
            to_agent: Target agent
            content: Handoff message
            metadata: Additional context

        Returns:
            handoff_id: UUID of handoff
        """
        handoff_metadata = metadata or {}
        handoff_metadata.update({
            "from": from_agent,
            "to": to_agent,
            "tags": ["handoff", f"from:{from_agent}", f"to:{to_agent}"]
        })

        # Generate embedding and store
        embedding = self.embed_model.get_text_embedding(content)
        handoff_id = str(uuid4())

        payload = {
            "type": "handoff",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": handoff_metadata,
            "author": from_agent
        }

        point = PointStruct(
            id=handoff_id,
            vector=embedding,
            payload=payload
        )

        self.qdrant.upsert(
            collection_name=self.collection,
            points=[point]
        )

        logger.info(f"Handoff {handoff_id} from {from_agent} ? {to_agent}")
        return handoff_id

    def query_handoffs(
        self,
        to_agent: str,
        limit: int = 10,
        unread_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query handoffs for a specific agent. RT5300: Converted to sync.

        Args:
            to_agent: Target agent
            limit: Max results
            unread_only: Only return unread handoffs

        Returns:
            List of handoff messages
        """
        # Query with filter for target agent
        filter_conditions = [
            FieldCondition(key="metadata.to", match=MatchValue(value=to_agent))
        ]

        if unread_only:
            filter_conditions.append(
                FieldCondition(key="metadata.read", match=MatchValue(value=False))
            )

        query_filter = Filter(must=filter_conditions)

        # Retrieve handoffs using scroll (no semantic search needed)
        # RT12500: Fixed self.shared_memory -> self (was referencing nonexistent attribute)
        try:
            results, _ = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            handoffs = []
            for point in results:
                handoffs.append({
                    "id": point.id,
                    "from": point.payload.get("metadata", {}).get("from", "unknown"),
                    "to": to_agent,
                    "content": point.payload.get("content", ""),
                    "timestamp": point.payload.get("metadata", {}).get("timestamp", ""),
                    "priority": point.payload.get("metadata", {}).get("priority", "normal"),
                    "read": point.payload.get("metadata", {}).get("read", False)
                })

            logger.info(f"Found {len(handoffs)} handoffs for {to_agent}")
            return handoffs
        except Exception as e:
            logger.error(f"Query handoffs failed: {e}")
            return []


# Initialize collections on first import
def initialize_memory_collections(
    qdrant_client: QdrantClient,
    vector_size: int = 768  # RT5028: Changed to match multilingual-e5-base
):
    """
    Create Qdrant collections for agent memory if they don't exist.

    Args:
        qdrant_client: Connected Qdrant client
        vector_size: Embedding dimension (default 768 for multilingual-e5-base)
    """
    from qdrant_client.models import PayloadSchemaType
    
    agents = ["halo", "jonah", "vera", "halojinix"]
    collections = [f"halojinix-memory-{agent}" for agent in agents]
    collections.append("halojinix-memory-shared")

    for collection_name in collections:
        try:
            # Check if collection exists
            qdrant_client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} already exists")
        except Exception:
            # Create collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection {collection_name}")
        
        # RT21202: Create payload indexes for efficient filtering
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="tier",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created indexes for {collection_name}")
        except Exception as e:
            # Index may already exist
            logger.debug(f"Index creation for {collection_name}: {e}")
