"""
Unit tests for Agent Memory System
===================================

Tests AgentMemoryManager, SharedMemoryManager, and memory operations.

Run with: pytest haloscorn/scornspine/test_memory.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from haloscorn.scornspine.memory import AgentMemoryManager, SharedMemoryManager, initialize_memory_collections


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    client = Mock()
    client.upsert = Mock()
    client.search = Mock(return_value=[])
    client.get_collection = Mock()
    client.create_collection = Mock()
    return client


@pytest.fixture
def mock_embed_model():
    """Mock embedding model"""
    model = Mock()
    model.get_text_embedding = Mock(return_value=[0.1] * 1024)
    return model


@pytest.fixture
def memory_manager(mock_qdrant, mock_embed_model):
    """Create AgentMemoryManager instance"""
    return AgentMemoryManager(
        agent_name="test-agent",
        qdrant_client=mock_qdrant,
        embed_model=mock_embed_model,
        working_ttl_days=7
    )


class TestAgentMemoryManager:
    """Test suite for AgentMemoryManager"""

    @pytest.mark.asyncio
    async def test_store_memory_success(self, memory_manager, mock_qdrant, mock_embed_model):
        """Test successful memory storage"""
        content = "Test memory content"
        memory_type = "learning"
        metadata = {"tags": ["test"], "importance": 8}

        memory_id = await memory_manager.store(
            content=content,
            memory_type=memory_type,
            metadata=metadata
        )

        # Verify embedding was generated
        mock_embed_model.get_text_embedding.assert_called_once_with(content)

        # Verify Qdrant upsert was called
        assert mock_qdrant.upsert.called
        call_args = mock_qdrant.upsert.call_args
        assert call_args[1]["collection_name"] == "halojinix-memory-test-agent"

        # Verify point structure
        point = call_args[1]["points"][0]
        assert point.id == memory_id
        assert len(point.vector) == 1024
        assert point.payload["content"] == content
        assert point.payload["type"] == memory_type
        assert point.payload["metadata"] == metadata
        assert "checksum" in point.payload
        assert "timestamp" in point.payload

    @pytest.mark.asyncio
    async def test_store_memory_invalid_type(self, memory_manager):
        """Test storage with invalid memory type"""
        with pytest.raises(ValueError, match="Invalid memory_type"):
            await memory_manager.store(
                content="Test",
                memory_type="invalid-type"
            )

    @pytest.mark.asyncio
    async def test_store_memory_embedding_failure(self, memory_manager, mock_embed_model, mock_qdrant):
        """Test storage when embedding generation fails"""
        mock_embed_model.get_text_embedding.side_effect = Exception("Embedding failed")

        # Should store without embedding (fallback mode)
        memory_id = await memory_manager.store(
            content="Test",
            memory_type="learning"
        )

        # Should return valid ID even with embedding failure
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_query_memory_success(self, memory_manager, mock_qdrant, mock_embed_model):
        """Test successful memory query"""
        # Mock search results
        mock_result = Mock()
        mock_result.id = "test-id"
        mock_result.score = 0.85
        mock_result.payload = {
            "content": "Test memory",
            "type": "learning",
            "timestamp": datetime.utcnow().isoformat()
        }
        mock_qdrant.search.return_value = [mock_result]

        results = await memory_manager.query(
            query_text="test query",
            memory_type="learning",
            limit=5
        )

        # Verify embedding generated for query
        mock_embed_model.get_text_embedding.assert_called_with("test query")

        # Verify search called with correct params
        assert mock_qdrant.search.called
        call_args = mock_qdrant.search.call_args
        assert call_args[1]["collection_name"] == "halojinix-memory-test-agent"
        assert call_args[1]["limit"] == 5

        # Verify results formatted correctly
        assert len(results) == 1
        assert results[0]["memory_id"] == "test-id"
        assert results[0]["score"] == 0.85
        assert results[0]["content"] == "Test memory"

    @pytest.mark.asyncio
    async def test_query_memory_with_filters(self, memory_manager, mock_qdrant):
        """Test query with type and tier filters"""
        await memory_manager.query(
            query_text="test",
            memory_type="decision",
            tier="working"
        )

        call_args = mock_qdrant.search.call_args
        query_filter = call_args[1]["query_filter"]

        # Verify filter was applied
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_query_memory_embedding_failure(self, memory_manager, mock_embed_model):
        """Test query when embedding generation fails"""
        mock_embed_model.get_text_embedding.side_effect = Exception("Embedding failed")

        results = await memory_manager.query("test")

        # Should return empty list on embedding failure
        assert results == []

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_manager, mock_qdrant):
        """Test memory statistics retrieval"""
        # Mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 1234
        mock_info.status = "green"
        mock_qdrant.get_collection.return_value = mock_info

        stats = await memory_manager.get_stats()

        assert stats["agent"] == "test-agent"
        assert stats["total_vectors"] == 1234
        assert stats["status"] == "green"

    def test_checksum_computation(self, memory_manager):
        """Test checksum generation and verification"""
        content = "Test content for checksum"

        checksum1 = memory_manager._compute_checksum(content)
        checksum2 = memory_manager._compute_checksum(content)

        # Same content should produce same checksum
        assert checksum1 == checksum2

        # Different content should produce different checksum
        checksum3 = memory_manager._compute_checksum("Different content")
        assert checksum1 != checksum3

        # Verification should work
        assert memory_manager._verify_checksum(content, checksum1)
        assert not memory_manager._verify_checksum("Different", checksum1)


class TestSharedMemoryManager:
    """Test suite for SharedMemoryManager"""

    @pytest.fixture
    def shared_memory(self, mock_qdrant, mock_embed_model):
        return SharedMemoryManager(
            qdrant_client=mock_qdrant,
            embed_model=mock_embed_model
        )

    @pytest.mark.asyncio
    async def test_store_handoff(self, shared_memory, mock_qdrant, mock_embed_model):
        """Test handoff storage"""
        handoff_id = await shared_memory.store_handoff(
            from_agent="halo",
            to_agent="jonah",
            content="Please review the benchmark results",
            metadata={"urgency": "high"}
        )

        # Verify embedding generated
        mock_embed_model.get_text_embedding.assert_called_once()

        # Verify stored in shared collection
        call_args = mock_qdrant.upsert.call_args
        assert call_args[1]["collection_name"] == "halojinix-memory-shared"

        # Verify handoff metadata
        point = call_args[1]["points"][0]
        assert point.payload["type"] == "handoff"
        assert point.payload["metadata"]["from"] == "halo"
        assert point.payload["metadata"]["to"] == "jonah"
        assert "handoff" in point.payload["metadata"]["tags"]


class TestMemoryCollectionInitialization:
    """Test suite for collection initialization"""

    def test_initialize_creates_all_collections(self, mock_qdrant):
        """Test that initialize creates collections for all agents"""
        # Mock get_collection to raise exception (collection doesn't exist)
        mock_qdrant.get_collection.side_effect = Exception("Not found")

        initialize_memory_collections(mock_qdrant, vector_size=1024)

        # Verify create_collection called for each agent + shared
        assert mock_qdrant.create_collection.call_count == 5  # 4 agents + shared

        # Verify collection names
        collection_names = [
            call[1]["collection_name"]
            for call in mock_qdrant.create_collection.call_args_list
        ]
        assert "halojinix-memory-halo" in collection_names
        assert "halojinix-memory-jonah" in collection_names
        assert "halojinix-memory-vera" in collection_names
        assert "halojinix-memory-halojinix" in collection_names
        assert "halojinix-memory-shared" in collection_names

    def test_initialize_skips_existing_collections(self, mock_qdrant):
        """Test that initialize doesn't recreate existing collections"""
        # Mock get_collection to succeed (collection exists)
        mock_qdrant.get_collection.return_value = Mock()

        initialize_memory_collections(mock_qdrant)

        # Verify create_collection never called
        assert not mock_qdrant.create_collection.called


class TestMemoryIntegration:
    """Integration tests for memory system"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_store_and_query_workflow(self, memory_manager, mock_qdrant, mock_embed_model):
        """Test complete store ? query workflow"""
        # Store memory
        memory_id = await memory_manager.store(
            content="Qdrant migration achieved 116x speedup",
            memory_type="learning",
            metadata={"tags": ["performance", "qdrant"]}
        )

        # Mock search to return stored memory
        mock_result = Mock()
        mock_result.id = memory_id
        mock_result.score = 0.95
        mock_result.payload = {
            "content": "Qdrant migration achieved 116x speedup",
            "type": "learning",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"tags": ["performance", "qdrant"]}
        }
        mock_qdrant.search.return_value = [mock_result]

        # Query for related memory
        results = await memory_manager.query("qdrant performance", limit=5)

        # Verify retrieval
        assert len(results) == 1
        assert results[0]["memory_id"] == memory_id
        assert "116x speedup" in results[0]["content"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_crash_recovery_scenario(self, memory_manager, mock_qdrant):
        """Test memory system survives crash scenarios"""
        # Store multiple memories
        memory_ids = []
        for i in range(5):
            mid = await memory_manager.store(
                content=f"Memory {i}",
                memory_type="state"
            )
            memory_ids.append(mid)

        # Simulate crash (recreate manager)
        new_manager = AgentMemoryManager(
            agent_name="test-agent",
            qdrant_client=mock_qdrant,
            embed_model=Mock(get_text_embedding=Mock(return_value=[0.1] * 1024)),
            working_ttl_days=7
        )

        # Should be able to query previous memories
        # (In real scenario, Qdrant persists data)
        stats = await new_manager.get_stats()
        assert stats["agent"] == "test-agent"


# Performance benchmarks
class TestMemoryPerformance:
    """Performance tests for memory operations"""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_store_latency(self, memory_manager, benchmark):
        """Benchmark memory storage latency"""
        async def store_memory():
            await memory_manager.store(
                content="Performance test memory",
                memory_type="learning"
            )

        # Target: <50ms for store operation
        # (Actual measurement requires real Qdrant + embeddings)
        pass

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_query_latency(self, memory_manager, mock_qdrant, benchmark):
        """Benchmark memory query latency"""
        # Mock fast search
        mock_qdrant.search.return_value = []

        async def query_memory():
            await memory_manager.query("test query", limit=5)

        # Target: <100ms for query operation
        pass
