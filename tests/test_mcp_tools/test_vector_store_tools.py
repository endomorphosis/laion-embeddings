"""
Comprehensive tests for vector store MCP tools.
"""

import pytest
import asyncio
import tempfile
import json
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Import the vector store tools
from src.mcp_server.tools.vector_store_tools import (
    create_vector_store_tool,
    add_embeddings_to_store_tool,
    search_vector_store_tool,
    get_vector_store_stats_tool,
    delete_from_vector_store_tool,
    optimize_vector_store_tool
)


class TestCreateVectorStoreTool:
    """Test create_vector_store_tool function."""
    
    @pytest.mark.asyncio
    async def test_create_vector_store_success(self, temp_dir):
        """Test successful vector store creation."""
        store_path = Path(temp_dir) / "test_store"
        
        result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss",
            index_type="flat"
        )
        
        assert result["success"] is True
        assert "store_id" in result
        assert result["provider"] == "faiss"
        assert result["dimension"] == 384
        assert result["index_type"] == "flat"
    
    @pytest.mark.asyncio
    async def test_create_vector_store_with_config(self, temp_dir):
        """Test vector store creation with custom config."""
        store_path = Path(temp_dir) / "test_store_config"
        config = {"ef_construction": 200, "m": 16}
        
        result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=768,
            provider="hnswlib",
            index_type="hnsw",
            config=config
        )
        
        assert result["success"] is True
        assert result["config"] == config
        assert result["provider"] == "hnswlib"
    
    @pytest.mark.asyncio
    async def test_create_vector_store_invalid_dimension(self, temp_dir):
        """Test vector store creation with invalid dimension."""
        store_path = Path(temp_dir) / "test_store_invalid"
        
        result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=0,
            provider="faiss"
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "dimension" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_create_vector_store_unsupported_provider(self, temp_dir):
        """Test vector store creation with unsupported provider."""
        store_path = Path(temp_dir) / "test_store_unsupported"
        
        result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="unsupported_provider"
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "provider" in result["error"].lower()


class TestAddEmbeddingsToStoreTool:
    """Test add_embeddings_to_store_tool function."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        return np.random.rand(10, 384).tolist()
    
    @pytest.fixture
    def sample_metadata(self):
        """Generate sample metadata for testing."""
        return [{"id": i, "text": f"sample text {i}"} for i in range(10)]
    
    @pytest.mark.asyncio
    async def test_add_embeddings_success(self, temp_dir, sample_embeddings, sample_metadata):
        """Test successful embeddings addition."""
        store_path = Path(temp_dir) / "test_store"
        
        # First create a store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        assert create_result["success"] is True
        store_id = create_result["store_id"]
        
        # Add embeddings
        result = await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=sample_embeddings,
            metadata=sample_metadata
        )
        
        assert result["success"] is True
        assert result["count"] == 10
        assert "store_id" in result
    
    @pytest.mark.asyncio
    async def test_add_embeddings_batch(self, temp_dir, sample_embeddings, sample_metadata):
        """Test adding embeddings in batches."""
        store_path = Path(temp_dir) / "test_store_batch"
        
        # Create a store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        # Add embeddings with batch size
        result = await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            batch_size=5
        )
        
        assert result["success"] is True
        assert result["count"] == 10
        assert "batches_processed" in result
    
    @pytest.mark.asyncio
    async def test_add_embeddings_dimension_mismatch(self, temp_dir, sample_metadata):
        """Test adding embeddings with dimension mismatch."""
        store_path = Path(temp_dir) / "test_store_mismatch"
        
        # Create a store with 384 dimensions
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        # Try to add embeddings with wrong dimension
        wrong_embeddings = np.random.rand(10, 256).tolist()
        
        result = await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=wrong_embeddings,
            metadata=sample_metadata
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "dimension" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_add_embeddings_nonexistent_store(self, sample_embeddings, sample_metadata):
        """Test adding embeddings to non-existent store."""
        result = await add_embeddings_to_store_tool(
            store_id="nonexistent_store_id",
            embeddings=sample_embeddings,
            metadata=sample_metadata
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestSearchVectorStoreTool:
    """Test search_vector_store_tool function."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        return np.random.rand(50, 384).tolist()
    
    @pytest.fixture
    def sample_metadata(self):
        """Generate sample metadata for testing."""
        return [{"id": i, "text": f"sample text {i}", "category": f"cat_{i % 5}"} for i in range(50)]
    
    @pytest.fixture
    async def populated_store(self, temp_dir, sample_embeddings, sample_metadata):
        """Create and populate a vector store for testing."""
        store_path = Path(temp_dir) / "test_search_store"
        
        # Create store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        # Add embeddings
        await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=sample_embeddings,
            metadata=sample_metadata
        )
        
        return store_id
    
    @pytest.mark.asyncio
    async def test_search_vector_store_success(self, populated_store):
        """Test successful vector store search."""
        query_vector = np.random.rand(384).tolist()
        
        result = await search_vector_store_tool(
            store_id=populated_store,
            query_vector=query_vector,
            k=5
        )
        
        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) <= 5
        assert "search_time" in result
        
        # Check result structure
        if result["results"]:
            for item in result["results"]:
                assert "distance" in item or "score" in item
                assert "metadata" in item
    
    @pytest.mark.asyncio
    async def test_search_vector_store_with_filter(self, populated_store):
        """Test vector store search with metadata filter."""
        query_vector = np.random.rand(384).tolist()
        filter_criteria = {"category": "cat_1"}
        
        result = await search_vector_store_tool(
            store_id=populated_store,
            query_vector=query_vector,
            k=10,
            filter_criteria=filter_criteria
        )
        
        assert result["success"] is True
        assert "results" in result
        
        # Check that all results match the filter
        for item in result["results"]:
            assert item["metadata"]["category"] == "cat_1"
    
    @pytest.mark.asyncio
    async def test_search_vector_store_invalid_dimension(self, populated_store):
        """Test search with invalid query vector dimension."""
        wrong_query_vector = np.random.rand(256).tolist()  # Wrong dimension
        
        result = await search_vector_store_tool(
            store_id=populated_store,
            query_vector=wrong_query_vector,
            k=5
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "dimension" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_search_nonexistent_store(self):
        """Test search on non-existent store."""
        query_vector = np.random.rand(384).tolist()
        
        result = await search_vector_store_tool(
            store_id="nonexistent_store",
            query_vector=query_vector,
            k=5
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestGetVectorStoreStatsTool:
    """Test get_vector_store_stats_tool function."""
    
    @pytest.mark.asyncio
    async def test_get_stats_success(self, temp_dir):
        """Test successful stats retrieval."""
        store_path = Path(temp_dir) / "test_stats_store"
        
        # Create store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        # Get stats
        result = await get_vector_store_stats_tool(store_id=store_id)
        
        assert result["success"] is True
        assert "stats" in result
        assert "total_vectors" in result["stats"]
        assert "dimension" in result["stats"]
        assert "provider" in result["stats"]
        assert "store_id" in result
    
    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, temp_dir):
        """Test stats retrieval with data in store."""
        store_path = Path(temp_dir) / "test_stats_store_data"
        sample_embeddings = np.random.rand(20, 384).tolist()
        sample_metadata = [{"id": i} for i in range(20)]
        
        # Create and populate store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=sample_embeddings,
            metadata=sample_metadata
        )
        
        # Get stats
        result = await get_vector_store_stats_tool(store_id=store_id)
        
        assert result["success"] is True
        assert result["stats"]["total_vectors"] == 20
        assert "memory_usage" in result["stats"]
        assert "index_type" in result["stats"]
    
    @pytest.mark.asyncio
    async def test_get_stats_nonexistent_store(self):
        """Test stats retrieval for non-existent store."""
        result = await get_vector_store_stats_tool(store_id="nonexistent_store")
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestDeleteFromVectorStoreTool:
    """Test delete_from_vector_store_tool function."""
    
    @pytest.mark.asyncio
    async def test_delete_by_ids_success(self, temp_dir):
        """Test successful deletion by IDs."""
        store_path = Path(temp_dir) / "test_delete_store"
        sample_embeddings = np.random.rand(10, 384).tolist()
        sample_metadata = [{"id": f"item_{i}"} for i in range(10)]
        
        # Create and populate store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=sample_embeddings,
            metadata=sample_metadata
        )
        
        # Delete specific items
        ids_to_delete = ["item_0", "item_1", "item_2"]
        result = await delete_from_vector_store_tool(
            store_id=store_id,
            ids=ids_to_delete
        )
        
        assert result["success"] is True
        assert result["deleted_count"] == len(ids_to_delete)
        assert "remaining_count" in result
    
    @pytest.mark.asyncio
    async def test_delete_by_filter_success(self, temp_dir):
        """Test successful deletion by filter."""
        store_path = Path(temp_dir) / "test_delete_filter_store"
        sample_embeddings = np.random.rand(20, 384).tolist()
        sample_metadata = [{"id": f"item_{i}", "category": f"cat_{i % 3}"} for i in range(20)]
        
        # Create and populate store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=sample_embeddings,
            metadata=sample_metadata
        )
        
        # Delete by filter
        filter_criteria = {"category": "cat_1"}
        result = await delete_from_vector_store_tool(
            store_id=store_id,
            filter_criteria=filter_criteria
        )
        
        assert result["success"] is True
        assert result["deleted_count"] > 0
        assert "remaining_count" in result
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_ids(self, temp_dir):
        """Test deletion of non-existent IDs."""
        store_path = Path(temp_dir) / "test_delete_nonexistent"
        
        # Create empty store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        # Try to delete non-existent items
        result = await delete_from_vector_store_tool(
            store_id=store_id,
            ids=["nonexistent_1", "nonexistent_2"]
        )
        
        assert result["success"] is True
        assert result["deleted_count"] == 0
    
    @pytest.mark.asyncio
    async def test_delete_from_nonexistent_store(self):
        """Test deletion from non-existent store."""
        result = await delete_from_vector_store_tool(
            store_id="nonexistent_store",
            ids=["item_1"]
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestOptimizeVectorStoreTool:
    """Test optimize_vector_store_tool function."""
    
    @pytest.mark.asyncio
    async def test_optimize_store_success(self, temp_dir):
        """Test successful store optimization."""
        store_path = Path(temp_dir) / "test_optimize_store"
        sample_embeddings = np.random.rand(100, 384).tolist()
        sample_metadata = [{"id": i} for i in range(100)]
        
        # Create and populate store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=sample_embeddings,
            metadata=sample_metadata
        )
        
        # Optimize store
        result = await optimize_vector_store_tool(store_id=store_id)
        
        assert result["success"] is True
        assert "optimization_time" in result
        assert "stats_before" in result
        assert "stats_after" in result
    
    @pytest.mark.asyncio
    async def test_optimize_with_options(self, temp_dir):
        """Test store optimization with custom options."""
        store_path = Path(temp_dir) / "test_optimize_options"
        
        # Create store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        # Optimize with options
        optimization_options = {"rebuild_index": True, "compress": True}
        result = await optimize_vector_store_tool(
            store_id=store_id,
            optimization_options=optimization_options
        )
        
        assert result["success"] is True
        assert result["options_applied"] == optimization_options
    
    @pytest.mark.asyncio
    async def test_optimize_nonexistent_store(self):
        """Test optimization of non-existent store."""
        result = await optimize_vector_store_tool(store_id="nonexistent_store")
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestVectorStoreToolsIntegration:
    """Integration tests for vector store tools."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, temp_dir):
        """Test complete vector store workflow."""
        store_path = Path(temp_dir) / "integration_store"
        sample_embeddings = np.random.rand(50, 384).tolist()
        sample_metadata = [{"id": f"doc_{i}", "text": f"document {i}"} for i in range(50)]
        
        # 1. Create store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        assert create_result["success"] is True
        store_id = create_result["store_id"]
        
        # 2. Add embeddings
        add_result = await add_embeddings_to_store_tool(
            store_id=store_id,
            embeddings=sample_embeddings,
            metadata=sample_metadata
        )
        assert add_result["success"] is True
        assert add_result["count"] == 50
        
        # 3. Get stats
        stats_result = await get_vector_store_stats_tool(store_id=store_id)
        assert stats_result["success"] is True
        assert stats_result["stats"]["total_vectors"] == 50
        
        # 4. Search
        query_vector = np.random.rand(384).tolist()
        search_result = await search_vector_store_tool(
            store_id=store_id,
            query_vector=query_vector,
            k=5
        )
        assert search_result["success"] is True
        assert len(search_result["results"]) <= 5
        
        # 5. Delete some items
        delete_result = await delete_from_vector_store_tool(
            store_id=store_id,
            ids=["doc_0", "doc_1", "doc_2"]
        )
        assert delete_result["success"] is True
        assert delete_result["deleted_count"] == 3
        
        # 6. Check stats after deletion
        stats_after_delete = await get_vector_store_stats_tool(store_id=store_id)
        assert stats_after_delete["success"] is True
        assert stats_after_delete["stats"]["total_vectors"] == 47
        
        # 7. Optimize
        optimize_result = await optimize_vector_store_tool(store_id=store_id)
        assert optimize_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_dir):
        """Test concurrent operations on vector store."""
        store_path = Path(temp_dir) / "concurrent_store"
        
        # Create store
        create_result = await create_vector_store_tool(
            store_path=str(store_path),
            dimension=384,
            provider="faiss"
        )
        store_id = create_result["store_id"]
        
        # Prepare data for concurrent operations
        embeddings_batch1 = np.random.rand(25, 384).tolist()
        embeddings_batch2 = np.random.rand(25, 384).tolist()
        metadata_batch1 = [{"id": f"batch1_{i}"} for i in range(25)]
        metadata_batch2 = [{"id": f"batch2_{i}"} for i in range(25)]
        
        # Run concurrent add operations
        add_tasks = [
            add_embeddings_to_store_tool(store_id, embeddings_batch1, metadata_batch1),
            add_embeddings_to_store_tool(store_id, embeddings_batch2, metadata_batch2)
        ]
        
        results = await asyncio.gather(*add_tasks, return_exceptions=True)
        
        # Check that at least one operation succeeded
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        assert success_count >= 1
        
        # Check final state
        stats_result = await get_vector_store_stats_tool(store_id=store_id)
        assert stats_result["success"] is True
        assert stats_result["stats"]["total_vectors"] > 0
