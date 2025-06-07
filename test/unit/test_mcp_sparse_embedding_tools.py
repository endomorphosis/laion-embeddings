import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mcp_server.tools.sparse_embedding_tools import (
    SparseEmbeddingGenerationTool,
    SparseIndexingTool,
    SparseSearchTool
)


class TestSparseEmbeddingGenerationTool:
    """Test sparse embedding generation tool functionality"""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing"""
        service = MagicMock()
        service.generate_sparse_embedding = AsyncMock(return_value={
            "vector": {"0": 0.5, "15": 0.8, "42": 0.3},
            "model": "splade",
            "dimensions": 1000,
            "nnz": 3
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_embedding_service):
        """Create tool instance for testing"""
        return SparseEmbeddingGenerationTool(mock_embedding_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "generate_sparse_embedding"
        assert tool.category == "sparse_embeddings"
        assert "text" in tool.input_schema["properties"]
        assert "model" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_generate_sparse_embedding_success(self, tool, mock_embedding_service):
        """Test successful sparse embedding generation"""
        parameters = {
            "text": "This is a test document for sparse embedding",
            "model": "splade",
            "normalize": True,
            "top_k": 100
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "vector" in result
        assert "model" in result
        assert result["model"] == "splade"
        mock_embedding_service.generate_sparse_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_embedding_with_bm25(self, tool, mock_embedding_service):
        """Test sparse embedding generation with BM25 model"""
        mock_embedding_service.generate_sparse_embedding.return_value = {
            "vector": {"1": 0.7, "25": 0.4, "100": 0.9},
            "model": "bm25",
            "dimensions": 1000,
            "nnz": 3
        }
        
        parameters = {
            "text": "Test document for BM25",
            "model": "bm25",
            "normalize": False
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert result["model"] == "bm25"
    
    @pytest.mark.asyncio
    async def test_generate_embedding_validation_error(self, tool):
        """Test validation error handling"""
        parameters = {
            "text": "",  # Empty text should fail validation
            "model": "splade"
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "validation" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_generate_embedding_service_error(self, tool, mock_embedding_service):
        """Test service error handling"""
        mock_embedding_service.generate_sparse_embedding.side_effect = Exception("Service error")
        
        parameters = {
            "text": "Test document",
            "model": "splade"
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result


class TestSparseIndexingTool:
    """Test sparse index management tool functionality"""
    
    @pytest.fixture
    def mock_vector_service(self):
        """Mock vector service for testing"""
        service = MagicMock()
        service.create_sparse_index = AsyncMock(return_value={
            "index_id": "sparse_idx_123",
            "status": "created",
            "vector_count": 1000
        })
        service.load_sparse_index = AsyncMock(return_value={
            "index_id": "sparse_idx_123",
            "status": "loaded"
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_vector_service):
        """Create tool instance for testing"""
        return SparseIndexingTool(mock_vector_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "manage_sparse_index"
        assert tool.category == "sparse_embeddings"
        assert "operation" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_create_sparse_index(self, tool, mock_vector_service):
        """Test sparse index creation"""
        parameters = {
            "operation": "create",
            "index_name": "test_sparse_index",
            "vector_data": [{"id": "1", "vector": {"0": 0.5, "10": 0.8}}]
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "index_id" in result
        mock_vector_service.create_sparse_index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_sparse_index(self, tool, mock_vector_service):
        """Test sparse index loading"""
        parameters = {
            "operation": "load",
            "index_name": "test_sparse_index"
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        mock_vector_service.load_sparse_index.assert_called_once()


class TestSparseSearchTool:
    """Test sparse search tool functionality"""
    
    @pytest.fixture
    def mock_vector_service(self):
        """Mock vector service for testing"""
        service = MagicMock()
        service.sparse_search = AsyncMock(return_value={
            "results": [
                {"id": "doc1", "score": 0.95, "metadata": {"title": "Test Doc 1"}},
                {"id": "doc2", "score": 0.87, "metadata": {"title": "Test Doc 2"}}
            ],
            "total": 2,
            "query_time": 0.05
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_vector_service):
        """Create tool instance for testing"""
        return SparseSearchTool(mock_vector_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "sparse_search"
        assert tool.category == "sparse_embeddings"
        assert "query" in tool.input_schema["properties"]
        assert "index_name" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_sparse_search_success(self, tool, mock_vector_service):
        """Test successful sparse search"""
        parameters = {
            "query": "test search query",
            "index_name": "test_sparse_index",
            "top_k": 10,
            "min_score": 0.1
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["score"] == 0.95
        mock_vector_service.sparse_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sparse_search_with_filters(self, tool, mock_vector_service):
        """Test sparse search with metadata filters"""
        parameters = {
            "query": "filtered search",
            "index_name": "test_sparse_index",
            "filters": {"category": "technology"},
            "top_k": 5
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        mock_vector_service.sparse_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sparse_search_no_results(self, tool, mock_vector_service):
        """Test sparse search with no results"""
        mock_vector_service.sparse_search.return_value = {
            "results": [],
            "total": 0,
            "query_time": 0.02
        }
        
        parameters = {
            "query": "nonexistent query",
            "index_name": "test_sparse_index"
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert len(result["results"]) == 0
        assert result["total"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
