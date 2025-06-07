"""
Tests for embedding-related MCP tools.
"""

import pytest
import asyncio
import os
import tempfile
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

import sys
sys.path.append('/home/barberb/laion-embeddings-1/tests/test_mcp_tools')
sys.path.append('/home/barberb/laion-embeddings-1')
from tests.test_mcp_tools.conftest import (
    mock_embedding_service, sample_embeddings, sample_metadata,
    create_sample_file, TEST_MODEL_NAME, TEST_BATCH_SIZE
)


@pytest.mark.asyncio
class TestEmbeddingTools:
    """Test suite for embedding MCP tools."""
    
    @patch('src.mcp_server.tools.embedding_tools.EmbeddingService')
    async def test_create_embeddings_from_text_tool(self, mock_service_class, mock_embedding_service):
        """Test creating embeddings from text."""
        from src.mcp_server.tools.embedding_tools import create_embeddings_from_text_tool
        
        mock_service_class.return_value = mock_embedding_service
        
        texts = ["Hello world", "This is a test", "Embedding creation"]
        
        result = await create_embeddings_from_text_tool(
            texts=texts,
            model_name=TEST_MODEL_NAME,
            normalize=True,
            batch_size=TEST_BATCH_SIZE
        )
        
        assert result["success"] is True
        assert result["embeddings_created"] == len(texts)
        assert result["model_name"] == TEST_MODEL_NAME
        assert "embeddings" in result
        assert len(result["embeddings"]) == len(texts)
        
        # Verify service was called correctly
        mock_embedding_service.create_embeddings.assert_called_once()
    
    @patch('src.mcp_server.tools.embedding_tools.EmbeddingService')
    async def test_create_embeddings_from_file_tool(self, mock_service_class, mock_embedding_service, temp_dir):
        """Test creating embeddings from file."""
        from src.mcp_server.tools.embedding_tools import create_embeddings_from_file_tool
        
        mock_service_class.return_value = mock_embedding_service
        
        # Create test file
        test_file = os.path.join(temp_dir, "texts.txt")
        create_sample_file(test_file, "Line 1\nLine 2\nLine 3\n")
        
        result = await create_embeddings_from_file_tool(
            file_path=test_file,
            model_name=TEST_MODEL_NAME,
            chunk_size=1000,
            output_path=os.path.join(temp_dir, "embeddings.npy")
        )
        
        assert result["success"] is True
        assert result["file_path"] == test_file
        assert result["model_name"] == TEST_MODEL_NAME
        assert "output_path" in result
        
        mock_embedding_service.create_embeddings.assert_called_once()
    
    async def test_create_embeddings_from_file_tool_invalid_file(self, temp_dir):
        """Test creating embeddings from nonexistent file."""
        from src.mcp_server.tools.embedding_tools import create_embeddings_from_file_tool
        
        invalid_file = os.path.join(temp_dir, "nonexistent.txt")
        
        result = await create_embeddings_from_file_tool(
            file_path=invalid_file,
            model_name=TEST_MODEL_NAME
        )
        
        assert result["success"] is False
        assert "does not exist" in result["error"]
    
    @patch('src.mcp_server.tools.embedding_tools.EmbeddingService')
    async def test_batch_create_embeddings_tool(self, mock_service_class, mock_embedding_service):
        """Test batch embedding creation."""
        from src.mcp_server.tools.embedding_tools import batch_create_embeddings_tool
        
        mock_service_class.return_value = mock_embedding_service
        
        batch_requests = [
            {"texts": ["Text 1", "Text 2"], "model_name": "model1"},
            {"texts": ["Text 3", "Text 4"], "model_name": "model2"},
            {"texts": ["Text 5"], "model_name": "model1"}
        ]
        
        result = await batch_create_embeddings_tool(
            batch_requests=batch_requests,
            parallel=True,
            max_workers=2
        )
        
        assert result["success"] is True
        assert result["total_batches"] == 3
        assert result["successful_batches"] == 3
        assert result["failed_batches"] == 0
        assert len(result["results"]) == 3
    
    @patch('src.mcp_server.tools.embedding_tools.get_supported_models')
    async def test_list_available_models_tool(self, mock_get_models):
        """Test listing available embedding models."""
        from src.mcp_server.tools.embedding_tools import list_available_models_tool
        
        mock_get_models.return_value = [
            {"name": "model1", "dimension": 384, "description": "Small model"},
            {"name": "model2", "dimension": 768, "description": "Large model"}
        ]
        
        result = await list_available_models_tool(provider="sentence-transformers")
        
        assert result["success"] is True
        assert result["provider"] == "sentence-transformers"
        assert len(result["models"]) == 2
        assert result["models"][0]["name"] == "model1"
        assert result["models"][1]["dimension"] == 768
    
    @patch('src.mcp_server.tools.embedding_tools.EmbeddingService')
    async def test_compare_embeddings_tool(self, mock_service_class, mock_embedding_service, sample_embeddings):
        """Test comparing embeddings similarity."""
        from src.mcp_server.tools.embedding_tools import compare_embeddings_tool
        
        mock_service_class.return_value = mock_embedding_service
        
        embedding1 = sample_embeddings[0]
        embedding2 = sample_embeddings[1]
        
        result = await compare_embeddings_tool(
            embedding1=embedding1,
            embedding2=embedding2,
            metric="cosine"
        )
        
        assert result["success"] is True
        assert "similarity_score" in result
        assert result["metric"] == "cosine"
        assert 0 <= result["similarity_score"] <= 1
    
    def test_tool_metadata_structure(self):
        """Test that tool metadata is properly structured."""
        from src.mcp_server.tools.embedding_tools import TOOL_METADATA
        
        # Check create_embeddings_from_text_tool metadata
        text_meta = TOOL_METADATA["create_embeddings_from_text_tool"]
        assert text_meta["name"] == "create_embeddings_from_text_tool"
        assert "description" in text_meta
        assert "parameters" in text_meta
        
        params = text_meta["parameters"]
        assert params["type"] == "object"
        assert "texts" in params["required"]
        assert "model_name" in params["required"]
        
        # Check create_embeddings_from_file_tool metadata
        file_meta = TOOL_METADATA["create_embeddings_from_file_tool"]
        assert file_meta["name"] == "create_embeddings_from_file_tool"
        assert "file_path" in file_meta["parameters"]["required"]
        
        # Check default values
        file_props = file_meta["parameters"]["properties"]
        assert file_props["normalize"]["default"] is True
        assert file_props["batch_size"]["default"] == 32


@pytest.mark.asyncio
class TestStorageTools:
    """Test suite for storage-related MCP tools."""
    
    @patch('src.mcp_server.tools.storage_tools.StorageManager')
    async def test_save_embeddings_tool(self, mock_storage_class, sample_embeddings, sample_metadata, temp_dir):
        """Test saving embeddings to storage."""
        from src.mcp_server.tools.storage_tools import save_embeddings_tool
        
        mock_storage = Mock()
        mock_storage.save_embeddings = AsyncMock(return_value={
            "success": True,
            "file_path": "/saved/embeddings.parquet",
            "count": len(sample_embeddings),
            "size_bytes": 1024000
        })
        mock_storage_class.return_value = mock_storage
        
        output_path = os.path.join(temp_dir, "embeddings.parquet")
        
        result = await save_embeddings_tool(
            embeddings=sample_embeddings[:10],
            metadata=sample_metadata[:10],
            output_path=output_path,
            format="parquet",
            compression="gzip"
        )
        
        assert result["success"] is True
        assert result["embeddings_saved"] == 10
        assert result["output_path"] == output_path
        assert result["format"] == "parquet"
        assert "file_size" in result
    
    @patch('src.mcp_server.tools.storage_tools.StorageManager')
    async def test_load_embeddings_tool(self, mock_storage_class, sample_embeddings, temp_dir):
        """Test loading embeddings from storage."""
        from src.mcp_server.tools.storage_tools import load_embeddings_tool
        
        mock_storage = Mock()
        mock_storage.load_embeddings = AsyncMock(return_value={
            "success": True,
            "embeddings": sample_embeddings[:5],
            "metadata": [{"id": i} for i in range(5)],
            "count": 5
        })
        mock_storage_class.return_value = mock_storage
        
        input_path = os.path.join(temp_dir, "embeddings.parquet")
        
        result = await load_embeddings_tool(
            input_path=input_path,
            limit=5,
            offset=0,
            include_metadata=True
        )
        
        assert result["success"] is True
        assert result["input_path"] == input_path
        assert result["embeddings_loaded"] == 5
        assert len(result["embeddings"]) == 5
        assert "metadata" in result
    
    async def test_load_embeddings_tool_invalid_path(self, temp_dir):
        """Test loading embeddings from invalid path."""
        from src.mcp_server.tools.storage_tools import load_embeddings_tool
        
        invalid_path = os.path.join(temp_dir, "nonexistent.parquet")
        
        result = await load_embeddings_tool(input_path=invalid_path)
        
        assert result["success"] is False
        assert "does not exist" in result["error"]


@pytest.mark.asyncio 
class TestSearchTools:
    """Test suite for search-related MCP tools."""
    
    @patch('src.mcp_server.tools.search_tools.SearchService')
    async def test_semantic_search_tool(self, mock_service_class, sample_embeddings):
        """Test semantic search functionality."""
        from src.mcp_server.tools.search_tools import semantic_search_tool
        
        mock_service = Mock()
        mock_service.search = AsyncMock(return_value={
            "success": True,
            "results": [
                {"id": "1", "score": 0.95, "text": "Result 1"},
                {"id": "2", "score": 0.85, "text": "Result 2"}
            ],
            "query_time": 0.5
        })
        mock_service_class.return_value = mock_service
        
        result = await semantic_search_tool(
            query="test query",
            top_k=5,
            index_id="test_index",
            filter_metadata={"category": "documents"}
        )
        
        assert result["success"] is True
        assert result["query"] == "test query"
        assert result["top_k"] == 5
        assert result["index_id"] == "test_index"
        assert len(result["results"]) == 2
        assert result["query_time"] == 0.5
    
    @patch('src.mcp_server.tools.search_tools.SearchService')
    async def test_batch_search_tool(self, mock_service_class):
        """Test batch search functionality."""
        from src.mcp_server.tools.search_tools import batch_search_tool
        
        mock_service = Mock()
        mock_service.batch_search = AsyncMock(return_value={
            "success": True,
            "total_queries": 3,
            "results": [
                {"query": "query1", "results": [{"id": "1", "score": 0.9}]},
                {"query": "query2", "results": [{"id": "2", "score": 0.8}]},
                {"query": "query3", "results": [{"id": "3", "score": 0.7}]}
            ]
        })
        mock_service_class.return_value = mock_service
        
        queries = ["query1", "query2", "query3"]
        
        result = await batch_search_tool(
            queries=queries,
            index_id="test_index",
            top_k=3,
            parallel=True
        )
        
        assert result["success"] is True
        assert result["total_queries"] == 3
        assert len(result["results"]) == 3
        assert result["parallel"] is True
    
    def test_search_tool_metadata_structure(self):
        """Test search tool metadata structure."""
        from src.mcp_server.tools.search_tools import TOOL_METADATA
        
        # Check semantic_search_tool metadata
        search_meta = TOOL_METADATA["semantic_search_tool"]
        assert search_meta["name"] == "semantic_search_tool"
        assert "query" in search_meta["parameters"]["required"]
        
        # Check default values
        search_props = search_meta["parameters"]["properties"]
        assert search_props["top_k"]["default"] == 10


if __name__ == "__main__":
    pytest.main([__file__])
