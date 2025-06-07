"""
Tests for create embeddings MCP tools.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

import sys
sys.path.append('/home/barberb/laion-embeddings-1/tests/test_mcp_tools')
sys.path.append('/home/barberb/laion-embeddings-1')
from tests.test_mcp_tools.conftest import (
    MockCreateEmbeddingsProcessor, create_sample_file, create_sample_json_file,
    TEST_MODEL_NAME, TEST_BATCH_SIZE, TEST_EMBEDDING_DIM
)


@pytest.mark.asyncio
class TestCreateEmbeddingsTools:
    """Test suite for create embeddings MCP tools."""
    
    @patch('src.mcp_server.tools.create_embeddings_tool.CreateEmbeddingsProcessor')
    async def test_create_embeddings_tool_success(self, mock_processor_class, temp_dir):
        """Test successful embedding creation."""
        from src.mcp_server.tools.create_embeddings_tool import create_embeddings_tool
        
        # Setup mock processor
        mock_processor = MockCreateEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        # Create test input file
        input_path = os.path.join(temp_dir, "input.txt")
        create_sample_file(input_path, "Sample text for embedding")
        
        output_path = os.path.join(temp_dir, "output.parquet")
        
        # Execute tool
        result = await create_embeddings_tool(
            input_path=input_path,
            output_path=output_path,
            model_name=TEST_MODEL_NAME,
            batch_size=TEST_BATCH_SIZE
        )
        
        # Verify result
        assert result["success"] is True
        assert result["input_path"] == input_path
        assert result["output_path"] == output_path
        assert result["model_name"] == TEST_MODEL_NAME
        assert result["batch_size"] == TEST_BATCH_SIZE
        assert result["embeddings_created"] == 100
        assert "processing_time" in result
        assert "output_size" in result
        
        # Verify processor was called correctly
        mock_processor_class.assert_called_once()
    
    async def test_create_embeddings_tool_invalid_input(self, temp_dir):
        """Test embedding creation with invalid input path."""
        from src.mcp_server.tools.create_embeddings_tool import create_embeddings_tool
        
        invalid_path = os.path.join(temp_dir, "nonexistent.txt")
        output_path = os.path.join(temp_dir, "output.parquet")
        
        result = await create_embeddings_tool(
            input_path=invalid_path,
            output_path=output_path
        )
        
        # Verify error handling
        assert result["success"] is False
        assert "does not exist" in result["error"]
        assert result["input_path"] == invalid_path
        assert result["output_path"] == output_path
    
    @patch('src.mcp_server.tools.create_embeddings_tool.CreateEmbeddingsProcessor')
    async def test_create_embeddings_tool_processor_exception(self, mock_processor_class, temp_dir):
        """Test handling of processor exceptions."""
        from src.mcp_server.tools.create_embeddings_tool import create_embeddings_tool
        
        # Setup mock to raise exception
        mock_processor_class.side_effect = Exception("Processing failed")
        
        input_path = os.path.join(temp_dir, "input.txt")
        create_sample_file(input_path, "Sample text")
        output_path = os.path.join(temp_dir, "output.parquet")
        
        result = await create_embeddings_tool(
            input_path=input_path,
            output_path=output_path
        )
        
        # Verify error handling
        assert result["success"] is False
        assert "Processing failed" in result["error"]
    
    @patch('src.mcp_server.tools.create_embeddings_tool.create_embeddings_tool')
    async def test_batch_create_embeddings_tool_success(self, mock_create_tool, temp_dir):
        """Test successful batch embedding creation."""
        from src.mcp_server.tools.create_embeddings_tool import batch_create_embeddings_tool
        
        # Setup mock for individual create_embeddings_tool calls
        mock_create_tool.side_effect = [
            {"success": True, "embeddings_created": 50},
            {"success": True, "embeddings_created": 75},
            {"success": True, "embeddings_created": 25}
        ]
        
        # Create batch configs
        batch_configs = [
            {
                "input_path": os.path.join(temp_dir, "input1.txt"),
                "output_path": os.path.join(temp_dir, "output1.parquet"),
                "model_name": TEST_MODEL_NAME
            },
            {
                "input_path": os.path.join(temp_dir, "input2.txt"),
                "output_path": os.path.join(temp_dir, "output2.parquet"),
                "model_name": TEST_MODEL_NAME
            },
            {
                "input_path": os.path.join(temp_dir, "input3.txt"),
                "output_path": os.path.join(temp_dir, "output3.parquet"),
                "model_name": TEST_MODEL_NAME
            }
        ]
        
        result = await batch_create_embeddings_tool(batch_configs)
        
        # Verify batch result
        assert result["success"] is True
        assert result["total_batches"] == 3
        assert result["successful"] == 3
        assert result["failed"] == 0
        assert len(result["results"]) == 3
        
        # Verify each batch result
        for i, batch_result in enumerate(result["results"]):
            assert batch_result["batch_index"] == i
            assert batch_result["result"]["success"] is True
    
    @patch('src.mcp_server.tools.create_embeddings_tool.create_embeddings_tool')
    async def test_batch_create_embeddings_tool_partial_failure(self, mock_create_tool):
        """Test batch embedding creation with some failures."""
        from src.mcp_server.tools.create_embeddings_tool import batch_create_embeddings_tool
        
        # Setup mock with mixed success/failure
        mock_create_tool.side_effect = [
            {"success": True, "embeddings_created": 50},
            {"success": False, "error": "Processing failed"},
            {"success": True, "embeddings_created": 25}
        ]
        
        batch_configs = [
            {"input_path": "/test1", "output_path": "/out1"},
            {"input_path": "/test2", "output_path": "/out2"},
            {"input_path": "/test3", "output_path": "/out3"}
        ]
        
        result = await batch_create_embeddings_tool(batch_configs)
        
        # Verify partial success handling
        assert result["success"] is True
        assert result["total_batches"] == 3
        assert result["successful"] == 2
        assert result["failed"] == 1
        
        # Check individual results
        assert result["results"][0]["result"]["success"] is True
        assert result["results"][1]["result"]["success"] is False
        assert result["results"][2]["result"]["success"] is True
    
    async def test_batch_create_embeddings_tool_empty_configs(self):
        """Test batch tool with empty configuration list."""
        from src.mcp_server.tools.create_embeddings_tool import batch_create_embeddings_tool
        
        result = await batch_create_embeddings_tool([])
        
        assert result["success"] is True
        assert result["total_batches"] == 0
        assert result["successful"] == 0
        assert result["failed"] == 0
        assert result["results"] == []
    
    async def test_batch_create_embeddings_tool_exception(self):
        """Test batch tool exception handling."""
        from src.mcp_server.tools.create_embeddings_tool import batch_create_embeddings_tool
        
        # Pass invalid argument to cause exception
        result = await batch_create_embeddings_tool(None)
        
        assert result["success"] is False
        assert "error" in result
        assert result["total_batches"] == 0
    
    def test_tool_metadata_structure(self):
        """Test that tool metadata is properly structured."""
        from src.mcp_server.tools.create_embeddings_tool import TOOL_METADATA
        
        # Check create_embeddings_tool metadata
        create_meta = TOOL_METADATA["create_embeddings_tool"]
        assert create_meta["name"] == "create_embeddings_tool"
        assert "description" in create_meta
        assert "parameters" in create_meta
        
        params = create_meta["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        assert "input_path" in params["required"]
        assert "output_path" in params["required"]
        
        # Check batch_create_embeddings_tool metadata
        batch_meta = TOOL_METADATA["batch_create_embeddings_tool"]
        assert batch_meta["name"] == "batch_create_embeddings_tool"
        assert "batch_configs" in batch_meta["parameters"]["required"]
    
    @patch('src.mcp_server.tools.create_embeddings_tool.CreateEmbeddingsProcessor')
    async def test_create_embeddings_tool_with_custom_params(self, mock_processor_class, temp_dir):
        """Test embedding creation with custom parameters."""
        from src.mcp_server.tools.create_embeddings_tool import create_embeddings_tool
        
        mock_processor = MockCreateEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        input_path = os.path.join(temp_dir, "input.txt")
        create_sample_file(input_path, "Sample text")
        output_path = os.path.join(temp_dir, "output.hdf5")
        
        result = await create_embeddings_tool(
            input_path=input_path,
            output_path=output_path,
            model_name="custom-model",
            batch_size=64,
            chunk_size=5000,
            max_length=512,
            normalize=False,
            use_gpu=True,
            num_workers=4,
            output_format="hdf5",
            compression="lz4",
            metadata={"experiment": "test"}
        )
        
        assert result["success"] is True
        assert result["model_name"] == "custom-model"
        assert result["batch_size"] == 64
        assert result["output_format"] == "hdf5"
    
    @patch('src.mcp_server.tools.create_embeddings_tool.EmbeddingConfig')
    @patch('src.mcp_server.tools.create_embeddings_tool.DataConfig')
    @patch('src.mcp_server.tools.create_embeddings_tool.OutputConfig')
    @patch('src.mcp_server.tools.create_embeddings_tool.CreateEmbeddingsProcessor')
    async def test_create_embeddings_tool_config_objects(self, mock_processor_class, 
                                                        mock_output_config, mock_data_config, 
                                                        mock_embedding_config, temp_dir):
        """Test that configuration objects are created correctly."""
        from src.mcp_server.tools.create_embeddings_tool import create_embeddings_tool
        
        mock_processor = MockCreateEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        input_path = os.path.join(temp_dir, "input.txt")
        create_sample_file(input_path, "Sample text")
        output_path = os.path.join(temp_dir, "output.parquet")
        
        await create_embeddings_tool(
            input_path=input_path,
            output_path=output_path,
            model_name="test-model",
            batch_size=32,
            max_length=256,
            normalize=True,
            use_gpu=False
        )
        
        # Verify config objects were created with correct parameters
        mock_embedding_config.assert_called_once_with(
            model_name="test-model",
            batch_size=32,
            max_length=256,
            normalize=True,
            use_gpu=False
        )
        
        mock_data_config.assert_called_once_with(
            input_path=input_path,
            chunk_size=None,
            num_workers=1
        )
        
        mock_output_config.assert_called_once_with(
            output_path=output_path,
            format="parquet",
            compression=None,
            metadata={}
        )


if __name__ == "__main__":
    pytest.main([__file__])
