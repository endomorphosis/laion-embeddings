"""
Tests for shard embeddings MCP tools.
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
    MockShardEmbeddingsProcessor, create_sample_file, create_sample_embeddings_file,
    TEST_SHARD_SIZE, sample_embeddings
)


@pytest.mark.asyncio
class TestShardEmbeddingsTools:
    """Test suite for shard embeddings MCP tools."""
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_shard_embeddings_tool_success(self, mock_processor_class, temp_dir, sample_embeddings):
        """Test successful embedding sharding."""
        from src.mcp_server.tools.shard_embeddings_tool import shard_embeddings_tool
        
        # Setup mock processor
        mock_processor = MockShardEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        # Create test input file
        input_path = os.path.join(temp_dir, "embeddings.npy")
        create_sample_embeddings_file(input_path, sample_embeddings)
        
        output_dir = os.path.join(temp_dir, "shards")
        
        # Execute tool
        result = await shard_embeddings_tool(
            input_path=input_path,
            output_dir=output_dir,
            shard_size=TEST_SHARD_SIZE,
            compression="gzip",
            validate_shards=True
        )
        
        # Verify result
        assert result["success"] is True
        assert result["input_path"] == input_path
        assert result["output_dir"] == output_dir
        assert result["shard_size"] == TEST_SHARD_SIZE
        assert result["total_shards"] == 5
        assert result["total_embeddings"] == 5000
        assert result["compression"] == "gzip"
        assert "validation" in result
        assert result["validation"]["valid"] is True
        assert len(result["shard_files"]) == 2
        
        # Verify processor was called correctly
        mock_processor_class.assert_called_once()
    
    async def test_shard_embeddings_tool_invalid_input(self, temp_dir):
        """Test sharding with invalid input path."""
        from src.mcp_server.tools.shard_embeddings_tool import shard_embeddings_tool
        
        invalid_path = os.path.join(temp_dir, "nonexistent.npy")
        output_dir = os.path.join(temp_dir, "shards")
        
        result = await shard_embeddings_tool(
            input_path=invalid_path,
            output_dir=output_dir
        )
        
        # Verify error handling
        assert result["success"] is False
        assert "does not exist" in result["error"]
        assert result["input_path"] == invalid_path
        assert result["output_dir"] == output_dir
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_shard_embeddings_tool_with_options(self, mock_processor_class, temp_dir, sample_embeddings):
        """Test sharding with custom options."""
        from src.mcp_server.tools.shard_embeddings_tool import shard_embeddings_tool
        
        mock_processor = MockShardEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        input_path = os.path.join(temp_dir, "embeddings.npy")
        create_sample_embeddings_file(input_path, sample_embeddings)
        output_dir = os.path.join(temp_dir, "shards")
        
        result = await shard_embeddings_tool(
            input_path=input_path,
            output_dir=output_dir,
            shard_size=2000,
            max_shards=10,
            overlap_size=100,
            shuffle=True,
            seed=42,
            compression="lz4",
            output_format="hdf5",
            metadata={"experiment": "test"},
            validate_shards=False
        )
        
        assert result["success"] is True
        assert result["shard_size"] == 2000
        assert result["compression"] == "lz4"
        assert result["output_format"] == "hdf5"
        assert "validation" not in result
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_merge_shards_tool_success(self, mock_processor_class, temp_dir):
        """Test successful shard merging."""
        from src.mcp_server.tools.shard_embeddings_tool import merge_shards_tool
        
        mock_processor = MockShardEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        # Create test shard directory
        shard_dir = os.path.join(temp_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        
        # Create sample shard files
        for i in range(3):
            shard_file = os.path.join(shard_dir, f"shard_{i}.parquet")
            create_sample_file(shard_file, f"shard data {i}")
        
        output_path = os.path.join(temp_dir, "merged.parquet")
        
        # Execute tool
        result = await merge_shards_tool(
            shard_dir=shard_dir,
            output_path=output_path,
            shard_pattern="shard_*.parquet",
            validate_merge=True,
            remove_shards=False
        )
        
        # Verify result
        assert result["success"] is True
        assert result["shard_dir"] == shard_dir
        assert result["output_path"] == output_path
        assert result["shards_processed"] == 5
        assert result["total_embeddings"] == 5000
        assert result["output_size"] == 2048000
        assert result["shards_removed"] is False
        assert "validation" in result
        assert result["validation"]["valid"] is True
    
    async def test_merge_shards_tool_invalid_directory(self, temp_dir):
        """Test merging with invalid shard directory."""
        from src.mcp_server.tools.shard_embeddings_tool import merge_shards_tool
        
        invalid_dir = os.path.join(temp_dir, "nonexistent")
        output_path = os.path.join(temp_dir, "merged.parquet")
        
        result = await merge_shards_tool(
            shard_dir=invalid_dir,
            output_path=output_path
        )
        
        # Verify error handling
        assert result["success"] is False
        assert "does not exist" in result["error"]
        assert result["shard_dir"] == invalid_dir
        assert result["output_path"] == output_path
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_merge_shards_tool_with_cleanup(self, mock_processor_class, temp_dir):
        """Test shard merging with cleanup option."""
        from src.mcp_server.tools.shard_embeddings_tool import merge_shards_tool
        
        mock_processor = MockShardEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        shard_dir = os.path.join(temp_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "merged.parquet")
        
        result = await merge_shards_tool(
            shard_dir=shard_dir,
            output_path=output_path,
            remove_shards=True
        )
        
        assert result["success"] is True
        assert result["shards_removed"] is True
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_shard_info_tool_success(self, mock_processor_class, temp_dir):
        """Test successful shard info retrieval."""
        from src.mcp_server.tools.shard_embeddings_tool import shard_info_tool
        
        mock_processor = MockShardEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        # Create test shard directory
        shard_path = os.path.join(temp_dir, "shards")
        os.makedirs(shard_path, exist_ok=True)
        
        result = await shard_info_tool(shard_path=shard_path)
        
        # Verify result
        assert result["success"] is True
        assert result["shard_path"] == shard_path
        assert "info" in result
        assert result["info"]["shard_count"] == 5
        assert result["info"]["total_embeddings"] == 5000
        assert result["info"]["shard_size"] == 1000
        assert result["info"]["format"] == "parquet"
    
    async def test_shard_info_tool_invalid_path(self, temp_dir):
        """Test shard info with invalid path."""
        from src.mcp_server.tools.shard_embeddings_tool import shard_info_tool
        
        invalid_path = os.path.join(temp_dir, "nonexistent")
        
        result = await shard_info_tool(shard_path=invalid_path)
        
        # Verify error handling
        assert result["success"] is False
        assert "does not exist" in result["error"]
        assert result["shard_path"] == invalid_path
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_shard_embeddings_tool_exception(self, mock_processor_class, temp_dir, sample_embeddings):
        """Test handling of processor exceptions."""
        from src.mcp_server.tools.shard_embeddings_tool import shard_embeddings_tool
        
        # Setup mock to raise exception
        mock_processor_class.side_effect = Exception("Sharding failed")
        
        input_path = os.path.join(temp_dir, "embeddings.npy")
        create_sample_embeddings_file(input_path, sample_embeddings)
        output_dir = os.path.join(temp_dir, "shards")
        
        result = await shard_embeddings_tool(
            input_path=input_path,
            output_dir=output_dir
        )
        
        # Verify error handling
        assert result["success"] is False
        assert "Sharding failed" in result["error"]
    
    def test_tool_metadata_structure(self):
        """Test that tool metadata is properly structured."""
        from src.mcp_server.tools.shard_embeddings_tool import TOOL_METADATA
        
        # Check shard_embeddings_tool metadata
        shard_meta = TOOL_METADATA["shard_embeddings_tool"]
        assert shard_meta["name"] == "shard_embeddings_tool"
        assert "description" in shard_meta
        assert "parameters" in shard_meta
        
        params = shard_meta["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        assert "input_path" in params["required"]
        assert "output_dir" in params["required"]
        
        # Check default values
        properties = params["properties"]
        assert properties["shard_size"]["default"] == 1000000
        assert properties["overlap_size"]["default"] == 0
        assert properties["shuffle"]["default"] is False
        assert properties["compression"]["default"] == "gzip"
        assert properties["output_format"]["default"] == "parquet"
        assert properties["validate_shards"]["default"] is True
        
        # Check merge_shards_tool metadata
        merge_meta = TOOL_METADATA["merge_shards_tool"]
        assert merge_meta["name"] == "merge_shards_tool"
        assert "shard_dir" in merge_meta["parameters"]["required"]
        assert "output_path" in merge_meta["parameters"]["required"]
        
        # Check shard_info_tool metadata
        info_meta = TOOL_METADATA["shard_info_tool"]
        assert info_meta["name"] == "shard_info_tool"
        assert "shard_path" in info_meta["parameters"]["required"]
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardConfig')
    @patch('src.mcp_server.tools.shard_embeddings_tool.InputConfig')
    @patch('src.mcp_server.tools.shard_embeddings_tool.OutputConfig')
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_shard_embeddings_tool_config_objects(self, mock_processor_class, 
                                                       mock_output_config, mock_input_config, 
                                                       mock_shard_config, temp_dir, sample_embeddings):
        """Test that configuration objects are created correctly."""
        from src.mcp_server.tools.shard_embeddings_tool import shard_embeddings_tool
        
        mock_processor = MockShardEmbeddingsProcessor(None, None, None)
        mock_processor_class.return_value = mock_processor
        
        input_path = os.path.join(temp_dir, "embeddings.npy")
        create_sample_embeddings_file(input_path, sample_embeddings)
        output_dir = os.path.join(temp_dir, "shards")
        
        await shard_embeddings_tool(
            input_path=input_path,
            output_dir=output_dir,
            shard_size=2000,
            max_shards=5,
            overlap_size=50,
            shuffle=True,
            seed=123
        )
        
        # Verify config objects were created with correct parameters
        mock_shard_config.assert_called_once_with(
            shard_size=2000,
            max_shards=5,
            overlap_size=50,
            shuffle=True,
            seed=123
        )
        
        mock_input_config.assert_called_once_with(
            input_path=input_path
        )
        
        mock_output_config.assert_called_once_with(
            output_dir=output_dir,
            compression="gzip",
            format="parquet",
            metadata={}
        )
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_merge_shards_tool_exception_handling(self, mock_processor_class, temp_dir):
        """Test exception handling in merge shards tool."""
        from src.mcp_server.tools.shard_embeddings_tool import merge_shards_tool
        
        # Setup mock to raise exception during merge
        mock_processor = MockShardEmbeddingsProcessor(None, None, None)
        mock_processor.merge_shards = Mock(side_effect=Exception("Merge failed"))
        mock_processor_class.return_value = mock_processor
        
        shard_dir = os.path.join(temp_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "merged.parquet")
        
        result = await merge_shards_tool(
            shard_dir=shard_dir,
            output_path=output_path
        )
        
        assert result["success"] is False
        assert "Merge failed" in result["error"]
    
    @patch('src.mcp_server.tools.shard_embeddings_tool.ShardEmbeddingsProcessor')
    async def test_shard_info_tool_exception_handling(self, mock_processor_class, temp_dir):
        """Test exception handling in shard info tool."""
        from src.mcp_server.tools.shard_embeddings_tool import shard_info_tool
        
        # Setup mock to raise exception
        mock_processor = MockShardEmbeddingsProcessor(None, None, None)
        mock_processor.get_shard_info = Mock(side_effect=Exception("Info retrieval failed"))
        mock_processor_class.return_value = mock_processor
        
        shard_path = os.path.join(temp_dir, "shards")
        os.makedirs(shard_path, exist_ok=True)
        
        result = await shard_info_tool(shard_path=shard_path)
        
        assert result["success"] is False
        assert "Info retrieval failed" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__])
