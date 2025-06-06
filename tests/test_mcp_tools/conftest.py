"""
Test configuration and utilities for MCP tools testing.
"""

import pytest
import asyncio
import tempfile
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Test fixtures and utilities
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    import numpy as np
    return np.random.rand(100, 384).tolist()

@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    return [{"id": i, "text": f"sample text {i}"} for i in range(100)]

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    service = Mock()
    service.create_embeddings = AsyncMock(return_value={"embeddings": [[0.1, 0.2, 0.3]], "success": True})
    service.search = AsyncMock(return_value={"results": [], "success": True})
    return service

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = Mock()
    store.add = AsyncMock(return_value={"success": True, "count": 100})
    store.search = AsyncMock(return_value={"results": [], "success": True})
    store.get_stats = AsyncMock(return_value={"count": 100, "dimension": 384})
    store.get_info = AsyncMock(return_value={"store_id": "test", "provider": "mock"})
    store.get_capabilities = Mock(return_value=["add", "search", "delete"])
    store.delete = AsyncMock(return_value={"success": True, "deleted": 10})
    store.optimize = AsyncMock(return_value={"success": True, "optimized": True})
    store.load_embeddings_from_file = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return store

@pytest.fixture
def sample_dataset_data():
    """Sample dataset data for testing."""
    return {
        "data": [
            {"text": "This is sample text 1", "id": 1},
            {"text": "This is sample text 2", "id": 2},
            {"text": "This is sample text 3", "id": 3}
        ]
    }

@pytest.fixture
def sample_workflow_definition():
    """Sample workflow definition for testing."""
    return {
        "name": "test_workflow",
        "description": "Test workflow for embeddings",
        "steps": [
            {
                "type": "create_embeddings",
                "parameters": {
                    "input_path": "/test/input",
                    "output_path": "/test/output",
                    "model_name": "test-model"
                }
            },
            {
                "type": "shard_embeddings", 
                "parameters": {
                    "input_path": "/test/output",
                    "output_dir": "/test/shards",
                    "shard_size": 1000
                }
            }
        ],
        "continue_on_error": False
    }

class MockVectorStoreFactory:
    """Mock factory for creating vector stores."""
    
    @staticmethod
    def create(provider: str, config: Dict[str, Any]):
        """Create a mock vector store."""
        mock_store = Mock()
        mock_store.add = AsyncMock(return_value={"success": True, "count": 100})
        mock_store.search = AsyncMock(return_value={
            "results": [
                {"id": "1", "score": 0.95, "metadata": {"text": "sample"}},
                {"id": "2", "score": 0.85, "metadata": {"text": "example"}}
            ],
            "success": True
        })
        mock_store.get_stats = AsyncMock(return_value={
            "count": 1000,
            "dimension": 384,
            "provider": provider
        })
        mock_store.get_info = AsyncMock(return_value={
            "store_id": "test_store",
            "provider": provider,
            "config": config
        })
        mock_store.get_capabilities = Mock(return_value=["add", "search", "delete", "optimize"])
        mock_store.delete = AsyncMock(return_value={"success": True, "deleted": 5})
        mock_store.optimize = AsyncMock(return_value={"success": True, "optimized": True})
        mock_store.load_embeddings_from_file = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        return mock_store

class MockCreateEmbeddingsProcessor:
    """Mock processor for create embeddings."""
    
    def __init__(self, embedding_config, data_config, output_config):
        self.embedding_config = embedding_config
        self.data_config = data_config
        self.output_config = output_config
    
    def process(self):
        """Mock process method."""
        return {
            "embeddings_count": 100,
            "processing_time": 10.5,
            "output_size": 1024000,
            "success": True
        }

class MockShardEmbeddingsProcessor:
    """Mock processor for shard embeddings."""
    
    def __init__(self, shard_config, input_config, output_config):
        self.shard_config = shard_config
        self.input_config = input_config
        self.output_config = output_config
    
    def process(self):
        """Mock process method."""
        return {
            "total_shards": 5,
            "total_embeddings": 5000,
            "shard_files": ["shard_0.parquet", "shard_1.parquet"],
            "success": True
        }
    
    def validate_shards(self):
        """Mock validation method."""
        return {"valid": True, "all_shards_ok": True}
    
    def merge_shards(self, shard_dir, output_path, pattern):
        """Mock merge method."""
        return {
            "success": True,
            "shards_processed": 5,
            "total_embeddings": 5000,
            "output_size": 2048000
        }
    
    def validate_merged_file(self, output_path):
        """Mock merged file validation."""
        return {"valid": True, "embeddings_count": 5000}
    
    def cleanup_shards(self, shard_dir, pattern):
        """Mock cleanup method."""
        return {"cleaned": True, "files_removed": 5}
    
    def get_shard_info(self, shard_path):
        """Mock shard info method."""
        return {
            "shard_count": 5,
            "total_embeddings": 5000,
            "shard_size": 1000,
            "format": "parquet"
        }

class MockIPFSClusterIndex:
    """Mock IPFS cluster index."""
    
    def __init__(self, config):
        self.config = config
    
    def index_embeddings(self, **kwargs):
        """Mock index embeddings method."""
        return {
            "success": True,
            "indexed_files": 5,
            "total_size": 1024000,
            "cluster_id": "test_cluster"
        }

# Utility functions
def create_sample_file(file_path: str, content: str) -> None:
    """Create a sample file for testing."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)

def create_sample_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """Create a sample JSON file for testing."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f)

def create_sample_embeddings_file(file_path: str, embeddings: List[List[float]]) -> None:
    """Create a sample embeddings file for testing."""
    import numpy as np
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, np.array(embeddings))

async def run_async_test(coro):
    """Run an async test function."""
    return await coro

# Constants for testing
TEST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEST_BATCH_SIZE = 32
TEST_EMBEDDING_DIM = 384
TEST_SHARD_SIZE = 1000

# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
