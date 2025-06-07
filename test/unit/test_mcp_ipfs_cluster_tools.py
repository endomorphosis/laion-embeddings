import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mcp_server.tools.ipfs_cluster_tools import (
    IPFSClusterTool,
    DistributedVectorTool,
    IPFSMetadataTool
)


class TestIPFSClusterTool:
    """Test IPFS cluster management tool functionality"""
    
    @pytest.fixture
    def mock_ipfs_service(self):
        """Mock IPFS vector service for testing"""
        service = MagicMock()
        service.create_cluster = AsyncMock(return_value={
            "cluster_id": "QmTest123",
            "status": "created",
            "node_count": 3,
            "replication_factor": 2
        })
        service.add_to_cluster = AsyncMock(return_value={
            "success": True,
            "hash": "QmVector456"
        })
        service.get_cluster_status = AsyncMock(return_value={
            "cluster_id": "QmTest123",
            "status": "healthy",
            "nodes": ["node1", "node2", "node3"],
            "total_vectors": 1500
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_ipfs_service):
        """Create tool instance for testing"""
        return IPFSClusterTool(mock_ipfs_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "ipfs_cluster_management"
        assert tool.category == "ipfs_cluster"
        assert "operation" in tool.input_schema["properties"]
        assert "cluster_id" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_create_cluster_success(self, tool, mock_ipfs_service):
        """Test successful cluster creation"""
        parameters = {
            "operation": "create",
            "replication_factor": 2,
            "node_list": ["node1", "node2", "node3"]
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "cluster_id" in result
        assert result["cluster_id"] == "QmTest123"
        mock_ipfs_service.create_cluster.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_vectors_to_cluster(self, tool, mock_ipfs_service):
        """Test adding vectors to cluster"""
        parameters = {
            "operation": "add_vectors",
            "cluster_id": "QmTest123",
            "vectors": [
                {"id": "vec1", "data": [0.1, 0.2, 0.3]},
                {"id": "vec2", "data": [0.4, 0.5, 0.6]}
            ]
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        mock_ipfs_service.add_to_cluster.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cluster_status(self, tool, mock_ipfs_service):
        """Test getting cluster status"""
        parameters = {
            "operation": "status",
            "cluster_id": "QmTest123"
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "status" in result
        assert result["status"] == "healthy"
        assert "total_vectors" in result
        mock_ipfs_service.get_cluster_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self, tool):
        """Test invalid operation handling"""
        parameters = {
            "operation": "invalid_op",
            "cluster_id": "QmTest123"
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "invalid operation" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self, tool, mock_ipfs_service):
        """Test service error handling"""
        mock_ipfs_service.create_cluster.side_effect = Exception("IPFS connection error")
        
        parameters = {
            "operation": "create",
            "replication_factor": 2
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result


class TestDistributedVectorTool:
    """Test distributed vector management tool functionality"""
    
    @pytest.fixture
    def mock_distributed_service(self):
        """Mock distributed vector service for testing"""
        service = MagicMock()
        service.distribute_vectors = AsyncMock(return_value={
            "distribution_id": "dist_123",
            "shard_count": 4,
            "total_vectors": 10000,
            "shards": [
                {"shard_id": "shard_1", "vector_count": 2500, "hash": "QmShard1"},
                {"shard_id": "shard_2", "vector_count": 2500, "hash": "QmShard2"}
            ]
        })
        service.retrieve_distributed_vectors = AsyncMock(return_value={
            "vectors": [
                {"id": "vec1", "data": [0.1, 0.2]},
                {"id": "vec2", "data": [0.3, 0.4]}
            ],
            "total_retrieved": 2
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_distributed_service):
        """Create tool instance for testing"""
        return DistributedVectorTool(mock_distributed_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "distributed_vector_management"
        assert tool.category == "ipfs_cluster"
        assert "operation" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_distribute_vectors_success(self, tool, mock_distributed_service):
        """Test successful vector distribution"""
        parameters = {
            "operation": "distribute",
            "vectors": [{"id": f"vec_{i}", "data": [0.1, 0.2]} for i in range(100)],
            "shard_size": 25,
            "replication_factor": 2
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "distribution_id" in result
        assert "shard_count" in result
        mock_distributed_service.distribute_vectors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_vectors_success(self, tool, mock_distributed_service):
        """Test successful vector retrieval"""
        parameters = {
            "operation": "retrieve",
            "distribution_id": "dist_123",
            "vector_ids": ["vec1", "vec2"]
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "vectors" in result
        assert len(result["vectors"]) == 2
        mock_distributed_service.retrieve_distributed_vectors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_distribute_large_dataset(self, tool, mock_distributed_service):
        """Test distributing large dataset"""
        # Simulate large dataset
        large_vectors = [{"id": f"vec_{i}", "data": [0.1] * 100} for i in range(10000)]
        
        parameters = {
            "operation": "distribute",
            "vectors": large_vectors,
            "shard_size": 1000,
            "replication_factor": 3
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert result["shard_count"] == 4  # 10000/1000 = 10, but mock returns 4


class TestIPFSMetadataTool:
    """Test IPFS metadata management tool functionality"""
    
    @pytest.fixture
    def mock_ipfs_service(self):
        """Mock IPFS vector service for testing"""
        service = MagicMock()
        service.store_metadata = AsyncMock(return_value={
            "metadata_hash": "QmMeta789",
            "stored_at": "2025-06-05T12:00:00Z"
        })
        service.retrieve_metadata = AsyncMock(return_value={
            "metadata": {
                "vector_count": 1000,
                "model": "gte-small",
                "created_at": "2025-06-05T10:00:00Z",
                "tags": ["test", "embedding"]
            },
            "hash": "QmMeta789"
        })
        service.update_metadata = AsyncMock(return_value={
            "success": True,
            "new_hash": "QmMeta790"
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_ipfs_service):
        """Create tool instance for testing"""
        return IPFSMetadataTool(mock_ipfs_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "ipfs_metadata_management"
        assert tool.category == "ipfs_cluster"
        assert "operation" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_store_metadata_success(self, tool, mock_ipfs_service):
        """Test successful metadata storage"""
        parameters = {
            "operation": "store",
            "metadata": {
                "vector_count": 1000,
                "model": "gte-small",
                "dataset": "test_dataset"
            },
            "tags": ["embedding", "test"]
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "metadata_hash" in result
        assert result["metadata_hash"] == "QmMeta789"
        mock_ipfs_service.store_metadata.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_metadata_success(self, tool, mock_ipfs_service):
        """Test successful metadata retrieval"""
        parameters = {
            "operation": "retrieve",
            "metadata_hash": "QmMeta789"
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "metadata" in result
        assert result["metadata"]["vector_count"] == 1000
        mock_ipfs_service.retrieve_metadata.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_metadata_success(self, tool, mock_ipfs_service):
        """Test successful metadata update"""
        parameters = {
            "operation": "update",
            "metadata_hash": "QmMeta789",
            "updates": {
                "vector_count": 1500,
                "last_updated": "2025-06-05T12:00:00Z"
            }
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is True
        assert "new_hash" in result
        mock_ipfs_service.update_metadata.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_metadata_validation_error(self, tool):
        """Test metadata validation error"""
        parameters = {
            "operation": "store",
            "metadata": {},  # Empty metadata should fail validation
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_invalid_hash_format(self, tool):
        """Test invalid hash format handling"""
        parameters = {
            "operation": "retrieve",
            "metadata_hash": "invalid_hash"  # Invalid IPFS hash format
        }
        
        result = await tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "hash" in result["error"].lower()


class TestIPFSIntegration:
    """Integration tests for IPFS cluster tools"""
    
    @pytest.mark.asyncio
    async def test_full_cluster_workflow(self):
        """Test complete cluster creation and vector storage workflow"""
        # Mock services
        ipfs_service = MagicMock()
        distributed_service = MagicMock()
        
        # Setup mock responses
        ipfs_service.create_cluster = AsyncMock(return_value={"cluster_id": "QmTest123"})
        distributed_service.distribute_vectors = AsyncMock(return_value={"distribution_id": "dist_123"})
        ipfs_service.store_metadata = AsyncMock(return_value={"metadata_hash": "QmMeta789"})
        
        # Create tools
        cluster_tool = IPFSClusterTool(ipfs_service)
        vector_tool = DistributedVectorTool(distributed_service)
        metadata_tool = IPFSMetadataTool(ipfs_service)
        
        # Step 1: Create cluster
        cluster_result = await cluster_tool.execute({
            "operation": "create",
            "replication_factor": 2
        })
        assert cluster_result["success"] is True
        
        # Step 2: Distribute vectors
        vector_result = await vector_tool.execute({
            "operation": "distribute",
            "vectors": [{"id": "vec1", "data": [0.1, 0.2]}],
            "shard_size": 100
        })
        assert vector_result["success"] is True
        
        # Step 3: Store metadata
        metadata_result = await metadata_tool.execute({
            "operation": "store",
            "metadata": {"cluster_id": "QmTest123", "vector_count": 1}
        })
        assert metadata_result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
