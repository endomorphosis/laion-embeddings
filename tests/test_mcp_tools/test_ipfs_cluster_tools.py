"""
Comprehensive tests for IPFS cluster MCP tools.
"""

import pytest
import asyncio
import tempfile
import json
import hashlib
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Import the IPFS cluster tools
from src.mcp_server.tools.ipfs_cluster_tools import (
    IPFSClusterManagementTool,
    StorachaIntegrationTool,
    IPFSPinningTool
)


class TestIPFSClusterManagementTool:
    """Test IPFSClusterManagementTool class."""
    
    @pytest.fixture
    def cluster_management_tool(self, mock_embedding_service):
        """Create an IPFS cluster management tool for testing."""
        return IPFSClusterManagementTool(mock_embedding_service)
    
    @pytest.mark.asyncio
    async def test_initialize_cluster(self, cluster_management_tool, temp_dir):
        """Test initializing an IPFS cluster."""
        parameters = {
            "action": "initialize",
            "cluster_name": "test_cluster",
            "cluster_config": {
                "listen_multiaddress": "/ip4/0.0.0.0/tcp/9096",
                "secret": "test_secret_key",
                "replication_factor": 3
            },
            "data_dir": str(Path(temp_dir) / "cluster_data")
        }
        
        result = await cluster_management_tool.execute(parameters)
        
        assert result["success"] is True
        assert "cluster_id" in result
        assert result["cluster_name"] == "test_cluster"
        assert "peer_id" in result
        assert "addresses" in result
        assert result["status"] == "initialized"
    
    @pytest.mark.asyncio
    async def test_start_cluster(self, cluster_management_tool, temp_dir):
        """Test starting an IPFS cluster."""
        # First initialize a cluster
        init_params = {
            "action": "initialize",
            "cluster_name": "start_test_cluster",
            "data_dir": str(Path(temp_dir) / "start_cluster_data")
        }
        
        init_result = await cluster_management_tool.execute(init_params)
        cluster_id = init_result["cluster_id"]
        
        # Start the cluster
        start_params = {
            "action": "start",
            "cluster_id": cluster_id
        }
        
        result = await cluster_management_tool.execute(start_params)
        
        assert result["success"] is True
        assert result["cluster_id"] == cluster_id
        assert result["status"] == "running"
        assert "start_time" in result
        assert "peers" in result
    
    @pytest.mark.asyncio
    async def test_stop_cluster(self, cluster_management_tool, temp_dir):
        """Test stopping an IPFS cluster."""
        # Initialize and start a cluster
        init_params = {
            "action": "initialize",
            "cluster_name": "stop_test_cluster",
            "data_dir": str(Path(temp_dir) / "stop_cluster_data")
        }
        
        init_result = await cluster_management_tool.execute(init_params)
        cluster_id = init_result["cluster_id"]
        
        start_params = {
            "action": "start",
            "cluster_id": cluster_id
        }
        
        await cluster_management_tool.execute(start_params)
        
        # Stop the cluster
        stop_params = {
            "action": "stop",
            "cluster_id": cluster_id
        }
        
        result = await cluster_management_tool.execute(stop_params)
        
        assert result["success"] is True
        assert result["cluster_id"] == cluster_id
        assert result["status"] == "stopped"
        assert "stop_time" in result
    
    @pytest.mark.asyncio
    async def test_get_cluster_status(self, cluster_management_tool, temp_dir):
        """Test getting cluster status."""
        # Initialize a cluster
        init_params = {
            "action": "initialize",
            "cluster_name": "status_test_cluster",
            "data_dir": str(Path(temp_dir) / "status_cluster_data")
        }
        
        init_result = await cluster_management_tool.execute(init_params)
        cluster_id = init_result["cluster_id"]
        
        # Get status
        status_params = {
            "action": "status",
            "cluster_id": cluster_id
        }
        
        result = await cluster_management_tool.execute(status_params)
        
        assert result["success"] is True
        assert "cluster_info" in result
        assert "peers" in result["cluster_info"]
        assert "pinset_size" in result["cluster_info"]
        assert "status" in result["cluster_info"]
    
    @pytest.mark.asyncio
    async def test_add_peer_to_cluster(self, cluster_management_tool, temp_dir):
        """Test adding a peer to the cluster."""
        # Initialize a cluster
        init_params = {
            "action": "initialize",
            "cluster_name": "peer_test_cluster",
            "data_dir": str(Path(temp_dir) / "peer_cluster_data")
        }
        
        init_result = await cluster_management_tool.execute(init_params)
        cluster_id = init_result["cluster_id"]
        
        # Add peer
        add_peer_params = {
            "action": "add_peer",
            "cluster_id": cluster_id,
            "peer_multiaddress": "/ip4/192.168.1.100/tcp/9096/p2p/QmTestPeerID"
        }
        
        result = await cluster_management_tool.execute(add_peer_params)
        
        assert result["success"] is True
        assert "peer_id" in result
        assert "connected" in result
    
    @pytest.mark.asyncio
    async def test_remove_peer_from_cluster(self, cluster_management_tool, temp_dir):
        """Test removing a peer from the cluster."""
        # Initialize cluster and add peer
        init_params = {
            "action": "initialize",
            "cluster_name": "remove_peer_cluster",
            "data_dir": str(Path(temp_dir) / "remove_peer_data")
        }
        
        init_result = await cluster_management_tool.execute(init_params)
        cluster_id = init_result["cluster_id"]
        
        # Remove peer
        remove_peer_params = {
            "action": "remove_peer",
            "cluster_id": cluster_id,
            "peer_id": "QmTestPeerToRemove"
        }
        
        result = await cluster_management_tool.execute(remove_peer_params)
        
        assert result["success"] is True
        assert "peer_id" in result
        assert "removed" in result
    
    @pytest.mark.asyncio
    async def test_list_clusters(self, cluster_management_tool):
        """Test listing all clusters."""
        parameters = {
            "action": "list"
        }
        
        result = await cluster_management_tool.execute(parameters)
        
        assert result["success"] is True
        assert "clusters" in result
        assert isinstance(result["clusters"], list)
        
        for cluster in result["clusters"]:
            assert "cluster_id" in cluster
            assert "name" in cluster
            assert "status" in cluster
    
    @pytest.mark.asyncio
    async def test_cluster_health_check(self, cluster_management_tool, temp_dir):
        """Test cluster health check."""
        # Initialize a cluster
        init_params = {
            "action": "initialize",
            "cluster_name": "health_test_cluster",
            "data_dir": str(Path(temp_dir) / "health_cluster_data")
        }
        
        init_result = await cluster_management_tool.execute(init_params)
        cluster_id = init_result["cluster_id"]
        
        # Health check
        health_params = {
            "action": "health",
            "cluster_id": cluster_id
        }
        
        result = await cluster_management_tool.execute(health_params)
        
        assert result["success"] is True
        assert "health_status" in result
        assert "peer_health" in result
        assert "connectivity" in result
    
    @pytest.mark.asyncio
    async def test_invalid_cluster_action(self, cluster_management_tool):
        """Test invalid cluster action."""
        parameters = {
            "action": "invalid_action",
            "cluster_id": "test_cluster"
        }
        
        result = await cluster_management_tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "action" in result["error"].lower()


class TestStorachaIntegrationTool:
    """Test StorachaIntegrationTool class."""
    
    @pytest.fixture
    def storacha_tool(self, mock_embedding_service):
        """Create a Storacha integration tool for testing."""
        return StorachaIntegrationTool(mock_embedding_service)
    
    @pytest.mark.asyncio
    async def test_configure_storacha_connection(self, storacha_tool):
        """Test configuring Storacha connection."""
        parameters = {
            "action": "configure",
            "storacha_config": {
                "api_endpoint": "https://api.storacha.network",
                "space_did": "did:key:test_space_did",
                "proof": "test_proof_token"
            }
        }
        
        result = await storacha_tool.execute(parameters)
        
        assert result["success"] is True
        assert "connection_id" in result
        assert "space_info" in result
        assert result["status"] == "connected"
    
    @pytest.mark.asyncio
    async def test_upload_to_storacha(self, storacha_tool, temp_dir):
        """Test uploading data to Storacha."""
        # Create test file
        test_file = Path(temp_dir) / "test_upload.txt"
        test_content = "Test content for Storacha upload"
        test_file.write_text(test_content)
        
        parameters = {
            "action": "upload",
            "file_path": str(test_file),
            "metadata": {
                "name": "test_upload.txt",
                "description": "Test file for upload",
                "tags": ["test", "upload"]
            }
        }
        
        result = await storacha_tool.execute(parameters)
        
        assert result["success"] is True
        assert "cid" in result
        assert "upload_info" in result
        assert "size" in result["upload_info"]
        assert "receipt" in result
    
    @pytest.mark.asyncio
    async def test_upload_dataset_to_storacha(self, storacha_tool, temp_dir):
        """Test uploading a dataset to Storacha."""
        # Create test dataset files
        dataset_dir = Path(temp_dir) / "test_dataset"
        dataset_dir.mkdir()
        
        (dataset_dir / "embeddings.npz").write_text("mock embeddings data")
        (dataset_dir / "metadata.json").write_text('{"test": "metadata"}')
        (dataset_dir / "index.faiss").write_text("mock faiss index")
        
        parameters = {
            "action": "upload_dataset",
            "dataset_path": str(dataset_dir),
            "dataset_metadata": {
                "name": "test_embedding_dataset",
                "version": "1.0",
                "description": "Test dataset for Storacha",
                "model": "test-model",
                "dimension": 384
            }
        }
        
        result = await storacha_tool.execute(parameters)
        
        assert result["success"] is True
        assert "dataset_cid" in result
        assert "manifest" in result
        assert "component_cids" in result
    
    @pytest.mark.asyncio
    async def test_retrieve_from_storacha(self, storacha_tool, temp_dir):
        """Test retrieving data from Storacha."""
        parameters = {
            "action": "retrieve",
            "cid": "bafybeitest123456789",
            "output_path": str(Path(temp_dir) / "retrieved_file")
        }
        
        result = await storacha_tool.execute(parameters)
        
        assert result["success"] is True
        assert "local_path" in result
        assert "retrieved_size" in result
        assert "verification" in result
    
    @pytest.mark.asyncio
    async def test_list_storacha_uploads(self, storacha_tool):
        """Test listing uploads in Storacha space."""
        parameters = {
            "action": "list",
            "space_did": "did:key:test_space_did",
            "limit": 50
        }
        
        result = await storacha_tool.execute(parameters)
        
        assert result["success"] is True
        assert "uploads" in result
        assert "total_count" in result
        
        for upload in result["uploads"]:
            assert "cid" in upload
            assert "uploaded_at" in upload
            assert "size" in upload
    
    @pytest.mark.asyncio
    async def test_get_storacha_space_info(self, storacha_tool):
        """Test getting Storacha space information."""
        parameters = {
            "action": "space_info",
            "space_did": "did:key:test_space_did"
        }
        
        result = await storacha_tool.execute(parameters)
        
        assert result["success"] is True
        assert "space_info" in result
        assert "usage" in result["space_info"]
        assert "limits" in result["space_info"]
        assert "created_at" in result["space_info"]
    
    @pytest.mark.asyncio
    async def test_create_storacha_car_file(self, storacha_tool, temp_dir):
        """Test creating CAR file for Storacha."""
        # Create test files
        test_dir = Path(temp_dir) / "car_test"
        test_dir.mkdir()
        
        (test_dir / "file1.txt").write_text("Content 1")
        (test_dir / "file2.txt").write_text("Content 2")
        
        parameters = {
            "action": "create_car",
            "source_path": str(test_dir),
            "car_output_path": str(Path(temp_dir) / "test.car"),
            "wrap_with_directory": True
        }
        
        result = await storacha_tool.execute(parameters)
        
        assert result["success"] is True
        assert "car_path" in result
        assert "root_cid" in result
        assert "car_size" in result
        assert "file_count" in result
    
    @pytest.mark.asyncio
    async def test_storacha_invalid_action(self, storacha_tool):
        """Test invalid Storacha action."""
        parameters = {
            "action": "invalid_storacha_action"
        }
        
        result = await storacha_tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "action" in result["error"].lower()


class TestIPFSPinningTool:
    """Test IPFSPinningTool class."""
    
    @pytest.fixture
    def pinning_tool(self, mock_embedding_service):
        """Create an IPFS pinning tool for testing."""
        return IPFSPinningTool(mock_embedding_service)
    
    @pytest.mark.asyncio
    async def test_pin_cid(self, pinning_tool):
        """Test pinning a CID."""
        parameters = {
            "action": "pin",
            "cid": "QmTestCID123456789",
            "recursive": True,
            "metadata": {
                "name": "test_pin",
                "description": "Test pinning operation"
            }
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is True
        assert result["cid"] == "QmTestCID123456789"
        assert result["pinned"] is True
        assert "pin_time" in result
        assert "size" in result
    
    @pytest.mark.asyncio
    async def test_pin_file(self, pinning_tool, temp_dir):
        """Test pinning a file."""
        # Create test file
        test_file = Path(temp_dir) / "pin_test.txt"
        test_content = "Content to be pinned"
        test_file.write_text(test_content)
        
        parameters = {
            "action": "pin_file",
            "file_path": str(test_file),
            "wrap_with_directory": False
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is True
        assert "cid" in result
        assert result["pinned"] is True
        assert "file_info" in result
    
    @pytest.mark.asyncio
    async def test_pin_directory(self, pinning_tool, temp_dir):
        """Test pinning a directory."""
        # Create test directory with files
        test_dir = Path(temp_dir) / "pin_dir"
        test_dir.mkdir()
        
        (test_dir / "file1.txt").write_text("File 1 content")
        (test_dir / "file2.txt").write_text("File 2 content")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "file3.txt").write_text("File 3 content")
        
        parameters = {
            "action": "pin_directory",
            "directory_path": str(test_dir),
            "recursive": True
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is True
        assert "root_cid" in result
        assert result["pinned"] is True
        assert "file_count" in result
        assert "total_size" in result
    
    @pytest.mark.asyncio
    async def test_unpin_cid(self, pinning_tool):
        """Test unpinning a CID."""
        # First pin something
        pin_params = {
            "action": "pin",
            "cid": "QmTestUnpinCID123"
        }
        
        pin_result = await pinning_tool.execute(pin_params)
        assert pin_result["success"] is True
        
        # Then unpin it
        unpin_params = {
            "action": "unpin",
            "cid": "QmTestUnpinCID123"
        }
        
        result = await pinning_tool.execute(unpin_params)
        
        assert result["success"] is True
        assert result["cid"] == "QmTestUnpinCID123"
        assert result["unpinned"] is True
        assert "unpin_time" in result
    
    @pytest.mark.asyncio
    async def test_list_pins(self, pinning_tool):
        """Test listing pinned CIDs."""
        parameters = {
            "action": "list",
            "pin_type": "recursive",
            "limit": 100
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is True
        assert "pins" in result
        assert "total_count" in result
        
        for pin in result["pins"]:
            assert "cid" in pin
            assert "type" in pin
            assert "pinned_at" in pin
    
    @pytest.mark.asyncio
    async def test_get_pin_status(self, pinning_tool):
        """Test getting pin status."""
        parameters = {
            "action": "status",
            "cid": "QmTestStatusCID123"
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is True
        assert "cid" in result
        assert "pinned" in result
        assert "pin_type" in result
        if result["pinned"]:
            assert "pinned_at" in result
            assert "size" in result
    
    @pytest.mark.asyncio
    async def test_pin_with_cluster(self, pinning_tool):
        """Test pinning with cluster replication."""
        parameters = {
            "action": "pin",
            "cid": "QmClusterPinTest123",
            "cluster_pinning": {
                "enabled": True,
                "replication_factor": 3,
                "cluster_name": "test_cluster"
            }
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is True
        assert "cluster_info" in result
        assert "replicated_peers" in result["cluster_info"]
    
    @pytest.mark.asyncio
    async def test_pin_with_metadata(self, pinning_tool):
        """Test pinning with rich metadata."""
        parameters = {
            "action": "pin",
            "cid": "QmMetadataPinTest123",
            "metadata": {
                "name": "Test Dataset",
                "version": "1.0",
                "tags": ["embedding", "test", "dataset"],
                "model_info": {
                    "name": "test-model",
                    "dimension": 384
                },
                "created_by": "test_user",
                "license": "MIT"
            }
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is True
        assert "metadata_stored" in result
        assert result["metadata_stored"] is True
    
    @pytest.mark.asyncio
    async def test_bulk_pin_operations(self, pinning_tool):
        """Test bulk pinning operations."""
        cids_to_pin = [
            "QmBulkPin1",
            "QmBulkPin2", 
            "QmBulkPin3"
        ]
        
        parameters = {
            "action": "bulk_pin",
            "cids": cids_to_pin,
            "recursive": True
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) == len(cids_to_pin)
        assert "successful_pins" in result
        assert "failed_pins" in result
    
    @pytest.mark.asyncio
    async def test_pinning_invalid_action(self, pinning_tool):
        """Test invalid pinning action."""
        parameters = {
            "action": "invalid_pin_action"
        }
        
        result = await pinning_tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "action" in result["error"].lower()


class TestIPFSClusterToolsIntegration:
    """Integration tests for IPFS cluster tools."""
    
    @pytest.mark.asyncio
    async def test_complete_ipfs_workflow(self, mock_embedding_service, temp_dir):
        """Test complete IPFS workflow with cluster, pinning, and Storacha."""
        # Initialize tools
        cluster_tool = IPFSClusterManagementTool(mock_embedding_service)
        pinning_tool = IPFSPinningTool(mock_embedding_service)
        storacha_tool = StorachaIntegrationTool(mock_embedding_service)
        
        # 1. Initialize cluster
        init_params = {
            "action": "initialize",
            "cluster_name": "integration_cluster",
            "data_dir": str(Path(temp_dir) / "cluster_data")
        }
        
        cluster_result = await cluster_tool.execute(init_params)
        assert cluster_result["success"] is True
        cluster_id = cluster_result["cluster_id"]
        
        # 2. Start cluster
        start_params = {
            "action": "start",
            "cluster_id": cluster_id
        }
        
        start_result = await cluster_tool.execute(start_params)
        assert start_result["success"] is True
        
        # 3. Create and pin a test file
        test_file = Path(temp_dir) / "integration_test.txt"
        test_file.write_text("Integration test content")
        
        pin_params = {
            "action": "pin_file",
            "file_path": str(test_file)
        }
        
        pin_result = await pinning_tool.execute(pin_params)
        assert pin_result["success"] is True
        test_cid = pin_result["cid"]
        
        # 4. Upload to Storacha
        upload_params = {
            "action": "upload",
            "file_path": str(test_file),
            "metadata": {"integration_test": True}
        }
        
        upload_result = await storacha_tool.execute(upload_params)
        assert upload_result["success"] is True
        
        # 5. Verify cluster status
        status_params = {
            "action": "status",
            "cluster_id": cluster_id
        }
        
        status_result = await cluster_tool.execute(status_params)
        assert status_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_embedding_dataset_storage_workflow(self, mock_embedding_service, temp_dir):
        """Test storing embedding dataset through IPFS and Storacha."""
        pinning_tool = IPFSPinningTool(mock_embedding_service)
        storacha_tool = StorachaIntegrationTool(mock_embedding_service)
        
        # Create mock embedding dataset
        dataset_dir = Path(temp_dir) / "embedding_dataset"
        dataset_dir.mkdir()
        
        # Mock files
        (dataset_dir / "embeddings.npz").write_text("mock embeddings binary data")
        (dataset_dir / "metadata.json").write_text(json.dumps({
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "count": 1000,
            "created_at": "2024-01-01T00:00:00Z"
        }))
        (dataset_dir / "index.faiss").write_text("mock faiss index binary")
        
        # 1. Pin the dataset directory
        pin_params = {
            "action": "pin_directory",
            "directory_path": str(dataset_dir),
            "metadata": {
                "type": "embedding_dataset",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "version": "1.0"
            }
        }
        
        pin_result = await pinning_tool.execute(pin_params)
        assert pin_result["success"] is True
        dataset_cid = pin_result["root_cid"]
        
        # 2. Upload dataset to Storacha
        upload_params = {
            "action": "upload_dataset",
            "dataset_path": str(dataset_dir),
            "dataset_metadata": {
                "name": "test_embedding_dataset",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Test embedding dataset for integration"
            }
        }
        
        upload_result = await storacha_tool.execute(upload_params)
        assert upload_result["success"] is True
        
        # 3. Verify the upload
        list_params = {
            "action": "list",
            "space_did": "did:key:test_space_did"
        }
        
        list_result = await storacha_tool.execute(list_params)
        assert list_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_distributed_pinning_strategy(self, mock_embedding_service, temp_dir):
        """Test distributed pinning strategy across cluster."""
        cluster_tool = IPFSClusterManagementTool(mock_embedding_service)
        pinning_tool = IPFSPinningTool(mock_embedding_service)
        
        # Initialize cluster
        init_params = {
            "action": "initialize",
            "cluster_name": "distributed_pin_cluster",
            "cluster_config": {
                "replication_factor": 3
            },
            "data_dir": str(Path(temp_dir) / "distributed_cluster")
        }
        
        cluster_result = await cluster_tool.execute(init_params)
        assert cluster_result["success"] is True
        cluster_id = cluster_result["cluster_id"]
        
        # Pin with cluster replication
        test_file = Path(temp_dir) / "distributed_test.txt"
        test_file.write_text("Content for distributed pinning")
        
        pin_params = {
            "action": "pin_file",
            "file_path": str(test_file),
            "cluster_pinning": {
                "enabled": True,
                "replication_factor": 3,
                "cluster_name": "distributed_pin_cluster"
            }
        }
        
        pin_result = await pinning_tool.execute(pin_params)
        assert pin_result["success"] is True
        
        # Verify cluster status includes the pin
        status_params = {
            "action": "status",
            "cluster_id": cluster_id
        }
        
        status_result = await cluster_tool.execute(status_params)
        assert status_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_ipfs_error_handling_and_recovery(self, mock_embedding_service, temp_dir):
        """Test error handling and recovery in IPFS operations."""
        cluster_tool = IPFSClusterManagementTool(mock_embedding_service)
        
        # Try to start non-existent cluster
        start_params = {
            "action": "start",
            "cluster_id": "nonexistent_cluster"
        }
        
        start_result = await cluster_tool.execute(start_params)
        assert start_result["success"] is False
        assert "error" in start_result
        
        # Try to pin non-existent file
        pinning_tool = IPFSPinningTool(mock_embedding_service)
        
        pin_params = {
            "action": "pin_file",
            "file_path": "/nonexistent/file.txt"
        }
        
        pin_result = await pinning_tool.execute(pin_params)
        assert pin_result["success"] is False
        assert "error" in pin_result
