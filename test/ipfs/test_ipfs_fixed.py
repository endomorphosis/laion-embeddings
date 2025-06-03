"""
Tests for IPFS functionality with fixed implementations.

This module contains tests for IPFS storage and vector operations,
focusing on the fixed implementations that work reliably in the test environment.
"""

import re
import pytest
import time
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock

from services.ipfs_vector_service import IPFSVectorStorage, IPFSConfig
from services.vector_service import VectorService, VectorConfig
from services.distributed_vector_service import DistributedVectorIndex

@pytest.fixture
def mock_ipfs_client():
    """Mock IPFS HTTP client."""
    client = Mock()
    client.version = Mock(return_value={'Version': '0.12.0'})
    client.add_json = Mock(return_value='QmTestHash123')
    client.get_json = Mock(return_value={
        'shard_id': 'test_shard',
        'vectors': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        'metadata': {'texts': ['text1', 'text2']},
        'timestamp': '2023-01-01T00:00:00.000Z',
        'shape': [2, 3],
        'dtype': 'float32'
    })
    
    # Add pin attribute with add and rm methods
    client.pin = Mock()
    client.pin.add = Mock(return_value={'Pins': ['QmTestHash123']})
    client.pin.rm = Mock(return_value={'Pins': ['QmTestHash123']})
    
    return client

@pytest.fixture
def ipfs_config():
    """IPFS configuration for testing."""
    return IPFSConfig(api_url='/ip4/127.0.0.1/tcp/5001')

@pytest.fixture
def ipfs_storage(ipfs_config, mock_ipfs_client):
    """Create IPFS storage instance with mocked client."""
    with patch('services.ipfs_vector_service.ipfshttpclient.connect', return_value=mock_ipfs_client):
        return IPFSVectorStorage(ipfs_config)

@pytest.fixture
def distributed_vectors():
    """Create test vectors for distributed index testing."""
    return np.random.random((10, 128)).astype(np.float32)

@pytest.mark.asyncio
async def test_store_vector_shard(ipfs_storage, mock_ipfs_client):
    """Test storing vector shard in IPFS."""
    vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    metadata = {'texts': ['text1', 'text2']}
    
    ipfs_hash = await ipfs_storage.store_vector_shard(vectors, metadata)
    
    assert ipfs_hash == 'QmTestHash123'
    mock_ipfs_client.add_json.assert_called_once()
    
    # Check that pin was called if pin_content is True
    if ipfs_storage.config.pin_content:
        mock_ipfs_client.pin.add.assert_called_once()

@pytest.mark.asyncio
async def test_retrieve_vector_shard(ipfs_storage, mock_ipfs_client):
    """Test retrieving vector shard from IPFS."""
    ipfs_hash = 'QmTestHash123'
    
    shard_data = await ipfs_storage.retrieve_vector_shard(ipfs_hash)
    
    assert shard_data['shard_id'] == 'test_shard'
    assert isinstance(shard_data['vectors'], np.ndarray)
    assert shard_data['vectors'].shape == (2, 3)
    assert shard_data['vectors'].dtype == np.float32
    mock_ipfs_client.get_json.assert_called_once_with(ipfs_hash)

@pytest.mark.asyncio
async def test_store_index_manifest(ipfs_storage, mock_ipfs_client):
    """Test storing index manifest in IPFS."""
    manifest = {
        'total_vectors': 100,
        'total_shards': 2,
        'shard_size': 50,
        'shards': {
            'shard_1': {'ipfs_hash': 'QmShard1', 'vector_count': 50},
            'shard_2': {'ipfs_hash': 'QmShard2', 'vector_count': 50}
        }
    }
    
    ipfs_hash = await ipfs_storage.store_index_manifest(manifest)
    
    assert ipfs_hash == 'QmTestHash123'
    mock_ipfs_client.add_json.assert_called_once()

@pytest.mark.asyncio
async def test_retrieve_index_manifest(ipfs_storage, mock_ipfs_client):
    """Test retrieving index manifest from IPFS."""
    mock_ipfs_client.get_json.return_value = {
        'total_vectors': 100,
        'total_shards': 2,
        'shard_size': 50,
        'shards': {
            'shard_1': {'ipfs_hash': 'QmShard1', 'vector_count': 50},
            'shard_2': {'ipfs_hash': 'QmShard2', 'vector_count': 50}
        }
    }
    
    ipfs_hash = 'QmManifest123'
    manifest = await ipfs_storage.retrieve_index_manifest(ipfs_hash)
    
    assert manifest['total_vectors'] == 100
    assert manifest['total_shards'] == 2
    assert len(manifest['shards']) == 2
    mock_ipfs_client.get_json.assert_called_once_with(ipfs_hash)

@pytest.mark.asyncio
async def test_distributed_vector_index(ipfs_storage, distributed_vectors):
    """Test distributed vector index with IPFS storage."""
    # Create distributed index with mocked IPFS storage
    distributed_index = DistributedVectorIndex(
        storage=ipfs_storage,
        vector_config=VectorConfig(dimension=128),
        shard_size=5
    )
    
    # Mock the method on the specific instance
    with patch.object(distributed_index.vector_service, 'add_vectors', return_value={'index_id': 'test_index'}):
        # Add vectors to distributed index
        manifest_cid = await distributed_index.add_vectors_distributed(distributed_vectors)
        
        # Verify manifest was created and stored
        assert manifest_cid == 'QmTestHash123'
        
        # Verify vectors were split into shards (10 vectors with shard_size=5 should be 2 shards)
        assert distributed_index._calculate_shard_count(distributed_vectors) == 2

@pytest.mark.asyncio
async def test_search_distributed(ipfs_storage, mock_ipfs_client):
    """Test searching in distributed index."""
    # Setup mock responses for manifest and shards
    mock_ipfs_client.get_json.side_effect = [
        # First call: Get manifest
        {
            'total_vectors': 10,
            'total_shards': 2,
            'shard_size': 5,
            'dimension': 128,
            'shards': {
                'shard_1': {'ipfs_hash': 'QmShard1', 'vector_count': 5},
                'shard_2': {'ipfs_hash': 'QmShard2', 'vector_count': 5}
            }
        },
        # Second call: Get shard 1
        {
            'shard_id': 'shard_1',
            'vectors': [[0.1] * 128, [0.2] * 128],
            'metadata': {'ids': [1, 2]},
            'shape': [2, 128],
            'dtype': 'float32'
        },
        # Third call: Get shard 2
        {
            'shard_id': 'shard_2',
            'vectors': [[0.3] * 128, [0.4] * 128],
            'metadata': {'ids': [3, 4]},
            'shape': [2, 128],
            'dtype': 'float32'
        }
    ]
    
    distributed_index = DistributedVectorIndex(
        storage=ipfs_storage,
        vector_config=VectorConfig(dimension=128),
        shard_size=5
    )
    
    # Mock VectorService.search method to return appropriate format
    with patch.object(distributed_index.vector_service, 'search', side_effect=[
        (np.array([[0.1, 0.2]]), np.array([[0, 1]])),
        (np.array([[0.3, 0.4]]), np.array([[0, 1]]))
    ]):
        results = await distributed_index.search_distributed(
            'QmManifest123',
            np.array([0.5] * 128, dtype=np.float32),
            k=2
        )
        
        assert 'combined_results' in results
        assert len(results['combined_results']) <= 2  # We asked for k=2

@pytest.mark.asyncio
async def test_ipfs_connection_failure():
    """Test handling of IPFS connection failures."""
    config = IPFSConfig(api_url='/invalid/url')
    
    # Mock IPFS connection to raise an exception
    with patch('services.ipfs_vector_service.ipfshttpclient.connect', 
               side_effect=Exception("Connection failed")), \
         pytest.raises(Exception, match="Failed to connect"):
        
        # This should raise the exception
        IPFSVectorStorage(config)

@pytest.mark.asyncio
async def test_missing_ipfs_client():
    """Test handling of missing IPFS client."""
    # Set environment variable to simulate missing IPFS client
    import os
    os.environ['TEST_MISSING_IPFS'] = 'true'
    
    config = IPFSConfig(api_url='/ip4/127.0.0.1/tcp/5001')
    
    # This should raise ImportError
    with pytest.raises(ImportError, match="ipfshttpclient not available"):
        IPFSVectorStorage(config)
    
    # Clean up
    os.environ.pop('TEST_MISSING_IPFS')

@pytest.mark.asyncio
async def test_manifest_consistency(ipfs_storage):
    """Test consistency between stored and retrieved manifests."""
    manifest = {
        'total_vectors': 100,
        'total_shards': 2,
        'shard_size': 50,
        'shards': {
            'shard_1': {'ipfs_hash': 'QmShard1', 'vector_count': 50},
            'shard_2': {'ipfs_hash': 'QmShard2', 'vector_count': 50}
        }
    }
    
    # Store manifest
    cid = await ipfs_storage.store_index_manifest(manifest)
    
    # Retrieve manifest
    retrieved = await ipfs_storage.retrieve_index_manifest(cid)
    
    # Check consistency
    assert retrieved['total_vectors'] == manifest['total_vectors']
    assert retrieved['total_shards'] == manifest['total_shards']
    assert retrieved['shard_size'] == manifest['shard_size']
    assert len(retrieved['shards']) == len(manifest['shards'])
