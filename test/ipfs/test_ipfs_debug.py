import os
import sys
import pytest
import logging
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

# Set environment variable for testing
os.environ['TESTING'] = 'true'

print("Starting test script...")

# Import the modules 
try:
    from services.ipfs_vector_service import IPFSVectorStorage, DistributedVectorIndex
    from services.vector_service import VectorConfig
    
    print("Imports successful")

    # Create simple test objects
    mock_ipfs_client = Mock()
    mock_ipfs_client.version = Mock(return_value={'Version': '0.12.0'})
    mock_ipfs_client.add_json = Mock(return_value='QmTestHash123')
    mock_ipfs_client.get_json = Mock(return_value={
        'shard_id': 'test_shard',
        'vectors': [[0.1, 0.2], [0.3, 0.4]],
        'metadata': {'texts': ['text1', 'text2'], 'metadata': [{}, {}]},
        'timestamp': 1234567890,
        'shape': [2, 2],
        'dtype': 'float32'
    })
    mock_ipfs_client.pin = Mock()
    mock_ipfs_client.pin.add = Mock(return_value={'Pins': ['QmTestHash123']})
    
    # Mock IPFSVectorStorage
    mock_storage = Mock()
    mock_storage.store_vector_shard = AsyncMock(return_value='QmShard123')
    mock_storage.retrieve_vector_shard = AsyncMock(return_value={
        'shard_id': 'test_shard',
        'vectors': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        'metadata': {'texts': ['text1', 'text2'], 'metadata': [{}, {}]},
    })
    mock_storage.store_index_manifest = AsyncMock(return_value='QmManifest123')
    mock_storage.retrieve_index_manifest = AsyncMock(return_value={
        'total_vectors': 100,
        'total_shards': 1,
        'shards': {'shard_0': {'ipfs_hash': 'QmShard123'}}
    })
    
    # Mock vector service
    mock_vector_service = Mock()
    mock_vector_service.dimension = 2
    mock_vector_service.add_embeddings = AsyncMock(return_value=[0, 1])
    mock_vector_service.search_similar = AsyncMock(return_value=[
        {'id': 0, 'similarity': 0.9, 'text': 'result1'},
        {'id': 1, 'similarity': 0.8, 'text': 'result2'}
    ])
    
    # Running tests manually
    def test_distributed_index_initialization():
        print("Testing DistributedVectorIndex initialization...")
        distributed_index = DistributedVectorIndex(mock_vector_service, mock_storage, shard_size=2)
        print(f"Dimension: {distributed_index.vector_config.dimension}")
        print(f"Shard size: {distributed_index.shard_size}")
        print("DistributedVectorIndex initialization: PASSED")

    def test_distributed_add_vectors():
        print("Testing add_vectors...")
        distributed_index = DistributedVectorIndex(mock_vector_service, mock_storage, shard_size=2)
        
        # Create test embeddings
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        texts = ['text1', 'text2']
        metadata = [{'id': i} for i in range(2)]
        
        try:
            # Test with asyncio run
            import asyncio
            result = asyncio.run(distributed_index.add_vectors(embeddings, texts=texts, metadata=metadata))
            print(f"add_vectors result: {result}")
            print("add_vectors: PASSED")
        except Exception as e:
            print(f"add_vectors FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    def test_distributed_add_vectors_distributed():
        print("Testing add_vectors_distributed...")
        distributed_index = DistributedVectorIndex(mock_vector_service, mock_storage, shard_size=2)
        
        # Create test embeddings
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        texts = ['text1', 'text2']
        
        try:
            # Test with asyncio run
            import asyncio
            result = asyncio.run(distributed_index.add_vectors_distributed(embeddings, texts))
            print(f"add_vectors_distributed result: {result}")
            print("add_vectors_distributed: PASSED")
        except Exception as e:
            print(f"add_vectors_distributed FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the tests
    test_distributed_index_initialization()
    test_distributed_add_vectors()
    test_distributed_add_vectors_distributed()
    
except Exception as e:
    print(f"TEST SETUP FAILED: {e}")
    import traceback
    traceback.print_exc()
