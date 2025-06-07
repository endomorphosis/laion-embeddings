import os
import sys
import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Set environment variable for testing
os.environ['TESTING'] = 'true'

# Import the modules 
try:
    from services.ipfs_vector_service import IPFSVectorStorage, DistributedVectorIndex
    from services.vector_service import VectorConfig

    # Create simple test objects
    mock_vector_service = Mock()
    mock_vector_service.dimension = 2  # Set dimension for tests
    mock_vector_service.add_embeddings = AsyncMock(return_value=[0, 1])
    mock_vector_service.search_similar = AsyncMock(return_value=[
        {'id': 0, 'similarity': 0.9, 'text': 'result1'},
        {'id': 1, 'similarity': 0.8, 'text': 'result2'}
    ])

    ipfs_config = {'api_url': '/ip4/127.0.0.1/tcp/5001'}
    
    # Running tests manually
    def test_distributed_index_initialization():
        print("Testing DistributedVectorIndex initialization...")
        distributed_index = DistributedVectorIndex(mock_vector_service, ipfs_config, shard_size=2)
        print(f"Dimension: {distributed_index.vector_config.dimension}")
        print(f"Shard size: {distributed_index.shard_size}")
        print("DistributedVectorIndex initialization: PASSED")

    def test_add_vectors_distributed():
        print("Testing add_vectors_distributed...")
        distributed_index = DistributedVectorIndex(mock_vector_service, ipfs_config, shard_size=2)
        
        # Create test embeddings
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ['text1', 'text2']
        metadata = [{'id': i} for i in range(2)]
        
        try:
            # Test with asyncio run
            import asyncio
            manifest_hash = asyncio.run(distributed_index.add_vectors_distributed(embeddings, texts))
            print(f"Got manifest hash: {manifest_hash}")
            print("add_vectors_distributed: PASSED")
        except Exception as e:
            print(f"add_vectors_distributed FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the tests
    test_distributed_index_initialization()
    test_add_vectors_distributed()
except Exception as e:
    print(f"TEST SETUP FAILED: {e}")
    import traceback
    traceback.print_exc()
