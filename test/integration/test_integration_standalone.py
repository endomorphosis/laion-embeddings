#!/usr/bin/env python3
"""
Standalone test runner for integration tests.

This script runs integration tests without pytest to avoid conftest import issues.
"""

import sys
import os
import asyncio
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock problematic imports early
from unittest.mock import Mock, patch, MagicMock
import sys

# Set up mocks before importing services
torchvision_mock = MagicMock()
torchvision_mock.ops = MagicMock()
torchvision_mock.ops.nms = MagicMock()

transformers_mock = MagicMock()
transformers_mock.WhisperProcessor = MagicMock()
transformers_mock.WhisperTokenizer = MagicMock()
transformers_mock.WhisperFeatureExtractor = MagicMock()

# Apply mocks
sys.modules['torchvision'] = torchvision_mock
sys.modules['torchvision.ops'] = torchvision_mock.ops

# Set environment for testing
os.environ['TESTING'] = 'true'
os.environ['DISABLE_TELEMETRY'] = 'true'

# Test configuration
SAMPLE_DIMENSION = 384
SAMPLE_VECTORS_COUNT = 100

async def test_vector_service_basic():
    """Test basic vector service functionality."""
    print("Testing basic vector service...")
    
    try:
        from services.vector_service import VectorService, VectorConfig
        
        # Generate sample data
        np.random.seed(42)
        sample_vectors = np.random.random((50, SAMPLE_DIMENSION)).astype(np.float32)
        sample_texts = [f"Sample text {i}" for i in range(50)]
        sample_metadata = [{'id': f'doc_{i}', 'source': 'test'} for i in range(50)]
        query_vector = np.random.random(SAMPLE_DIMENSION).astype(np.float32)
        
        # Initialize service
        config = VectorConfig(dimension=SAMPLE_DIMENSION, index_type="Flat")
        service = VectorService(config)
        
        # Test adding embeddings
        print("Adding embeddings...")
        result = await service.add_embeddings(
            embeddings=sample_vectors,
            texts=sample_texts,
            metadata=sample_metadata
        )
        
        print(f"Add result: {result}")
        assert result['status'] == 'success'
        assert result['added_count'] == 50
        
        # Test searching
        print("Testing search...")
        search_result = await service.search_similar(
            query_embedding=query_vector,
            k=5
        )
        
        print(f"Search result: {search_result}")
        assert search_result['status'] == 'success'
        assert len(search_result['results']) <= 5
        
        print("✓ Vector service test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Vector service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ipfs_vector_service_basic():
    """Test basic IPFS vector service functionality."""
    print("Testing IPFS vector service...")
    
    try:
        from services.ipfs_vector_service import IPFSVectorService, IPFSConfig
        from services.vector_service import VectorConfig
        
        # Generate sample data
        np.random.seed(42)
        sample_vectors = np.random.random((30, SAMPLE_DIMENSION)).astype(np.float32)
        sample_texts = [f"Sample text {i}" for i in range(30)]
        sample_metadata = [{'id': f'doc_{i}', 'source': 'test'} for i in range(30)]
        query_vector = np.random.random(SAMPLE_DIMENSION).astype(np.float32)
        
        # Initialize service
        vector_config = VectorConfig(dimension=SAMPLE_DIMENSION, index_type="Flat")
        ipfs_config = IPFSConfig(chunk_size=15)
        service = IPFSVectorService(vector_config, ipfs_config)
        
        # Test adding embeddings
        print("Adding embeddings to IPFS service...")
        result = await service.add_embeddings(
            embeddings=sample_vectors,
            texts=sample_texts,
            metadata=sample_metadata,
            store_in_ipfs=True
        )
        
        print(f"Add result: {result}")
        assert result['status'] == 'success'
        
        # Test searching
        print("Testing search...")
        search_result = await service.search_similar(
            query_embedding=query_vector,
            k=5,
            use_local=True,
            use_distributed=False
        )
        
        print(f"Search result: {search_result}")
        assert search_result['status'] == 'success'
        
        print("✓ IPFS vector service test passed!")
        return True
        
    except Exception as e:
        print(f"✗ IPFS vector service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_clustering_service_basic():
    """Test basic clustering service functionality."""
    print("Testing clustering service...")
    
    try:
        from services.clustering_service import SmartShardingService, ClusterConfig
        from services.vector_service import VectorConfig
        
        # Generate sample data
        np.random.seed(42)
        sample_vectors = np.random.random((30, SAMPLE_DIMENSION)).astype(np.float32)
        sample_texts = [f"Sample text {i}" for i in range(30)]
        sample_metadata = [{'id': f'doc_{i}', 'source': 'test'} for i in range(30)]
        query_vector = np.random.random(SAMPLE_DIMENSION).astype(np.float32)
        
        # Initialize service
        vector_config = VectorConfig(dimension=SAMPLE_DIMENSION, index_type="Flat")
        cluster_config = ClusterConfig(n_clusters=3, algorithm="kmeans")
        service = SmartShardingService(vector_config, cluster_config)
        
        # Test adding vectors with clustering
        print("Adding vectors with clustering...")
        result = await service.add_vectors_with_clustering(
            vectors=sample_vectors,
            texts=sample_texts,
            metadata=sample_metadata
        )
        
        print(f"Add result: {result}")
        assert result['status'] == 'success'
        assert result['total_added'] == 30
        
        # Test search with cluster routing
        print("Testing search with cluster routing...")
        search_result = await service.search_with_cluster_routing(
            query_vector=query_vector,
            k=5,
            search_strategy="adaptive"
        )
        
        print(f"Search result: {search_result}")
        assert search_result['status'] == 'success'
        
        print("✓ Clustering service test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Clustering service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests."""
    print("Starting integration tests...\n")
    
    tests = [
        test_vector_service_basic,
        test_ipfs_vector_service_basic,
        test_clustering_service_basic
    ]
    
    results = []
    for test in tests:
        print(f"\n{'='*50}")
        result = await test()
        results.append(result)
        print(f"{'='*50}")
    
    print(f"\n\nTest Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
