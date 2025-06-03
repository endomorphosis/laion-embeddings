"""
Comprehensive integration tests for the complete vector quantization, clustering, and sharding implementation.

This test suite validates the end-to-end functionality of:
1. Vector service with FAISS integration
2. IPFS distributed vector storage  
3. Clustering-based smart sharding
4. Complete RAG workflow with IPFS
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any
import tempfile
import shutil
from pathlib import Path

# Test configuration
SAMPLE_DIMENSION = 384
SAMPLE_VECTORS_COUNT = 100
TEST_QUERY_VECTOR = np.random.random(SAMPLE_DIMENSION).astype(np.float32)


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.random((SAMPLE_VECTORS_COUNT, SAMPLE_DIMENSION)).astype(np.float32)


@pytest.fixture
def sample_texts():
    """Generate sample texts corresponding to vectors."""
    return [f"Sample text {i} for vector embeddings" for i in range(SAMPLE_VECTORS_COUNT)]


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for vectors."""
    return [{'id': f'doc_{i}', 'source': 'test', 'category': f'cat_{i % 5}'} for i in range(SAMPLE_VECTORS_COUNT)]


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock all external dependencies for isolated testing."""
    mocks = {}
    
    # Mock FAISS
    with patch.dict('sys.modules', {'faiss': Mock()}):
        faiss_mock = Mock()
        
        # Mock index types
        index_mock = Mock()
        index_mock.add = Mock()
        index_mock.add_with_ids = Mock()
        index_mock.search = Mock(return_value=(
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),  # distances
            np.array([[0, 1, 2, 3, 4]])  # indices
        ))
        index_mock.train = Mock()
        index_mock.ntotal = SAMPLE_VECTORS_COUNT
        index_mock.is_trained = True
        
        faiss_mock.IndexFlatL2 = Mock(return_value=index_mock)
        faiss_mock.IndexFlatIP = Mock(return_value=index_mock)
        faiss_mock.IndexIVFFlat = Mock(return_value=index_mock)
        faiss_mock.IndexIVFPQ = Mock(return_value=index_mock)
        faiss_mock.IndexHNSWFlat = Mock(return_value=index_mock)
        faiss_mock.IndexPQ = Mock(return_value=index_mock)
        
        faiss_mock.get_num_gpus = Mock(return_value=0)
        faiss_mock.StandardGpuResources = Mock()
        faiss_mock.index_cpu_to_gpu = Mock(return_value=index_mock)
        faiss_mock.index_gpu_to_cpu = Mock(return_value=index_mock)
        faiss_mock.write_index = Mock()
        faiss_mock.read_index = Mock(return_value=index_mock)
        faiss_mock.normalize_L2 = Mock()
        
        mocks['faiss'] = faiss_mock
    
    # Mock scikit-learn
    with patch.dict('sys.modules', {
        'sklearn': Mock(),
        'sklearn.cluster': Mock(),
        'sklearn.metrics': Mock(),
        'sklearn.preprocessing': Mock()
    }):
        # Mock clustering algorithms
        kmeans_mock = Mock()
        kmeans_mock.fit = Mock()
        kmeans_mock.predict = Mock(return_value=np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0]))
        kmeans_mock.fit_predict = Mock(return_value=np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0] * (SAMPLE_VECTORS_COUNT // 10 + 1))[:SAMPLE_VECTORS_COUNT])
        kmeans_mock.cluster_centers_ = np.random.random((3, SAMPLE_DIMENSION))
        
        sklearn_cluster = Mock()
        sklearn_cluster.KMeans = Mock(return_value=kmeans_mock)
        sklearn_cluster.AgglomerativeClustering = Mock(return_value=kmeans_mock)
        sklearn_cluster.DBSCAN = Mock(return_value=kmeans_mock)
        
        # Mock metrics
        sklearn_metrics = Mock()
        sklearn_metrics.silhouette_score = Mock(return_value=0.75)
        sklearn_metrics.calinski_harabasz_score = Mock(return_value=100.0)
        
        # Mock preprocessing
        scaler_mock = Mock()
        scaler_mock.fit_transform = Mock(side_effect=lambda x: x)
        scaler_mock.transform = Mock(side_effect=lambda x: x)
        sklearn_preprocessing = Mock()
        sklearn_preprocessing.StandardScaler = Mock(return_value=scaler_mock)
        
        mocks['sklearn'] = {
            'cluster': sklearn_cluster,
            'metrics': sklearn_metrics,
            'preprocessing': sklearn_preprocessing
        }
    
    # Mock IPFS client
    ipfs_client_mock = Mock()
    ipfs_client_mock.version = Mock(return_value={'Version': '0.12.0'})
    ipfs_client_mock.add_json = Mock(return_value='QmTestHash123')
    ipfs_client_mock.get_json = Mock(return_value={
        'shard_id': 'test_shard',
        'vectors': np.random.random((10, SAMPLE_DIMENSION)).tolist(),
        'metadata': {'texts': ['text1', 'text2'], 'metadata': [{}, {}]},
        'timestamp': '2025-01-01T00:00:00',
        'shape': [10, SAMPLE_DIMENSION],
        'dtype': 'float32'
    })
    ipfs_client_mock.pin = Mock()
    ipfs_client_mock.pin.add = Mock()
    
    with patch('services.ipfs_vector_service.ipfshttpclient') as ipfs_mock:
        ipfs_mock.connect = Mock(return_value=ipfs_client_mock)
        mocks['ipfs'] = ipfs_client_mock
    
    yield mocks


class TestVectorServiceIntegration:
    """Integration tests for vector service functionality."""
    
    @pytest.mark.asyncio
    async def test_vector_service_complete_workflow(self, sample_vectors, sample_texts, sample_metadata, temp_directory):
        """Test complete vector service workflow: add, search, save, load."""
        from services.vector_service import VectorService, VectorConfig
        
        # Initialize service
        config = VectorConfig(dimension=SAMPLE_DIMENSION, index_type="IVF")
        service = VectorService(config)
        
        # Test adding embeddings
        result = await service.add_embeddings(
            embeddings=sample_vectors[:50],
            texts=sample_texts[:50],
            metadata=sample_metadata[:50]
        )
        
        assert result['status'] == 'success'
        assert result['added_count'] == 50
        assert result['total_vectors'] == 50
        
        # Test searching
        search_result = await service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=5
        )
        
        assert search_result['status'] == 'success'
        assert len(search_result['results']) == 5
        assert all('similarity_score' in result for result in search_result['results'])
        
        # Test saving index
        index_path = Path(temp_directory) / "test_index.faiss"
        save_result = await service.save_index(str(index_path))
        
        assert save_result['status'] == 'success'
        assert save_result['vectors_count'] == 50
        
        # Test loading index
        new_service = VectorService(config)
        load_result = await new_service.load_index(str(index_path))
        
        assert load_result['status'] == 'success'
        assert load_result['vectors_count'] == 50
        
        # Test search on loaded index
        search_result_2 = await new_service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=3
        )
        
        assert search_result_2['status'] == 'success'
        assert len(search_result_2['results']) == 3
    
    @pytest.mark.asyncio
    async def test_different_index_types(self, sample_vectors):
        """Test different FAISS index types."""
        from services.vector_service import VectorService, VectorConfig
        
        index_types = ["Flat", "IVF", "HNSW", "PQ", "IVF_PQ"]
        
        for index_type in index_types:
            config = VectorConfig(dimension=SAMPLE_DIMENSION, index_type=index_type)
            service = VectorService(config)
            
            # Test adding vectors
            result = await service.add_embeddings(embeddings=sample_vectors[:20])
            assert result['status'] == 'success'
            
            # Test searching
            search_result = await service.search_similar(
                query_embedding=TEST_QUERY_VECTOR,
                k=5
            )
            assert search_result['status'] == 'success'
            assert len(search_result['results']) <= 5


class TestIPFSVectorServiceIntegration:
    """Integration tests for IPFS vector service."""
    
    @pytest.mark.asyncio
    async def test_ipfs_vector_service_workflow(self, sample_vectors, sample_texts, sample_metadata):
        """Test complete IPFS vector service workflow."""
        from services.ipfs_vector_service import IPFSVectorService, IPFSConfig
        from services.vector_service import VectorConfig
        
        # Initialize service
        vector_config = VectorConfig(dimension=SAMPLE_DIMENSION)
        ipfs_config = IPFSConfig(chunk_size=20)
        service = IPFSVectorService(vector_config, ipfs_config)
        
        # Test adding embeddings to both local and IPFS
        result = await service.add_embeddings(
            embeddings=sample_vectors[:40],
            texts=sample_texts[:40],
            metadata=sample_metadata[:40],
            store_in_ipfs=True
        )
        
        assert result['status'] == 'success'
        assert 'local' in result
        assert 'distributed' in result
        assert result['local']['added_count'] == 40
        assert result['distributed']['added_count'] == 40
        
        # Test searching with local index
        search_result = await service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=5,
            use_local=True,
            use_distributed=False
        )
        
        assert search_result['status'] == 'success'
        assert 'local' in search_result['results']
        assert len(search_result['results']['local']['results']) == 5
        
        # Test searching with distributed index
        search_result = await service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=5,
            use_local=False,
            use_distributed=True
        )
        
        assert search_result['status'] == 'success'
        assert 'distributed' in search_result['results']
        
        # Test saving to IPFS
        save_cid = await service.save_to_ipfs("test_index")
        assert save_cid.startswith('Qm')  # IPFS CID format
        
        # Test loading from IPFS
        await service.load_from_ipfs(save_cid)
        
        # Test service statistics
        stats = service.get_service_stats()
        assert 'local_index' in stats
        assert 'distributed_index' in stats
        assert 'ipfs_config' in stats
    
    @pytest.mark.asyncio
    async def test_distributed_vector_index_sharding(self, sample_vectors):
        """Test vector sharding in distributed index."""
        from services.ipfs_vector_service import DistributedVectorIndex, IPFSConfig
        from services.vector_service import VectorConfig
        
        # Small chunk size to force multiple shards
        vector_config = VectorConfig(dimension=SAMPLE_DIMENSION)
        ipfs_config = IPFSConfig(chunk_size=15)
        
        index = DistributedVectorIndex(vector_config, ipfs_config)
        
        # Add vectors that will create multiple shards
        result = await index.add_vectors(sample_vectors[:50])
        
        assert result['status'] == 'success'
        assert result['added_count'] == 50
        assert len(result['shard_cids']) >= 3  # Should create at least 3 shards
        
        # Test searching across shards
        search_result = await index.search_vectors(TEST_QUERY_VECTOR, k=10)
        
        assert search_result['status'] == 'success'
        assert len(search_result['results']) <= 10
        assert search_result['searched_shards'] >= 3


class TestClusteringServiceIntegration:
    """Integration tests for clustering and smart sharding."""
    
    @pytest.mark.asyncio
    async def test_smart_sharding_complete_workflow(self, sample_vectors, sample_texts, sample_metadata):
        """Test complete smart sharding workflow with clustering."""
        from services.clustering_service import SmartShardingService, ClusterConfig
        from services.vector_service import VectorConfig
        
        # Initialize service
        vector_config = VectorConfig(dimension=SAMPLE_DIMENSION)
        cluster_config = ClusterConfig(n_clusters=3, algorithm="kmeans")
        service = SmartShardingService(vector_config, cluster_config)
        
        # Test adding vectors with clustering
        result = await service.add_vectors_with_clustering(
            vectors=sample_vectors[:30],
            texts=sample_texts[:30],
            metadata=sample_metadata[:30]
        )
        
        assert result['status'] == 'success'
        assert result['total_added'] == 30
        assert len(result['shards_used']) >= 1
        assert 'clustering_quality' in result
        assert result['clustering_quality']['silhouette_score'] >= 0.0
        
        # Test search with cluster routing
        search_result = await service.search_with_cluster_routing(
            query_vector=TEST_QUERY_VECTOR,
            k=5,
            search_strategy="adaptive"
        )
        
        assert search_result['status'] == 'success'
        assert len(search_result['results']) <= 5
        assert 'shards_searched' in search_result
        
        # Test different search strategies
        strategies = ["all", "nearest_clusters", "adaptive"]
        for strategy in strategies:
            search_result = await service.search_with_cluster_routing(
                query_vector=TEST_QUERY_VECTOR,
                k=3,
                search_strategy=strategy
            )
            assert search_result['status'] == 'success'
        
        # Test cluster optimization
        optimization_result = await service.optimize_clusters(sample_vectors[:30])
        assert optimization_result['status'] == 'success'
        
        # Test statistics
        stats = service.get_sharding_stats()
        assert stats['total_vectors'] == 30
        assert stats['total_shards'] >= 1
        assert 'clusters_info' in stats
    
    @pytest.mark.asyncio
    async def test_different_clustering_algorithms(self, sample_vectors):
        """Test different clustering algorithms."""
        from services.clustering_service import SmartShardingService, ClusterConfig
        from services.vector_service import VectorConfig
        
        algorithms = ["kmeans", "hierarchical", "dbscan"]
        vector_config = VectorConfig(dimension=SAMPLE_DIMENSION)
        
        for algorithm in algorithms:
            cluster_config = ClusterConfig(n_clusters=3, algorithm=algorithm)
            service = SmartShardingService(vector_config, cluster_config)
            
            # Test clustering
            result = await service.add_vectors_with_clustering(vectors=sample_vectors[:20])
            assert result['status'] == 'success'
            assert result['total_added'] == 20


class TestCompleteRAGWorkflow:
    """Integration tests for complete RAG workflow with IPFS."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_rag_workflow(self, sample_vectors, sample_texts, sample_metadata):
        """Test complete end-to-end RAG workflow."""
        from services.vector_service import VectorService, VectorConfig
        from services.ipfs_vector_service import IPFSVectorService, IPFSConfig
        from services.clustering_service import SmartShardingService, ClusterConfig
        
        # 1. Initialize all services
        vector_config = VectorConfig(dimension=SAMPLE_DIMENSION, index_type="IVF")
        ipfs_config = IPFSConfig(chunk_size=20)
        cluster_config = ClusterConfig(n_clusters=3)
        
        # Basic vector service
        basic_service = VectorService(vector_config)
        
        # IPFS-enhanced service
        ipfs_service = IPFSVectorService(vector_config, ipfs_config)
        
        # Smart sharding service
        sharding_service = SmartShardingService(vector_config, cluster_config)
        
        # 2. Add data to all services
        data_slice = slice(0, 40)
        
        # Add to basic service
        basic_result = await basic_service.add_embeddings(
            embeddings=sample_vectors[data_slice],
            texts=sample_texts[data_slice],
            metadata=sample_metadata[data_slice]
        )
        assert basic_result['status'] == 'success'
        
        # Add to IPFS service
        ipfs_result = await ipfs_service.add_embeddings(
            embeddings=sample_vectors[data_slice],
            texts=sample_texts[data_slice],
            metadata=sample_metadata[data_slice],
            store_in_ipfs=True
        )
        assert ipfs_result['status'] == 'success'
        
        # Add to sharding service
        sharding_result = await sharding_service.add_vectors_with_clustering(
            vectors=sample_vectors[data_slice],
            texts=sample_texts[data_slice],
            metadata=sample_metadata[data_slice]
        )
        assert sharding_result['status'] == 'success'
        
        # 3. Test search performance across all services
        query_results = {}
        
        # Search basic service
        query_results['basic'] = await basic_service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=5
        )
        
        # Search IPFS service (local)
        query_results['ipfs_local'] = await ipfs_service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=5,
            use_local=True,
            use_distributed=False
        )
        
        # Search IPFS service (distributed)
        query_results['ipfs_distributed'] = await ipfs_service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=5,
            use_local=False,
            use_distributed=True
        )
        
        # Search sharding service
        query_results['sharding'] = await sharding_service.search_with_cluster_routing(
            query_vector=TEST_QUERY_VECTOR,
            k=5,
            search_strategy="adaptive"
        )
        
        # 4. Validate all searches succeeded
        for service_name, result in query_results.items():
            assert result['status'] == 'success'
            if service_name == 'ipfs_local':
                assert len(result['results']['local']['results']) <= 5
            elif service_name == 'ipfs_distributed':
                assert len(result['results']['distributed']['results']) <= 5
            else:
                assert len(result['results']) <= 5
        
        # 5. Test persistence and recovery
        # Save IPFS index
        ipfs_cid = await ipfs_service.save_to_ipfs("integration_test_index")
        assert ipfs_cid.startswith('Qm')
        
        # Create new service and load from IPFS
        new_ipfs_service = IPFSVectorService(vector_config, ipfs_config)
        await new_ipfs_service.load_from_ipfs(ipfs_cid)
        
        # Test search on recovered service
        recovered_result = await new_ipfs_service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=3,
            use_distributed=True
        )
        assert recovered_result['status'] == 'success'


class TestPerformanceAndScalability:
    """Performance and scalability tests."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        from services.vector_service import VectorService, VectorConfig
        
        # Generate larger dataset
        large_vectors = np.random.random((500, SAMPLE_DIMENSION)).astype(np.float32)
        
        config = VectorConfig(dimension=SAMPLE_DIMENSION, batch_size=100)
        service = VectorService(config)
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(large_vectors), batch_size):
            batch = large_vectors[i:i + batch_size]
            result = await service.add_embeddings(embeddings=batch)
            assert result['status'] == 'success'
        
        # Test search performance
        search_result = await service.search_similar(
            query_embedding=TEST_QUERY_VECTOR,
            k=10
        )
        assert search_result['status'] == 'success'
        assert len(search_result['results']) <= 10
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, sample_vectors):
        """Test concurrent vector operations."""
        from services.vector_service import VectorService, VectorConfig
        
        config = VectorConfig(dimension=SAMPLE_DIMENSION)
        service = VectorService(config)
        
        # Add initial data
        await service.add_embeddings(embeddings=sample_vectors[:50])
        
        # Test concurrent searches
        search_tasks = []
        for _ in range(5):
            task = service.search_similar(
                query_embedding=np.random.random(SAMPLE_DIMENSION).astype(np.float32),
                k=5
            )
            search_tasks.append(task)
        
        # Execute concurrent searches
        results = await asyncio.gather(*search_tasks)
        
        # Validate all searches succeeded
        for result in results:
            assert result['status'] == 'success'
            assert len(result['results']) <= 5


# Test runner for comprehensive validation
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])
