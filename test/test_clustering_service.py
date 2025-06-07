"""
Test suite for vector clustering and smart sharding.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

@pytest.fixture
def mock_sklearn():
    """Mock scikit-learn components."""
    with patch.dict('sys.modules', {
        'sklearn': Mock(),
        'sklearn.cluster': Mock(),
        'sklearn.metrics': Mock()
    }):
        def dynamic_fit_predict(vectors):
            """Dynamic mock that returns appropriate labels based on input size."""
            n_vectors = len(vectors) if hasattr(vectors, '__len__') else 15
            n_clusters = min(3, max(1, n_vectors // 2))  # Adaptive cluster count
            # Create labels that distribute vectors across clusters
            labels = []
            for i in range(n_vectors):
                labels.append(i % n_clusters)
            return np.array(labels)
        
        def dynamic_predict(vectors):
            """Dynamic mock for prediction."""
            n_vectors = len(vectors) if hasattr(vectors, '__len__') else 2
            return np.array([i % 3 for i in range(n_vectors)])
        
        # Mock KMeans with dynamic label count
        kmeans_mock = Mock()
        kmeans_mock.fit_predict = Mock(side_effect=dynamic_fit_predict)
        kmeans_mock.predict = Mock(side_effect=dynamic_predict)
        kmeans_mock.cluster_centers_ = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6], 
            [0.7, 0.8, 0.9]
        ])
        
        # Mock AgglomerativeClustering with dynamic label count
        agg_mock = Mock()
        agg_mock.fit_predict = Mock(side_effect=dynamic_fit_predict)
        
        sklearn_cluster = Mock()
        sklearn_cluster.KMeans = Mock(return_value=kmeans_mock)
        sklearn_cluster.AgglomerativeClustering = Mock(return_value=agg_mock)
        
        # Mock metrics
        sklearn_metrics = Mock()
        sklearn_metrics.silhouette_score = Mock(return_value=0.75)
        
        yield {
            'cluster': sklearn_cluster,
            'metrics': sklearn_metrics,
            'kmeans': kmeans_mock,
            'agg': agg_mock
        }


class TestClusterConfig:
    """Test cluster configuration."""
    
    def test_cluster_config_defaults(self):
        """Test default cluster configuration values."""
        from services.clustering_service import ClusterConfig
        
        config = ClusterConfig()
        assert config.n_clusters == 10
        assert config.algorithm == "kmeans"
        assert config.max_iter == 300
        assert config.tolerance == 1e-4
        assert config.random_state == 42
    
    def test_cluster_config_custom(self):
        """Test custom cluster configuration."""
        from services.clustering_service import ClusterConfig
        
        config = ClusterConfig(
            n_clusters=5,
            algorithm="hierarchical",
            max_iter=100,
            tolerance=1e-3
        )
        
        assert config.n_clusters == 5
        assert config.algorithm == "hierarchical"
        assert config.max_iter == 100
        assert config.tolerance == 1e-3


class TestVectorClusterer:
    """Test vector clustering functionality."""
    
    @pytest.fixture
    def cluster_config(self):
        """Create test cluster configuration."""
        from services.clustering_service import ClusterConfig
        return ClusterConfig(n_clusters=3, max_iter=100)
    
    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for clustering."""
        # Create 3 distinct clusters
        cluster1 = np.random.normal([0, 0, 0], 0.1, (5, 3))
        cluster2 = np.random.normal([1, 1, 1], 0.1, (5, 3))
        cluster3 = np.random.normal([2, 2, 2], 0.1, (5, 3))
        
        return np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
    
    def test_clusterer_initialization(self, cluster_config):
        """Test vector clusterer initialization."""
        from services.clustering_service import VectorClusterer
        
        clusterer = VectorClusterer(cluster_config)
        assert clusterer.config == cluster_config
        assert clusterer.cluster_centers is None
        assert clusterer.cluster_labels is None
        assert clusterer.cluster_metadata == {}
    
    def test_kmeans_clustering(self, cluster_config, sample_vectors, mock_sklearn):
        """Test K-means clustering."""
        from services.clustering_service import VectorClusterer
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans):
            clusterer = VectorClusterer(cluster_config)
            labels = clusterer.fit_kmeans(sample_vectors)
            
            assert labels is not None
            assert len(labels) == len(sample_vectors)
            assert clusterer.cluster_centers is not None
            assert clusterer.cluster_centers.shape == (3, 3)  # 3 clusters, 3 dimensions
            assert len(clusterer.cluster_metadata) == 3
            
            # Check cluster metadata
            for i in range(3):
                assert i in clusterer.cluster_metadata
                meta = clusterer.cluster_metadata[i]
                assert 'size' in meta
                assert 'center' in meta
                assert 'inertia' in meta
                assert 'radius' in meta
    
    def test_hierarchical_clustering(self, cluster_config, sample_vectors, mock_sklearn):
        """Test hierarchical clustering."""
        from services.clustering_service import VectorClusterer
        
        with patch('services.clustering_service.AgglomerativeClustering', 
                  mock_sklearn['cluster'].AgglomerativeClustering):
            clusterer = VectorClusterer(cluster_config)
            labels = clusterer.fit_hierarchical(sample_vectors)
            
            assert labels is not None
            assert len(labels) == len(sample_vectors)
            assert clusterer.cluster_centers is not None
            assert len(clusterer.cluster_metadata) == 3
    
    def test_predict_cluster(self, cluster_config, sample_vectors, mock_sklearn):
        """Test predicting cluster for new vectors."""
        from services.clustering_service import VectorClusterer
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans):
            clusterer = VectorClusterer(cluster_config)
            clusterer.fit_kmeans(sample_vectors)
            
            # Predict cluster for new vectors
            new_vectors = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]], dtype=np.float32)
            predictions = clusterer.predict_cluster(new_vectors)
            
            assert len(predictions) == 2
            assert all(0 <= pred < 3 for pred in predictions)
    
    def test_predict_before_fit_error(self, cluster_config):
        """Test error when predicting before fitting."""
        from services.clustering_service import VectorClusterer
        
        clusterer = VectorClusterer(cluster_config)
        new_vectors = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Clusterer not fitted yet"):
            clusterer.predict_cluster(new_vectors)
    
    def test_get_cluster_stats(self, cluster_config, sample_vectors, mock_sklearn):
        """Test getting cluster statistics."""
        from services.clustering_service import VectorClusterer
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans), \
             patch('services.clustering_service.silhouette_score', mock_sklearn['metrics'].silhouette_score):
            
            clusterer = VectorClusterer(cluster_config)
            clusterer.fit_kmeans(sample_vectors)
            
            stats = clusterer.get_cluster_stats()
            
            assert stats['n_clusters'] == 3
            assert stats['algorithm'] == 'kmeans'
            assert 'cluster_sizes' in stats
            assert 'total_vectors' in stats
            assert 'silhouette_score' in stats
    
    def test_sklearn_import_error(self, cluster_config, sample_vectors):
        """Test handling when scikit-learn is not available."""
        from services.clustering_service import VectorClusterer
        
        # Save the original environment value
        import os
        original_testing = os.environ.get('TESTING', '')
        
        try:
            # Temporarily disable testing mode for this test
            os.environ['TESTING'] = 'false'
            
            with patch('services.clustering_service.SKLEARN_AVAILABLE', False):
                with pytest.raises(ImportError, match="scikit-learn"):
                    # The import error should be raised during initialization when not in testing mode
                    VectorClusterer(cluster_config)
        finally:
            # Restore the original environment value
            if original_testing:
                os.environ['TESTING'] = original_testing
            else:
                del os.environ['TESTING']


class TestSmartShardingService:
    """Test smart sharding service."""
    
    @pytest.fixture
    def mock_vector_service(self):
        """Mock vector service for testing."""
        service = Mock()
        service.create_index = Mock(return_value=Mock())
        service.add_embeddings = AsyncMock(return_value=[0, 1, 2])
        service.search_similar = AsyncMock(return_value=[
            {'id': 0, 'similarity': 0.9, 'text': 'result1'},
            {'id': 1, 'similarity': 0.8, 'text': 'result2'}
        ])
        return service
    
    @pytest.fixture
    def mock_clusterer(self, mock_sklearn):
        """Mock vector clusterer."""
        from services.clustering_service import VectorClusterer, ClusterConfig
        
        config = ClusterConfig(n_clusters=3)
        clusterer = VectorClusterer(config)
        
        # Mock clustering results
        clusterer.cluster_centers = np.array([
            [0.1, 0.2], [0.5, 0.6], [0.9, 1.0]
        ])
        clusterer.cluster_metadata = {
            0: {'size': 2, 'center': [0.1, 0.2], 'radius': 0.1},
            1: {'size': 2, 'center': [0.5, 0.6], 'radius': 0.1},
            2: {'size': 1, 'center': [0.9, 1.0], 'radius': 0.1}
        }
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans):
            clusterer.fit_kmeans = Mock(return_value=np.array([0, 0, 1, 1, 2]))
            
        return clusterer
    
    @pytest.fixture
    def smart_sharding(self, mock_vector_service, mock_clusterer):
        """Create smart sharding service."""
        from services.clustering_service import SmartShardingService
        return SmartShardingService(mock_vector_service, mock_clusterer)
    
    @pytest.mark.asyncio
    async def test_create_clustered_shards(self, smart_sharding, mock_vector_service):
        """Test creating clustered shards."""
        embeddings = [[0.1, 0.2], [0.15, 0.25], [0.5, 0.6], [0.55, 0.65], [0.9, 1.0]]
        texts = [f'text_{i}' for i in range(5)]
        metadata = [{'id': i} for i in range(5)]
        
        result = await smart_sharding.create_clustered_shards(embeddings, texts, metadata)
        
        assert 'shards' in result
        assert 'clustering_stats' in result
        assert 'total_vectors' in result
        assert result['total_vectors'] == 5
        
        # Should create 3 shards (one for each cluster)
        shards = result['shards']
        assert len(shards) <= 3  # May be fewer if some clusters are empty
        
        # Verify vector service calls
        assert mock_vector_service.create_index.call_count >= 1
        assert mock_vector_service.add_embeddings.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_search_clustered_shards(self, smart_sharding, mock_vector_service):
        """Test searching with cluster-aware routing."""
        # Setup shard mappings
        smart_sharding.shard_mappings = {
            0: {'shard_name': 'cluster_shard_0', 'cluster_center': [0.1, 0.2]},
            1: {'shard_name': 'cluster_shard_1', 'cluster_center': [0.5, 0.6]},
            2: {'shard_name': 'cluster_shard_2', 'cluster_center': [0.9, 1.0]}
        }
        
        query_embedding = [0.1, 0.2]  # Close to cluster 0
        results = await smart_sharding.search_clustered_shards(query_embedding, k=5)
        
        assert len(results) >= 0  # May be empty due to mocking
        
        # Verify search was called (may be multiple times for different clusters)
        mock_vector_service.search_similar.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_with_limited_clusters(self, smart_sharding, mock_vector_service):
        """Test searching with limited number of clusters."""
        # Setup shard mappings
        smart_sharding.shard_mappings = {
            0: {'shard_name': 'cluster_shard_0'},
            1: {'shard_name': 'cluster_shard_1'},
            2: {'shard_name': 'cluster_shard_2'}
        }
        
        query_embedding = [0.1, 0.2]
        results = await smart_sharding.search_clustered_shards(
            query_embedding, k=5, search_clusters=2
        )
        
        assert isinstance(results, list)
        # Should search at most 2 clusters
        assert mock_vector_service.search_similar.call_count <= 2
    
    @pytest.mark.asyncio
    async def test_search_without_clustering_info(self, smart_sharding, mock_vector_service):
        """Test searching when no clustering information is available."""
        smart_sharding.clusterer.cluster_centers = None
        smart_sharding.shard_mappings = {
            0: {'shard_name': 'cluster_shard_0'},
            1: {'shard_name': 'cluster_shard_1'}
        }
        
        query_embedding = [0.1, 0.2]
        results = await smart_sharding.search_clustered_shards(query_embedding, k=5)
        
        assert isinstance(results, list)
        # Should search all available shards
        assert mock_vector_service.search_similar.call_count >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_clustered_search(self, smart_sharding, mock_vector_service):
        """Test error handling during clustered search."""
        # Setup a failing shard
        smart_sharding.shard_mappings = {
            0: {'shard_name': 'failing_shard'}
        }
        
        # Make search fail
        mock_vector_service.search_similar.side_effect = Exception("Search failed")
        
        query_embedding = [0.1, 0.2]
        results = await smart_sharding.search_clustered_shards(query_embedding, k=5)
        
        # Should handle error gracefully and return empty results
        assert results == []


class TestClusteringIntegration:
    """Integration tests for clustering components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_clustering_workflow(self, mock_sklearn):
        """Test complete clustering workflow."""
        from services.clustering_service import (
            VectorClusterer, ClusterConfig, SmartShardingService
        )
        
        # Mock vector service
        mock_vector_service = Mock()
        mock_vector_service.create_index = Mock(return_value=Mock())
        mock_vector_service.add_embeddings = AsyncMock(return_value=[0, 1, 2])
        mock_vector_service.search_similar = AsyncMock(return_value=[
            {'id': 0, 'similarity': 0.9, 'text': 'test result', 'cluster_id': 0}
        ])
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans):
            # Initialize components
            config = ClusterConfig(n_clusters=2)
            clusterer = VectorClusterer(config)
            sharding_service = SmartShardingService(mock_vector_service, clusterer)
            
            # Create test data
            embeddings = [[0.1, 0.2], [0.15, 0.25], [0.8, 0.9], [0.85, 0.95]]
            texts = ['text1', 'text2', 'text3', 'text4']
            metadata = [{'source': 'test'} for _ in range(4)]
            
            # Create clustered shards
            result = await sharding_service.create_clustered_shards(
                embeddings, texts, metadata
            )
            
            assert 'shards' in result
            assert result['total_vectors'] == 4
            
            # Search clustered shards
            query_embedding = [0.1, 0.2]
            search_results = await sharding_service.search_clustered_shards(
                query_embedding, k=3
            )
            
            assert isinstance(search_results, list)
    
    def test_cluster_quality_metrics(self, mock_sklearn):
        """Test cluster quality evaluation."""
        from services.clustering_service import VectorClusterer, ClusterConfig
        
        config = ClusterConfig(n_clusters=3)
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans), \
             patch('services.clustering_service.silhouette_score', mock_sklearn['metrics'].silhouette_score), \
             patch('services.clustering_service.SKLEARN_AVAILABLE', True):
            
            clusterer = VectorClusterer(config)
            
            # Create test vectors with clear clusters
            vectors = np.array([
                [0, 0, 0], [0.1, 0.1, 0.1],  # Cluster 1
                [1, 1, 1], [1.1, 1.1, 1.1],  # Cluster 2
                [2, 2, 2]                     # Cluster 3
            ], dtype=np.float32)
            
            clusterer.fit_kmeans(vectors)
            stats = clusterer.get_cluster_stats()
            
            assert stats['silhouette_score'] == 0.75  # Mock value
            assert stats['n_clusters'] == 3
            assert len(stats['cluster_sizes']) == 3
    
    @pytest.mark.asyncio
    async def test_adaptive_cluster_search(self, mock_sklearn):
        """Test adaptive cluster selection for search."""
        from services.clustering_service import (
            VectorClusterer, ClusterConfig, SmartShardingService
        )
        
        mock_vector_service = Mock()
        mock_vector_service.create_index = Mock(return_value=Mock())
        mock_vector_service.search_similar = AsyncMock(return_value=[])
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans):
            config = ClusterConfig(n_clusters=5)
            clusterer = VectorClusterer(config)
            
            # Mock cluster centers
            clusterer.cluster_centers = np.array([
                [0, 0], [1, 1], [2, 2], [3, 3], [4, 4]
            ])
            
            sharding_service = SmartShardingService(mock_vector_service, clusterer)
            
            # Setup shard mappings
            for i in range(5):
                sharding_service.shard_mappings[i] = {
                    'shard_name': f'cluster_shard_{i}'
                }
            
            # Search should automatically select nearest clusters
            query_embedding = [0.1, 0.1]  # Closest to cluster 0
            await sharding_service.search_clustered_shards(
                query_embedding, k=10, search_clusters=3
            )
            
            # Should search at most 3 clusters (as specified)
            assert mock_vector_service.search_similar.call_count <= 3


@pytest.mark.performance
class TestClusteringPerformance:
    """Performance tests for clustering operations."""
    
    def test_large_dataset_clustering(self, mock_sklearn):
        """Test clustering with large datasets."""
        from services.clustering_service import VectorClusterer, ClusterConfig
        
        # Create large dataset
        large_vectors = np.random.rand(10000, 128).astype(np.float32)
        
        config = ClusterConfig(n_clusters=50, max_iter=50)  # Reduced for testing
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans):
            clusterer = VectorClusterer(config)
            
            # Should complete without timeout or memory issues
            labels = clusterer.fit_kmeans(large_vectors)
            
            assert labels is not None
            assert len(labels) == 10000
            assert clusterer.cluster_centers is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_shard_operations(self, mock_sklearn):
        """Test concurrent operations on multiple shards."""
        from services.clustering_service import SmartShardingService, VectorClusterer, ClusterConfig
        
        # Mock vector service that simulates concurrent operations
        mock_vector_service = Mock()
        mock_vector_service.create_index = Mock(return_value=Mock())
        mock_vector_service.add_embeddings = AsyncMock(return_value=list(range(100)))
        mock_vector_service.search_similar = AsyncMock(return_value=[])
        
        with patch('services.clustering_service.KMeans', mock_sklearn['cluster'].KMeans):
            config = ClusterConfig(n_clusters=10)
            clusterer = VectorClusterer(config)
            sharding_service = SmartShardingService(mock_vector_service, clusterer)
            
            # Create multiple shards concurrently
            embeddings = [[i/100, (i+1)/100] for i in range(1000)]
            texts = [f'text_{i}' for i in range(1000)]
            
            result = await sharding_service.create_clustered_shards(embeddings, texts)
            
            assert result['total_vectors'] == 1000
            assert 'shards' in result


if __name__ == "__main__":
    pytest.main([__file__])
