# tests/test_mcp_tools/test_analysis_tools.py

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.mcp_server.tools.analysis_tools import (
    ClusterAnalysisTool,
    QualityAssessmentTool,
    DimensionalityReductionTool
)
from src.mcp_server.error_handlers import MCPError, ValidationError


class TestClusterAnalysisTool:
    """Test cases for ClusterAnalysisTool."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        service = Mock()
        service.get_embeddings = AsyncMock(return_value=np.random.rand(100, 384))
        service.get_collection_embeddings = AsyncMock(return_value=np.random.rand(100, 384))
        return service

    @pytest.fixture
    def cluster_tool(self, mock_embedding_service):
        """Create ClusterAnalysisTool instance."""
        tool = ClusterAnalysisTool()
        tool.embedding_service = mock_embedding_service
        return tool

    @pytest.mark.asyncio
    async def test_kmeans_clustering(self, cluster_tool):
        """Test K-means clustering analysis."""
        parameters = {
            "algorithm": "kmeans",
            "data_source": "collection",
            "collection_name": "test_collection",
            "n_clusters": 5,
            "random_state": 42
        }

        with patch('sklearn.cluster.KMeans') as mock_kmeans:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.labels_ = np.array([0, 1, 2, 0, 1] * 20)
            mock_model.cluster_centers_ = np.random.rand(5, 384)
            mock_model.inertia_ = 123.45
            mock_kmeans.return_value = mock_model

            result = await cluster_tool.execute(parameters)

            assert result["status"] == "success"
            assert result["algorithm"] == "kmeans"
            assert result["n_clusters"] == 5
            assert "cluster_labels" in result
            assert "cluster_centers" in result
            assert "metrics" in result
            assert result["metrics"]["inertia"] == 123.45

    @pytest.mark.asyncio
    async def test_dbscan_clustering(self, cluster_tool):
        """Test DBSCAN clustering analysis."""
        parameters = {
            "algorithm": "dbscan",
            "data_source": "ids",
            "embedding_ids": ["emb1", "emb2", "emb3"],
            "eps": 0.5,
            "min_samples": 5
        }

        with patch('sklearn.cluster.DBSCAN') as mock_dbscan:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.labels_ = np.array([0, 0, -1, 1, 1] * 20)
            mock_model.core_sample_indices_ = np.array([0, 1, 3, 4])
            mock_dbscan.return_value = mock_model

            result = await cluster_tool.execute(parameters)

            assert result["status"] == "success"
            assert result["algorithm"] == "dbscan"
            assert "cluster_labels" in result
            assert "n_clusters" in result
            assert "n_noise" in result

    @pytest.mark.asyncio
    async def test_hierarchical_clustering(self, cluster_tool):
        """Test hierarchical clustering analysis."""
        parameters = {
            "algorithm": "hierarchical",
            "data_source": "collection",
            "collection_name": "test_collection",
            "n_clusters": 3,
            "linkage": "ward"
        }

        with patch('sklearn.cluster.AgglomerativeClustering') as mock_agg:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.labels_ = np.array([0, 1, 2, 0, 1] * 20)
            mock_model.n_clusters_ = 3
            mock_agg.return_value = mock_model

            result = await cluster_tool.execute(parameters)

            assert result["status"] == "success"
            assert result["algorithm"] == "hierarchical"
            assert result["n_clusters"] == 3
            assert "cluster_labels" in result

    @pytest.mark.asyncio
    async def test_invalid_algorithm(self, cluster_tool):
        """Test handling of invalid clustering algorithm."""
        parameters = {
            "algorithm": "invalid_algorithm",
            "data_source": "collection",
            "collection_name": "test_collection"
        }

        with pytest.raises(ValidationError):
            await cluster_tool.execute(parameters)

    @pytest.mark.asyncio
    async def test_empty_data_source(self, cluster_tool):
        """Test handling of empty data source."""
        cluster_tool.embedding_service.get_collection_embeddings.return_value = np.array([])

        parameters = {
            "algorithm": "kmeans",
            "data_source": "collection",
            "collection_name": "empty_collection"
        }

        with pytest.raises(MCPError, match="No embeddings found"):
            await cluster_tool.execute(parameters)

    @pytest.mark.parametrize("algorithm,expected_params", [
        ("kmeans", {"n_clusters": 5, "random_state": 42}),
        ("dbscan", {"eps": 0.5, "min_samples": 5}),
        ("gaussian_mixture", {"n_components": 4, "random_state": 42}),
        ("spectral", {"n_clusters": 3, "random_state": 42})
    ])
    @pytest.mark.asyncio
    async def test_algorithm_parameters(self, cluster_tool, algorithm, expected_params):
        """Test algorithm-specific parameter handling."""
        parameters = {
            "algorithm": algorithm,
            "data_source": "collection",
            "collection_name": "test_collection",
            **expected_params
        }

        with patch(f'sklearn.cluster.{algorithm.title().replace("_", "")}' if algorithm != "gaussian_mixture" 
                  else 'sklearn.mixture.GaussianMixture') as mock_class:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.labels_ = np.array([0, 1, 0, 1] * 25)
            mock_class.return_value = mock_model

            result = await cluster_tool.execute(parameters)

            assert result["status"] == "success"
            mock_class.assert_called_once()


class TestQualityAssessmentTool:
    """Test cases for QualityAssessmentTool."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        service = Mock()
        service.get_embeddings = AsyncMock(return_value=np.random.rand(100, 384))
        service.get_collection_embeddings = AsyncMock(return_value=np.random.rand(100, 384))
        return service

    @pytest.fixture
    def quality_tool(self, mock_embedding_service):
        """Create QualityAssessmentTool instance."""
        tool = QualityAssessmentTool()
        tool.embedding_service = mock_embedding_service
        return tool

    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, quality_tool):
        """Test quality metrics calculation."""
        parameters = {
            "data_source": "collection",
            "collection_name": "test_collection",
            "metrics": ["diversity", "coverage", "outliers", "consistency"]
        }

        result = await quality_tool.execute(parameters)

        assert result["status"] == "success"
        assert "quality_metrics" in result
        assert "diversity_score" in result["quality_metrics"]
        assert "coverage_score" in result["quality_metrics"]
        assert "outlier_count" in result["quality_metrics"]
        assert "consistency_score" in result["quality_metrics"]

    @pytest.mark.asyncio
    async def test_outlier_detection(self, quality_tool):
        """Test outlier detection functionality."""
        parameters = {
            "data_source": "collection",
            "collection_name": "test_collection",
            "metrics": ["outliers"],
            "outlier_method": "isolation_forest",
            "contamination": 0.1
        }

        with patch('sklearn.ensemble.IsolationForest') as mock_forest:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([1, 1, -1, 1, -1] * 20)
            mock_forest.return_value = mock_model

            result = await quality_tool.execute(parameters)

            assert result["status"] == "success"
            assert "outlier_indices" in result["quality_metrics"]
            assert "outlier_count" in result["quality_metrics"]

    @pytest.mark.asyncio
    async def test_diversity_calculation(self, quality_tool):
        """Test diversity score calculation."""
        parameters = {
            "data_source": "collection",
            "collection_name": "test_collection",
            "metrics": ["diversity"],
            "diversity_method": "cosine"
        }

        result = await quality_tool.execute(parameters)

        assert result["status"] == "success"
        assert "diversity_score" in result["quality_metrics"]
        assert 0 <= result["quality_metrics"]["diversity_score"] <= 1


class TestDimensionalityReductionTool:
    """Test cases for DimensionalityReductionTool."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        service = Mock()
        service.get_embeddings = AsyncMock(return_value=np.random.rand(100, 384))
        return service

    @pytest.fixture
    def reduction_tool(self, mock_embedding_service):
        """Create DimensionalityReductionTool instance."""
        tool = DimensionalityReductionTool()
        tool.embedding_service = mock_embedding_service
        return tool

    @pytest.mark.asyncio
    async def test_pca_reduction(self, reduction_tool):
        """Test PCA dimensionality reduction."""
        parameters = {
            "method": "pca",
            "data_source": "collection",
            "collection_name": "test_collection",
            "target_dimensions": 50,
            "random_state": 42
        }

        with patch('sklearn.decomposition.PCA') as mock_pca:
            mock_model = Mock()
            mock_model.fit_transform.return_value = np.random.rand(100, 50)
            mock_model.explained_variance_ratio_ = np.random.rand(50)
            mock_model.singular_values_ = np.random.rand(50)
            mock_pca.return_value = mock_model

            result = await reduction_tool.execute(parameters)

            assert result["status"] == "success"
            assert result["method"] == "pca"
            assert result["original_dimensions"] == 384
            assert result["target_dimensions"] == 50
            assert "reduced_embeddings" in result
            assert "explained_variance_ratio" in result

    @pytest.mark.asyncio
    async def test_umap_reduction(self, reduction_tool):
        """Test UMAP dimensionality reduction."""
        parameters = {
            "method": "umap",
            "data_source": "collection",
            "collection_name": "test_collection",
            "target_dimensions": 2,
            "n_neighbors": 15,
            "min_dist": 0.1
        }

        with patch('umap.UMAP') as mock_umap:
            mock_model = Mock()
            mock_model.fit_transform.return_value = np.random.rand(100, 2)
            mock_umap.return_value = mock_model

            result = await reduction_tool.execute(parameters)

            assert result["status"] == "success"
            assert result["method"] == "umap"
            assert result["target_dimensions"] == 2
            assert "reduced_embeddings" in result

    @pytest.mark.asyncio
    async def test_tsne_reduction(self, reduction_tool):
        """Test t-SNE dimensionality reduction."""
        parameters = {
            "method": "tsne",
            "data_source": "collection",
            "collection_name": "test_collection",
            "target_dimensions": 2,
            "perplexity": 30,
            "random_state": 42
        }

        with patch('sklearn.manifold.TSNE') as mock_tsne:
            mock_model = Mock()
            mock_model.fit_transform.return_value = np.random.rand(100, 2)
            mock_tsne.return_value = mock_model

            result = await reduction_tool.execute(parameters)

            assert result["status"] == "success"
            assert result["method"] == "tsne"
            assert "reduced_embeddings" in result

    @pytest.mark.asyncio
    async def test_invalid_method(self, reduction_tool):
        """Test handling of invalid reduction method."""
        parameters = {
            "method": "invalid_method",
            "data_source": "collection",
            "collection_name": "test_collection"
        }

        with pytest.raises(ValidationError):
            await reduction_tool.execute(parameters)


# class TestStatisticalAnalysisTool:
#     """Test cases for StatisticalAnalysisTool."""

#     @pytest.fixture
#     def mock_embedding_service(self):
#         """Mock embedding service."""
#         service = Mock()
#         service.get_embeddings = AsyncMock(return_value=np.random.rand(100, 384))
#         return service

#     @pytest.fixture
#     def stats_tool(self, mock_embedding_service):
#         """Create StatisticalAnalysisTool instance."""
#         tool = StatisticalAnalysisTool()
#         tool.embedding_service = mock_embedding_service
#         return tool

#     @pytest.mark.asyncio
#     async def test_descriptive_statistics(self, stats_tool):
#         """Test descriptive statistics calculation."""
#         parameters = {
#             "data_source": "collection",
#             "collection_name": "test_collection",
#             "analysis_type": "descriptive"
#         }

#         result = await stats_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert "statistics" in result
#         assert "mean" in result["statistics"]
#         assert "std" in result["statistics"]
#         assert "min" in result["statistics"]
#         assert "max" in result["statistics"]

#     @pytest.mark.asyncio
#     async def test_correlation_analysis(self, stats_tool):
#         """Test correlation analysis."""
#         parameters = {
#             "data_source": "collection",
#             "collection_name": "test_collection",
#             "analysis_type": "correlation",
#             "method": "pearson"
#         }

#         result = await stats_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert "correlation_matrix" in result
#         assert "analysis_type" in result
#         assert result["analysis_type"] == "correlation"

#     @pytest.mark.asyncio
#     async def test_distribution_analysis(self, stats_tool):
#         """Test distribution analysis."""
#         parameters = {
#             "data_source": "collection",
#             "collection_name": "test_collection",
#             "analysis_type": "distribution"
#         }

#         result = await stats_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert "distribution_stats" in result
#         assert "normality_test" in result


# class TestSimilarityAnalysisTool:
#     """Test cases for SimilarityAnalysisTool."""

#     @pytest.fixture
#     def mock_embedding_service(self):
#         """Mock embedding service."""
#         service = Mock()
#         service.get_embeddings = AsyncMock(return_value=np.random.rand(100, 384))
#         return service

#     @pytest.fixture
#     def similarity_tool(self, mock_embedding_service):
#         """Create SimilarityAnalysisTool instance."""
#         tool = SimilarityAnalysisTool()
#         tool.embedding_service = mock_embedding_service
#         return tool

#     @pytest.mark.asyncio
#     async def test_cosine_similarity_matrix(self, similarity_tool):
#         """Test cosine similarity matrix calculation."""
#         parameters = {
#             "data_source": "collection",
#             "collection_name": "test_collection",
#             "similarity_metric": "cosine",
#             "analysis_type": "matrix"
#         }

#         result = await similarity_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert "similarity_matrix" in result
#         assert result["similarity_metric"] == "cosine"
#         assert "matrix_shape" in result

#     @pytest.mark.asyncio
#     async def test_pairwise_similarity(self, similarity_tool):
#         """Test pairwise similarity calculation."""
#         parameters = {
#             "data_source": "ids",
#             "embedding_ids": ["emb1", "emb2", "emb3"],
#             "similarity_metric": "euclidean",
#             "analysis_type": "pairwise"
#         }

#         result = await similarity_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert "pairwise_similarities" in result
#         assert result["similarity_metric"] == "euclidean"

#     @pytest.mark.asyncio
#     async def test_nearest_neighbors(self, similarity_tool):
#         """Test nearest neighbors analysis."""
#         parameters = {
#             "data_source": "collection",
#             "collection_name": "test_collection",
#             "similarity_metric": "cosine",
#             "analysis_type": "nearest_neighbors",
#             "k": 5
#         }

#         result = await similarity_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert "nearest_neighbors" in result
#         assert "k" in result
#         assert result["k"] == 5

#     @pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan", "dot_product"])
#     @pytest.mark.asyncio
#     async def test_similarity_metrics(self, similarity_tool, metric):
#         """Test different similarity metrics."""
#         parameters = {
#             "data_source": "collection",
#             "collection_name": "test_collection",
#             "similarity_metric": metric,
#             "analysis_type": "matrix"
#         }

#         result = await similarity_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["similarity_metric"] == metric
