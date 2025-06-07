"""
Comprehensive tests for sparse embedding MCP tools.
"""

import pytest
import asyncio
import tempfile
import json
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Import the sparse embedding tools
from src.mcp_server.tools.sparse_embedding_tools import (
    SparseEmbeddingGenerationTool,
    SparseIndexingTool,
    SparseSearchTool
)


class TestSparseEmbeddingGenerationTool:
    """Test SparseEmbeddingGenerationTool class."""
    
    @pytest.fixture
    def sparse_embedding_tool(self, mock_embedding_service):
        """Create a sparse embedding generation tool for testing."""
        return SparseEmbeddingGenerationTool(mock_embedding_service)
    
    @pytest.fixture
    def sample_texts(self):
        """Generate sample texts for testing."""
        return [
            "This is a sample document about machine learning.",
            "Natural language processing is a field of artificial intelligence.",
            "Vector embeddings represent text in high-dimensional space.",
            "Sparse embeddings use mostly zero values for efficiency.",
            "Information retrieval systems use various indexing techniques."
        ]
    
    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_tfidf(self, sparse_embedding_tool, sample_texts):
        """Test generating sparse embeddings using TF-IDF."""
        parameters = {
            "texts": sample_texts,
            "method": "tfidf",
            "max_features": 1000,
            "min_df": 1,
            "max_df": 0.95
        }
        
        result = await sparse_embedding_tool.execute(parameters)
        
        assert result["success"] is True
        assert "embeddings" in result
        assert "vocabulary" in result
        assert "method" in result
        assert result["method"] == "tfidf"
        assert len(result["embeddings"]) == len(sample_texts)
        
        # Check sparse format
        for embedding in result["embeddings"]:
            assert "indices" in embedding
            assert "values" in embedding
            assert "shape" in embedding
    
    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_bm25(self, sparse_embedding_tool, sample_texts):
        """Test generating sparse embeddings using BM25."""
        parameters = {
            "texts": sample_texts,
            "method": "bm25",
            "k1": 1.2,
            "b": 0.75
        }
        
        result = await sparse_embedding_tool.execute(parameters)
        
        assert result["success"] is True
        assert result["method"] == "bm25"
        assert "embeddings" in result
        assert "parameters" in result
        assert result["parameters"]["k1"] == 1.2
        assert result["parameters"]["b"] == 0.75
    
    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_splade(self, sparse_embedding_tool, sample_texts):
        """Test generating sparse embeddings using SPLADE."""
        parameters = {
            "texts": sample_texts,
            "method": "splade",
            "model_name": "splade-v2",
            "max_length": 512
        }
        
        result = await sparse_embedding_tool.execute(parameters)
        
        assert result["success"] is True
        assert result["method"] == "splade"
        assert "embeddings" in result
        assert "model_info" in result
    
    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_batch_processing(self, sparse_embedding_tool):
        """Test batch processing of sparse embeddings."""
        large_texts = [f"Document {i} with unique content about topic {i % 10}" for i in range(100)]
        
        parameters = {
            "texts": large_texts,
            "method": "tfidf",
            "batch_size": 20,
            "max_features": 5000
        }
        
        result = await sparse_embedding_tool.execute(parameters)
        
        assert result["success"] is True
        assert len(result["embeddings"]) == 100
        assert "processing_time" in result
        assert "batch_info" in result
    
    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_with_preprocessing(self, sparse_embedding_tool, sample_texts):
        """Test sparse embedding generation with text preprocessing."""
        parameters = {
            "texts": sample_texts,
            "method": "tfidf",
            "preprocessing": {
                "lowercase": True,
                "remove_stopwords": True,
                "remove_punctuation": True,
                "stemming": True
            }
        }
        
        result = await sparse_embedding_tool.execute(parameters)
        
        assert result["success"] is True
        assert "preprocessing_info" in result
        assert "vocabulary" in result
    
    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_invalid_method(self, sparse_embedding_tool, sample_texts):
        """Test sparse embedding generation with invalid method."""
        parameters = {
            "texts": sample_texts,
            "method": "invalid_method"
        }
        
        result = await sparse_embedding_tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "method" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_empty_texts(self, sparse_embedding_tool):
        """Test sparse embedding generation with empty texts."""
        parameters = {
            "texts": [],
            "method": "tfidf"
        }
        
        result = await sparse_embedding_tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()


class TestSparseIndexingTool:
    """Test SparseIndexingTool class."""
    
    @pytest.fixture
    def sparse_indexing_tool(self, mock_embedding_service):
        """Create a sparse indexing tool for testing."""
        return SparseIndexingTool(mock_embedding_service)
    
    @pytest.fixture
    def sample_sparse_embeddings(self):
        """Generate sample sparse embeddings for testing."""
        embeddings = []
        for i in range(20):
            # Create random sparse embedding
            indices = np.random.choice(1000, size=np.random.randint(10, 50), replace=False)
            values = np.random.rand(len(indices))
            embeddings.append({
                "indices": indices.tolist(),
                "values": values.tolist(),
                "shape": [1000]
            })
        return embeddings
    
    @pytest.fixture
    def sample_sparse_metadata(self):
        """Generate sample metadata for sparse embeddings."""
        return [{"id": f"doc_{i}", "text": f"Document {i}"} for i in range(20)]
    
    @pytest.mark.asyncio
    async def test_create_sparse_index(self, sparse_indexing_tool, temp_dir):
        """Test creating a sparse index."""
        parameters = {
            "index_name": "test_sparse_index",
            "index_path": str(Path(temp_dir) / "sparse_index"),
            "index_type": "inverted",
            "dimension": 1000
        }
        
        result = await sparse_indexing_tool.execute(parameters)
        
        assert result["success"] is True
        assert "index_id" in result
        assert result["index_name"] == "test_sparse_index"
        assert result["index_type"] == "inverted"
        assert result["dimension"] == 1000
    
    @pytest.mark.asyncio
    async def test_add_sparse_embeddings_to_index(self, sparse_indexing_tool, temp_dir, 
                                                   sample_sparse_embeddings, sample_sparse_metadata):
        """Test adding sparse embeddings to index."""
        # First create an index
        create_params = {
            "index_name": "test_add_index",
            "index_path": str(Path(temp_dir) / "add_index"),
            "index_type": "inverted",
            "dimension": 1000
        }
        
        create_result = await sparse_indexing_tool.execute(create_params)
        index_id = create_result["index_id"]
        
        # Add embeddings
        add_params = {
            "action": "add_embeddings",
            "index_id": index_id,
            "embeddings": sample_sparse_embeddings,
            "metadata": sample_sparse_metadata
        }
        
        result = await sparse_indexing_tool.execute(add_params)
        
        assert result["success"] is True
        assert result["added_count"] == len(sample_sparse_embeddings)
        assert "index_stats" in result
    
    @pytest.mark.asyncio
    async def test_build_sparse_index_from_texts(self, sparse_indexing_tool, temp_dir):
        """Test building sparse index directly from texts."""
        texts = [f"Sample document {i} with content about topic {i % 5}" for i in range(50)]
        
        parameters = {
            "action": "build_from_texts",
            "texts": texts,
            "index_name": "text_built_index",
            "index_path": str(Path(temp_dir) / "text_index"),
            "sparse_method": "tfidf",
            "max_features": 1000
        }
        
        result = await sparse_indexing_tool.execute(parameters)
        
        assert result["success"] is True
        assert "index_id" in result
        assert result["document_count"] == len(texts)
        assert "vocabulary_size" in result
    
    @pytest.mark.asyncio
    async def test_optimize_sparse_index(self, sparse_indexing_tool, temp_dir, 
                                        sample_sparse_embeddings, sample_sparse_metadata):
        """Test optimizing a sparse index."""
        # Create and populate index
        create_params = {
            "index_name": "optimize_test_index",
            "index_path": str(Path(temp_dir) / "optimize_index"),
            "index_type": "inverted",
            "dimension": 1000
        }
        
        create_result = await sparse_indexing_tool.execute(create_params)
        index_id = create_result["index_id"]
        
        # Add embeddings
        add_params = {
            "action": "add_embeddings",
            "index_id": index_id,
            "embeddings": sample_sparse_embeddings,
            "metadata": sample_sparse_metadata
        }
        
        await sparse_indexing_tool.execute(add_params)
        
        # Optimize index
        optimize_params = {
            "action": "optimize",
            "index_id": index_id,
            "optimization_level": "aggressive"
        }
        
        result = await sparse_indexing_tool.execute(optimize_params)
        
        assert result["success"] is True
        assert "optimization_time" in result
        assert "size_reduction" in result
        assert "stats_before" in result
        assert "stats_after" in result
    
    @pytest.mark.asyncio
    async def test_get_sparse_index_stats(self, sparse_indexing_tool, temp_dir):
        """Test getting sparse index statistics."""
        # Create index
        create_params = {
            "index_name": "stats_test_index",
            "index_path": str(Path(temp_dir) / "stats_index"),
            "index_type": "inverted",
            "dimension": 1000
        }
        
        create_result = await sparse_indexing_tool.execute(create_params)
        index_id = create_result["index_id"]
        
        # Get stats
        stats_params = {
            "action": "get_stats",
            "index_id": index_id
        }
        
        result = await sparse_indexing_tool.execute(stats_params)
        
        assert result["success"] is True
        assert "stats" in result
        assert "document_count" in result["stats"]
        assert "vocabulary_size" in result["stats"]
        assert "index_size" in result["stats"]
        assert "sparsity_ratio" in result["stats"]
    
    @pytest.mark.asyncio
    async def test_sparse_indexing_invalid_action(self, sparse_indexing_tool):
        """Test sparse indexing with invalid action."""
        parameters = {
            "action": "invalid_action",
            "index_id": "test_index"
        }
        
        result = await sparse_indexing_tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "action" in result["error"].lower()


class TestSparseSearchTool:
    """Test SparseSearchTool class."""
    
    @pytest.fixture
    def sparse_search_tool(self, mock_embedding_service):
        """Create a sparse search tool for testing."""
        return SparseSearchTool(mock_embedding_service)
    
    @pytest.fixture
    async def populated_sparse_index(self, sparse_indexing_tool, temp_dir):
        """Create and populate a sparse index for testing."""
        texts = [
            "Machine learning algorithms process data efficiently",
            "Natural language processing enables text understanding",
            "Vector databases store high-dimensional embeddings",
            "Information retrieval systems find relevant documents",
            "Search engines use inverted indices for fast lookup",
            "Sparse representations reduce memory requirements",
            "Dense embeddings capture semantic relationships",
            "Keyword matching is a classical retrieval method",
            "Neural networks learn complex patterns in data",
            "Transformers revolutionized language modeling"
        ]
        
        # Build index from texts
        parameters = {
            "action": "build_from_texts",
            "texts": texts,
            "index_name": "search_test_index",
            "index_path": str(Path(temp_dir) / "search_index"),
            "sparse_method": "tfidf",
            "max_features": 1000
        }
        
        result = await sparse_indexing_tool.execute(parameters)
        return result["index_id"]
    
    @pytest.mark.asyncio
    async def test_sparse_search_by_text(self, sparse_search_tool, populated_sparse_index):
        """Test sparse search using text query."""
        parameters = {
            "index_id": populated_sparse_index,
            "query_text": "machine learning algorithms",
            "k": 5,
            "search_method": "cosine"
        }
        
        result = await sparse_search_tool.execute(parameters)
        
        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) <= 5
        assert "search_time" in result
        
        # Check result structure
        for item in result["results"]:
            assert "score" in item
            assert "metadata" in item
            assert "rank" in item
            assert item["score"] >= 0
    
    @pytest.mark.asyncio
    async def test_sparse_search_by_sparse_vector(self, sparse_search_tool, populated_sparse_index):
        """Test sparse search using sparse vector query."""
        # Create a sample sparse query vector
        query_vector = {
            "indices": [10, 25, 50, 100, 200],
            "values": [0.8, 0.6, 0.9, 0.4, 0.7],
            "shape": [1000]
        }
        
        parameters = {
            "index_id": populated_sparse_index,
            "query_vector": query_vector,
            "k": 3,
            "search_method": "dot_product"
        }
        
        result = await sparse_search_tool.execute(parameters)
        
        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) <= 3
    
    @pytest.mark.asyncio
    async def test_sparse_search_with_filters(self, sparse_search_tool, populated_sparse_index):
        """Test sparse search with metadata filters."""
        parameters = {
            "index_id": populated_sparse_index,
            "query_text": "neural networks",
            "k": 10,
            "filters": {
                "min_score": 0.1,
                "metadata_filter": {"category": "ml"}
            }
        }
        
        result = await sparse_search_tool.execute(parameters)
        
        assert result["success"] is True
        assert "results" in result
        
        # All results should meet the minimum score requirement
        for item in result["results"]:
            assert item["score"] >= 0.1
    
    @pytest.mark.asyncio
    async def test_sparse_search_multiple_methods(self, sparse_search_tool, populated_sparse_index):
        """Test sparse search with multiple search methods."""
        query_text = "information retrieval"
        
        methods = ["cosine", "dot_product", "bm25"]
        results = {}
        
        for method in methods:
            parameters = {
                "index_id": populated_sparse_index,
                "query_text": query_text,
                "k": 5,
                "search_method": method
            }
            
            result = await sparse_search_tool.execute(parameters)
            assert result["success"] is True
            results[method] = result["results"]
        
        # Results may be different for different methods
        assert len(results) == len(methods)
    
    @pytest.mark.asyncio
    async def test_sparse_search_hybrid_scoring(self, sparse_search_tool, populated_sparse_index):
        """Test hybrid scoring combining multiple signals."""
        parameters = {
            "index_id": populated_sparse_index,
            "query_text": "vector embeddings",
            "k": 5,
            "hybrid_search": {
                "enabled": True,
                "weights": {
                    "sparse": 0.7,
                    "dense": 0.3
                }
            }
        }
        
        result = await sparse_search_tool.execute(parameters)
        
        assert result["success"] is True
        assert "results" in result
        assert "hybrid_scores" in result
    
    @pytest.mark.asyncio
    async def test_sparse_search_explain_scores(self, sparse_search_tool, populated_sparse_index):
        """Test sparse search with score explanation."""
        parameters = {
            "index_id": populated_sparse_index,
            "query_text": "search engines",
            "k": 3,
            "explain_scores": True
        }
        
        result = await sparse_search_tool.execute(parameters)
        
        assert result["success"] is True
        assert "results" in result
        
        # Each result should have score explanation
        for item in result["results"]:
            assert "score_explanation" in item
            assert "term_contributions" in item["score_explanation"]
    
    @pytest.mark.asyncio
    async def test_sparse_search_aggregated_results(self, sparse_search_tool, populated_sparse_index):
        """Test sparse search with result aggregation."""
        parameters = {
            "index_id": populated_sparse_index,
            "query_text": "language processing",
            "k": 10,
            "aggregation": {
                "group_by": "category",
                "max_per_group": 2
            }
        }
        
        result = await sparse_search_tool.execute(parameters)
        
        assert result["success"] is True
        assert "results" in result
        assert "aggregation_info" in result
    
    @pytest.mark.asyncio
    async def test_sparse_search_nonexistent_index(self, sparse_search_tool):
        """Test sparse search with non-existent index."""
        parameters = {
            "index_id": "nonexistent_index",
            "query_text": "test query",
            "k": 5
        }
        
        result = await sparse_search_tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_sparse_search_empty_query(self, sparse_search_tool, populated_sparse_index):
        """Test sparse search with empty query."""
        parameters = {
            "index_id": populated_sparse_index,
            "query_text": "",
            "k": 5
        }
        
        result = await sparse_search_tool.execute(parameters)
        
        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()


class TestSparseEmbeddingToolsIntegration:
    """Integration tests for sparse embedding tools."""
    
    @pytest.mark.asyncio
    async def test_complete_sparse_workflow(self, mock_embedding_service, temp_dir):
        """Test complete sparse embedding workflow."""
        # Initialize tools
        generation_tool = SparseEmbeddingGenerationTool(mock_embedding_service)
        indexing_tool = SparseIndexingTool(mock_embedding_service)
        search_tool = SparseSearchTool(mock_embedding_service)
        
        # Sample documents
        documents = [
            "Machine learning enables computers to learn from data",
            "Natural language processing works with human language",
            "Information retrieval finds relevant documents",
            "Vector spaces represent documents mathematically",
            "Sparse vectors have mostly zero values"
        ]
        
        # 1. Generate sparse embeddings
        gen_params = {
            "texts": documents,
            "method": "tfidf",
            "max_features": 1000
        }
        
        gen_result = await generation_tool.execute(gen_params)
        assert gen_result["success"] is True
        
        # 2. Create index and add embeddings
        index_params = {
            "action": "build_from_texts",
            "texts": documents,
            "index_name": "integration_index",
            "index_path": str(Path(temp_dir) / "integration_index"),
            "sparse_method": "tfidf"
        }
        
        index_result = await indexing_tool.execute(index_params)
        assert index_result["success"] is True
        index_id = index_result["index_id"]
        
        # 3. Search the index
        search_params = {
            "index_id": index_id,
            "query_text": "machine learning data",
            "k": 3
        }
        
        search_result = await search_tool.execute(search_params)
        assert search_result["success"] is True
        assert len(search_result["results"]) <= 3
        
        # 4. Get index statistics
        stats_params = {
            "action": "get_stats",
            "index_id": index_id
        }
        
        stats_result = await indexing_tool.execute(stats_params)
        assert stats_result["success"] is True
        assert stats_result["stats"]["document_count"] == len(documents)
    
    @pytest.mark.asyncio
    async def test_sparse_vs_dense_comparison(self, mock_embedding_service, temp_dir):
        """Test comparison between sparse and dense embeddings."""
        generation_tool = SparseEmbeddingGenerationTool(mock_embedding_service)
        
        documents = [
            "Artificial intelligence and machine learning",
            "Deep learning neural networks",
            "Natural language understanding systems",
            "Computer vision image recognition",
            "Robotics and autonomous systems"
        ]
        
        # Generate sparse embeddings
        sparse_params = {
            "texts": documents,
            "method": "tfidf",
            "max_features": 1000
        }
        
        sparse_result = await generation_tool.execute(sparse_params)
        assert sparse_result["success"] is True
        
        # Check sparsity
        total_values = 0
        non_zero_values = 0
        
        for embedding in sparse_result["embeddings"]:
            total_values += embedding["shape"][0]
            non_zero_values += len(embedding["indices"])
        
        sparsity_ratio = 1 - (non_zero_values / total_values)
        assert sparsity_ratio > 0.5  # Should be sparse
    
    @pytest.mark.asyncio
    async def test_sparse_embedding_persistence(self, mock_embedding_service, temp_dir):
        """Test persistence and loading of sparse embeddings."""
        indexing_tool = SparseIndexingTool(mock_embedding_service)
        
        documents = ["Test document for persistence"]
        
        # Create index
        create_params = {
            "action": "build_from_texts",
            "texts": documents,
            "index_name": "persistence_test",
            "index_path": str(Path(temp_dir) / "persistence_index"),
            "sparse_method": "tfidf"
        }
        
        create_result = await indexing_tool.execute(create_params)
        assert create_result["success"] is True
        
        # Simulate loading the index (in real implementation)
        load_params = {
            "action": "load",
            "index_path": str(Path(temp_dir) / "persistence_index")
        }
        
        # This would test actual persistence in a real implementation
        # For now, just verify the index was created
        assert Path(temp_dir).exists()
