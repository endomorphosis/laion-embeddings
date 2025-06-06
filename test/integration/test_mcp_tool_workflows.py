import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import MCP tools for integration testing
from src.mcp_server.tools.embedding_tools import EmbeddingGenerationTool, BatchEmbeddingTool
from src.mcp_server.tools.search_tools import SemanticSearchTool, SimilaritySearchTool
from src.mcp_server.tools.storage_tools import StorageManagementTool, CollectionManagementTool
from src.mcp_server.tools.analysis_tools import ClusterAnalysisTool, QualityAssessmentTool
from src.mcp_server.tools.sparse_embedding_tools import SparseEmbeddingGenerationTool, SparseSearchTool
from src.mcp_server.tools.ipfs_cluster_tools import IPFSClusterTool, DistributedVectorTool
from src.mcp_server.tools.session_management_tools import SessionCreationTool, SessionMonitoringTool


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows using multiple MCP tools"""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for all tools"""
        return {
            "embedding_service": MagicMock(),
            "vector_service": MagicMock(),
            "storage_service": MagicMock(),
            "analysis_service": MagicMock(),
            "ipfs_service": MagicMock(),
            "distributed_service": MagicMock()
        }
    
    @pytest.mark.asyncio
    async def test_complete_embedding_pipeline(self, mock_services):
        """Test complete embedding pipeline: create -> store -> search"""
        # Setup mock responses
        mock_services["embedding_service"].generate_embeddings = AsyncMock(return_value={
            "embeddings": [
                {"id": "doc1", "vector": [0.1, 0.2, 0.3], "text": "Sample document 1"},
                {"id": "doc2", "vector": [0.4, 0.5, 0.6], "text": "Sample document 2"}
            ],
            "model": "gte-small",
            "dimensions": 3
        })
        
        mock_services["storage_service"].store_vectors = AsyncMock(return_value={
            "collection_id": "test_collection",
            "stored_count": 2,
            "index_status": "ready"
        })
        
        mock_services["vector_service"].semantic_search = AsyncMock(return_value={
            "results": [
                {"id": "doc1", "score": 0.95, "text": "Sample document 1"},
                {"id": "doc2", "score": 0.87, "text": "Sample document 2"}
            ],
            "query_time": 0.05
        })
        
        # Create tools
        embedding_tool = EmbeddingGenerationTool(mock_services["embedding_service"])
        storage_tool = StorageManagementTool(mock_services["storage_service"])
        search_tool = SemanticSearchTool(mock_services["vector_service"])
        
        # Step 1: Generate embeddings
        embedding_result = await embedding_tool.execute({
            "texts": ["Sample document 1", "Sample document 2"],
            "model": "gte-small"
        })
        
        assert embedding_result["success"] is True
        assert len(embedding_result["embeddings"]) == 2
        
        # Step 2: Store embeddings
        storage_result = await storage_tool.execute({
            "operation": "store",
            "collection_name": "test_collection",
            "vectors": embedding_result["embeddings"]
        })
        
        assert storage_result["success"] is True
        assert storage_result["stored_count"] == 2
        
        # Step 3: Search embeddings
        search_result = await search_tool.execute({
            "query": "sample text query",
            "collection_name": "test_collection",
            "top_k": 5
        })
        
        assert search_result["success"] is True
        assert len(search_result["results"]) == 2
        assert search_result["results"][0]["score"] > 0.9
        
        # Verify all services were called
        mock_services["embedding_service"].generate_embeddings.assert_called_once()
        mock_services["storage_service"].store_vectors.assert_called_once()
        mock_services["vector_service"].semantic_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, mock_services):
        """Test batch processing workflow with large datasets"""
        # Setup mock responses for batch processing
        mock_services["embedding_service"].batch_generate_embeddings = AsyncMock(return_value={
            "batch_id": "batch_123",
            "status": "completed",
            "total_processed": 1000,
            "embeddings": [{"id": f"doc_{i}", "vector": [0.1] * 384} for i in range(1000)],
            "processing_time": 45.2
        })
        
        mock_services["analysis_service"].analyze_quality = AsyncMock(return_value={
            "quality_score": 0.92,
            "dimension_analysis": {"mean": 0.05, "std": 0.23},
            "anomaly_count": 3,
            "recommendations": ["Consider filtering outliers"]
        })
        
        # Create tools
        batch_tool = BatchEmbeddingTool(mock_services["embedding_service"])
        quality_tool = QualityAssessmentTool(mock_services["analysis_service"])
        
        # Step 1: Batch generate embeddings
        batch_result = await batch_tool.execute({
            "dataset": {"texts": [f"Document {i}" for i in range(1000)]},
            "model": "gte-small",
            "batch_size": 100
        })
        
        assert batch_result["success"] is True
        assert batch_result["total_processed"] == 1000
        
        # Step 2: Quality assessment
        quality_result = await quality_tool.execute({
            "vectors": batch_result["embeddings"][:10],  # Sample for analysis
            "analysis_type": "comprehensive"
        })
        
        assert quality_result["success"] is True
        assert quality_result["quality_score"] > 0.9
        assert "recommendations" in quality_result
        
        mock_services["embedding_service"].batch_generate_embeddings.assert_called_once()
        mock_services["analysis_service"].analyze_quality.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sparse_embedding_workflow(self, mock_services):
        """Test sparse embedding generation and search workflow"""
        # Setup mock responses
        mock_services["embedding_service"].generate_sparse_embedding = AsyncMock(return_value={
            "vector": {"0": 0.5, "15": 0.8, "42": 0.3},
            "model": "splade",
            "dimensions": 1000,
            "nnz": 3
        })
        
        mock_services["vector_service"].sparse_search = AsyncMock(return_value={
            "results": [
                {"id": "sparse_doc1", "score": 0.94, "terms": ["query", "term"]},
                {"id": "sparse_doc2", "score": 0.89, "terms": ["search", "text"]}
            ],
            "total": 2,
            "query_time": 0.03
        })
        
        # Create tools
        sparse_embedding_tool = SparseEmbeddingGenerationTool(mock_services["embedding_service"])
        sparse_search_tool = SparseSearchTool(mock_services["vector_service"])
        
        # Step 1: Generate sparse embedding
        sparse_result = await sparse_embedding_tool.execute({
            "text": "machine learning and artificial intelligence",
            "model": "splade"
        })
        
        assert sparse_result["success"] is True
        assert "vector" in sparse_result
        assert sparse_result["nnz"] == 3
        
        # Step 2: Sparse search
        search_result = await sparse_search_tool.execute({
            "query": "machine learning",
            "index_name": "sparse_index",
            "top_k": 10
        })
        
        assert search_result["success"] is True
        assert len(search_result["results"]) == 2
        assert search_result["results"][0]["score"] > 0.9
        
        mock_services["embedding_service"].generate_sparse_embedding.assert_called_once()
        mock_services["vector_service"].sparse_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_distributed_storage_workflow(self, mock_services):
        """Test distributed storage workflow with IPFS clustering"""
        # Setup mock responses
        mock_services["ipfs_service"].create_cluster = AsyncMock(return_value={
            "cluster_id": "QmCluster123",
            "status": "active",
            "nodes": 3,
            "replication_factor": 2
        })
        
        mock_services["distributed_service"].distribute_vectors = AsyncMock(return_value={
            "distribution_id": "dist_456",
            "shard_count": 4,
            "total_vectors": 5000,
            "shards": [
                {"shard_id": "shard_1", "vector_count": 1250, "hash": "QmShard1"},
                {"shard_id": "shard_2", "vector_count": 1250, "hash": "QmShard2"}
            ]
        })
        
        # Create tools
        ipfs_tool = IPFSClusterTool(mock_services["ipfs_service"])
        distributed_tool = DistributedVectorTool(mock_services["distributed_service"])
        
        # Step 1: Create IPFS cluster
        cluster_result = await ipfs_tool.execute({
            "operation": "create",
            "replication_factor": 2,
            "cluster_config": {"nodes": 3}
        })
        
        assert cluster_result["success"] is True
        assert cluster_result["cluster_id"] == "QmCluster123"
        
        # Step 2: Distribute vectors
        distribution_result = await distributed_tool.execute({
            "operation": "distribute",
            "vectors": [{"id": f"vec_{i}", "data": [0.1] * 100} for i in range(5000)],
            "shard_size": 1250,
            "replication_factor": 2
        })
        
        assert distribution_result["success"] is True
        assert distribution_result["shard_count"] == 4
        assert distribution_result["total_vectors"] == 5000
        
        mock_services["ipfs_service"].create_cluster.assert_called_once()
        mock_services["distributed_service"].distribute_vectors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_managed_workflow(self, mock_services):
        """Test workflow with session management"""
        # Setup mock responses
        mock_services["embedding_service"].create_session = AsyncMock(return_value={
            "session_id": "session_789",
            "status": "active",
            "endpoint": "ws://localhost:8080/sessions/session_789"
        })
        
        mock_services["embedding_service"].monitor_sessions = AsyncMock(return_value={
            "session_id": "session_789",
            "status": "active",
            "metrics": {"cpu": {"current_usage": 25.0}, "memory": {"current_mb": 1200}}
        })
        
        mock_services["embedding_service"].generate_embeddings = AsyncMock(return_value={
            "embeddings": [{"id": "session_doc1", "vector": [0.1, 0.2]}],
            "session_id": "session_789"
        })
        
        # Create tools
        session_tool = SessionCreationTool(mock_services["embedding_service"])
        monitor_tool = SessionMonitoringTool(mock_services["embedding_service"])
        embedding_tool = EmbeddingGenerationTool(mock_services["embedding_service"])
        
        # Step 1: Create session
        session_result = await session_tool.execute({
            "session_name": "workflow_session",
            "session_config": {
                "models": ["gte-small"],
                "timeout_seconds": 3600
            }
        })
        
        assert session_result["type"] == "session_creation"
        session_id = session_result["result"]["session_id"]
        
        # Step 2: Monitor session
        monitor_result = await monitor_tool.execute({
            "session_id": session_id,
            "metrics_requested": ["cpu", "memory"]
        })
        
        assert monitor_result["type"] == "session_monitoring"
        assert monitor_result["result"]["status"] == "active"
        
        # Step 3: Use session for embeddings
        embedding_result = await embedding_tool.execute({
            "texts": ["Session-managed document"],
            "model": "gte-small",
            "session_id": session_id
        })
        
        assert embedding_result["success"] is True
        
        # Verify all services were called
        mock_services["embedding_service"].create_session.assert_called_once()
        mock_services["embedding_service"].monitor_sessions.assert_called_once()
        mock_services["embedding_service"].generate_embeddings.assert_called_once()


class TestToolChaining:
    """Test chaining multiple tools together for complex workflows"""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for tool chaining tests"""
        return {
            "embedding_service": MagicMock(),
            "vector_service": MagicMock(),
            "storage_service": MagicMock(),
            "analysis_service": MagicMock()
        }
    
    @pytest.mark.asyncio
    async def test_embedding_analysis_chain(self, mock_services):
        """Test chaining embedding generation -> quality analysis -> clustering"""
        # Setup mock responses
        mock_services["embedding_service"].generate_embeddings = AsyncMock(return_value={
            "embeddings": [{"id": f"doc_{i}", "vector": [0.1 * i] * 384} for i in range(100)],
            "model": "gte-small"
        })
        
        mock_services["analysis_service"].analyze_quality = AsyncMock(return_value={
            "quality_score": 0.88,
            "dimension_analysis": {"variance": 0.15},
            "outlier_indices": [45, 67, 89]
        })
        
        mock_services["analysis_service"].perform_clustering = AsyncMock(return_value={
            "clusters": [
                {"cluster_id": 0, "size": 40, "centroid": [0.2] * 384},
                {"cluster_id": 1, "size": 35, "centroid": [0.5] * 384},
                {"cluster_id": 2, "size": 25, "centroid": [0.8] * 384}
            ],
            "silhouette_score": 0.72
        })
        
        # Create tools
        embedding_tool = EmbeddingGenerationTool(mock_services["embedding_service"])
        quality_tool = QualityAssessmentTool(mock_services["analysis_service"])
        cluster_tool = ClusterAnalysisTool(mock_services["analysis_service"])
        
        # Chain 1: Generate embeddings
        embedding_result = await embedding_tool.execute({
            "texts": [f"Document {i} content" for i in range(100)],
            "model": "gte-small"
        })
        
        embeddings = embedding_result["embeddings"]
        
        # Chain 2: Analyze quality
        quality_result = await quality_tool.execute({
            "vectors": embeddings,
            "analysis_type": "comprehensive"
        })
        
        # Filter out outliers based on quality analysis
        outlier_indices = set(quality_result["outlier_indices"])
        filtered_embeddings = [
            emb for i, emb in enumerate(embeddings) 
            if i not in outlier_indices
        ]
        
        # Chain 3: Perform clustering on filtered embeddings
        cluster_result = await cluster_tool.execute({
            "vectors": filtered_embeddings,
            "algorithm": "kmeans",
            "n_clusters": 3
        })
        
        # Verify results
        assert len(embeddings) == 100
        assert quality_result["quality_score"] > 0.8
        assert len(filtered_embeddings) == 97  # 100 - 3 outliers
        assert len(cluster_result["clusters"]) == 3
        assert cluster_result["silhouette_score"] > 0.7
        
        # Verify all services were called
        mock_services["embedding_service"].generate_embeddings.assert_called_once()
        mock_services["analysis_service"].analyze_quality.assert_called_once()
        mock_services["analysis_service"].perform_clustering.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_refinement_chain(self, mock_services):
        """Test chaining search -> similarity search -> collection management"""
        # Setup mock responses
        mock_services["vector_service"].semantic_search = AsyncMock(return_value={
            "results": [
                {"id": "doc1", "score": 0.95, "text": "Primary result"},
                {"id": "doc2", "score": 0.87, "text": "Secondary result"},
                {"id": "doc3", "score": 0.82, "text": "Tertiary result"}
            ],
            "total": 3
        })
        
        mock_services["vector_service"].similarity_search = AsyncMock(return_value={
            "similar_documents": [
                {"id": "sim1", "similarity": 0.91, "text": "Similar to primary"},
                {"id": "sim2", "similarity": 0.86, "text": "Similar to secondary"}
            ],
            "search_depth": 2
        })
        
        mock_services["storage_service"].create_collection = AsyncMock(return_value={
            "collection_id": "refined_results",
            "status": "created",
            "document_count": 5
        })
        
        # Create tools
        search_tool = SemanticSearchTool(mock_services["vector_service"])
        similarity_tool = SimilaritySearchTool(mock_services["vector_service"])
        collection_tool = CollectionManagementTool(mock_services["storage_service"])
        
        # Chain 1: Initial search
        search_result = await search_tool.execute({
            "query": "machine learning algorithms",
            "collection_name": "main_collection",
            "top_k": 10
        })
        
        primary_results = search_result["results"]
        
        # Chain 2: Find similar documents to top result
        similarity_result = await similarity_tool.execute({
            "reference_vector": primary_results[0]["id"],
            "collection_name": "main_collection",
            "threshold": 0.8
        })
        
        similar_docs = similarity_result["similar_documents"]
        
        # Chain 3: Create refined collection with all results
        all_results = primary_results + similar_docs
        collection_result = await collection_tool.execute({
            "operation": "create",
            "collection_name": "refined_results",
            "documents": all_results,
            "metadata": {"search_query": "machine learning algorithms", "refinement": True}
        })
        
        # Verify results
        assert len(primary_results) == 3
        assert len(similar_docs) == 2
        assert len(all_results) == 5
        assert collection_result["status"] == "created"
        assert collection_result["document_count"] == 5
        
        # Verify all services were called
        mock_services["vector_service"].semantic_search.assert_called_once()
        mock_services["vector_service"].similarity_search.assert_called_once()
        mock_services["storage_service"].create_collection.assert_called_once()


class TestErrorHandlingWorkflows:
    """Test error handling and recovery in complex workflows"""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services with potential failure points"""
        return {
            "embedding_service": MagicMock(),
            "vector_service": MagicMock(),
            "storage_service": MagicMock()
        }
    
    @pytest.mark.asyncio
    async def test_workflow_with_service_failure(self, mock_services):
        """Test workflow handling when intermediate service fails"""
        # Setup mock responses with one failure
        mock_services["embedding_service"].generate_embeddings = AsyncMock(return_value={
            "embeddings": [{"id": "doc1", "vector": [0.1, 0.2]}],
            "model": "gte-small"
        })
        
        # Storage service fails
        mock_services["storage_service"].store_vectors = AsyncMock(
            side_effect=Exception("Storage service unavailable")
        )
        
        # Search service works
        mock_services["vector_service"].semantic_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95}]
        })
        
        # Create tools
        embedding_tool = EmbeddingGenerationTool(mock_services["embedding_service"])
        storage_tool = StorageManagementTool(mock_services["storage_service"])
        search_tool = SemanticSearchTool(mock_services["vector_service"])
        
        # Step 1: Generate embeddings (should succeed)
        embedding_result = await embedding_tool.execute({
            "texts": ["Test document"],
            "model": "gte-small"
        })
        
        assert embedding_result["success"] is True
        
        # Step 2: Try to store embeddings (should fail)
        with pytest.raises(Exception, match="Storage service unavailable"):
            await storage_tool.execute({
                "operation": "store",
                "collection_name": "test_collection",
                "vectors": embedding_result["embeddings"]
            })
        
        # Step 3: Search should still work (if we have fallback data)
        search_result = await search_tool.execute({
            "query": "test query",
            "collection_name": "backup_collection",
            "top_k": 5
        })
        
        assert search_result["success"] is True
        
        # Verify expected service calls
        mock_services["embedding_service"].generate_embeddings.assert_called_once()
        mock_services["storage_service"].store_vectors.assert_called_once()
        mock_services["vector_service"].semantic_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_with_partial_failures(self, mock_services):
        """Test workflow handling with partial failures and retries"""
        # Setup mock for batch processing with partial failures
        call_count = 0
        def batch_with_retries(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: partial failure
                return {
                    "batch_id": "batch_retry_123",
                    "status": "partial_failure",
                    "successful_count": 7,
                    "failed_count": 3,
                    "embeddings": [{"id": f"doc_{i}", "vector": [0.1] * 384} for i in range(7)],
                    "errors": ["Timeout on doc_7", "Timeout on doc_8", "Timeout on doc_9"]
                }
            else:
                # Retry call: success
                return {
                    "batch_id": "batch_retry_124", 
                    "status": "completed",
                    "successful_count": 3,
                    "failed_count": 0,
                    "embeddings": [{"id": f"doc_{i}", "vector": [0.1] * 384} for i in range(7, 10)]
                }
        
        mock_services["embedding_service"].batch_generate_embeddings = AsyncMock(
            side_effect=batch_with_retries
        )
        
        # Create tool
        batch_tool = BatchEmbeddingTool(mock_services["embedding_service"])
        
        # Step 1: Initial batch processing (partial failure)
        batch_result = await batch_tool.execute({
            "dataset": {"texts": [f"Document {i}" for i in range(10)]},
            "model": "gte-small",
            "batch_size": 10
        })
        
        # Verify partial success
        assert batch_result["status"] == "partial_failure"
        assert batch_result["successful_count"] == 7
        assert batch_result["failed_count"] == 3
        
        # Step 2: Retry failed documents
        retry_result = await batch_tool.execute({
            "dataset": {"texts": [f"Document {i}" for i in range(7, 10)]},
            "model": "gte-small",
            "batch_size": 3
        })
        
        # Verify retry success
        assert retry_result["status"] == "completed"
        assert retry_result["successful_count"] == 3
        assert retry_result["failed_count"] == 0
        
        # Verify service was called twice (original + retry)
        assert mock_services["embedding_service"].batch_generate_embeddings.call_count == 2


class TestPerformanceWorkflows:
    """Test performance aspects of tool workflows"""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services with performance metrics"""
        return {
            "embedding_service": MagicMock(),
            "vector_service": MagicMock()
        }
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, mock_services):
        """Test concurrent execution of multiple tools"""
        # Setup mock responses with realistic delays
        async def slow_embedding_generation(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "embeddings": [{"id": "concurrent_doc", "vector": [0.1] * 384}],
                "processing_time": 0.1
            }
        
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.05)  # Simulate search time
            return {
                "results": [{"id": "result1", "score": 0.9}],
                "query_time": 0.05
            }
        
        mock_services["embedding_service"].generate_embeddings = AsyncMock(
            side_effect=slow_embedding_generation
        )
        mock_services["vector_service"].semantic_search = AsyncMock(
            side_effect=slow_search
        )
        
        # Create tools
        embedding_tool = EmbeddingGenerationTool(mock_services["embedding_service"])
        search_tool = SemanticSearchTool(mock_services["vector_service"])
        
        # Execute tools concurrently
        start_time = datetime.now()
        
        tasks = [
            embedding_tool.execute({
                "texts": [f"Concurrent document {i}"],
                "model": "gte-small"
            })
            for i in range(5)
        ] + [
            search_tool.execute({
                "query": f"concurrent query {i}",
                "collection_name": "test_collection",
                "top_k": 5
            })
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Verify results
        assert len(results) == 8  # 5 embedding + 3 search
        assert all(result["success"] for result in results[:5])  # All embeddings succeeded
        assert all(result["success"] for result in results[5:])  # All searches succeeded
        
        # Verify concurrent execution was faster than sequential
        # Sequential would be: 5 * 0.1 + 3 * 0.05 = 0.65 seconds
        # Concurrent should be much less
        assert total_time < 0.3  # Should be much faster than sequential
        
        # Verify all services were called expected number of times
        assert mock_services["embedding_service"].generate_embeddings.call_count == 5
        assert mock_services["vector_service"].semantic_search.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
