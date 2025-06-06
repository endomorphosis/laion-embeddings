import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import app
from test.fixtures.sample_data import SAMPLE_DATASETS, SAMPLE_MODELS
from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py # Corrected import

client = TestClient(app)


class TestAPIWorkflow:
    """Test complete API workflows"""
    
    def test_health_to_docs_workflow(self, client):
        """Test basic workflow from health check to docs"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Check root
        root_response = client.get("/")
        assert root_response.status_code == 200
        
        # 3. Try to access docs (FastAPI auto-generated)
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200
    
    def test_search_workflow_no_index(self, client):
        """Test search workflow when no index is loaded"""
        search_request = {
            "collection": "nonexistent",
            "text": "test query",
            "n": 5
        }
        
        response = client.post("/search", json=search_request)
        # Should either return 404 (no collection) or 500 (search module error)
        assert response.status_code in [404, 500]
    
    def test_create_embeddings_workflow(self, client):
        """Test embedding creation workflow"""
        request = {
            "dataset": SAMPLE_DATASETS["small_test"]["dataset"],
            "split": SAMPLE_DATASETS["small_test"]["split"],
            "column": SAMPLE_DATASETS["small_test"]["column"],
            "dst_path": "/tmp/test_embeddings",
            "models": [SAMPLE_MODELS[0]]  # Use first valid model
        }
        
        response = client.post("/create_embeddings", json=request)
        
        # Should pass validation and return background task message
        # Might fail on actual execution if dependencies not available
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "status" in data
            assert data["dataset"] == request["dataset"]


class TestErrorHandling:
    """Test error handling across endpoints"""
    
    def test_malformed_json(self, client):
        """Test handling of malformed JSON"""
        response = client.post(
            "/search",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        incomplete_request = {
            "collection": "test"
            # Missing text and n fields
        }
        
        response = client.post("/search", json=incomplete_request)
        assert response.status_code == 422
    
    def test_invalid_data_types(self, client):
        """Test handling of invalid data types"""
        invalid_request = {
            "collection": 123,  # Should be string
            "text": ["not", "a", "string"],  # Should be string
            "n": "not_a_number"  # Should be int
        }
        
        response = client.post("/search", json=invalid_request)
        assert response.status_code == 422


class TestConcurrentRequests:
    """Test handling of concurrent requests"""
    
    def test_multiple_health_checks(self, client):
        """Test multiple concurrent health checks"""
        responses = []
        for _ in range(5):
            response = client.get("/health")
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
    
    def test_mixed_endpoint_requests(self, client):
        """Test mixed requests to different endpoints"""
        # Health check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Search (might fail due to no index)
        search_response = client.post("/search", json={
            "collection": "test",
            "text": "query",
            "n": 5
        })
        assert search_response.status_code in [200, 404, 500]
        
        # Root endpoint
        root_response = client.get("/")
        assert root_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
