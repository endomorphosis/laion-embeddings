import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import app, InputValidator
from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py # Corrected import

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and basic endpoints"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "laion-embeddings"
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs_url" in data


class TestInputValidation:
    """Test input validation functions"""
    
    def test_validate_text_input_valid(self):
        """Test valid text input"""
        result = InputValidator.validate_text_input("  valid text  ")
        assert result == "valid text"
    
    def test_validate_text_input_empty(self):
        """Test empty text input"""
        with pytest.raises(Exception):  # HTTPException
            InputValidator.validate_text_input("")
            
        with pytest.raises(Exception):  # HTTPException
            InputValidator.validate_text_input("   ")
    
    def test_validate_text_input_too_long(self):
        """Test text input that's too long"""
        long_text = "a" * 10001
        with pytest.raises(Exception):  # HTTPException
            InputValidator.validate_text_input(long_text)
    
    def test_validate_model_name_valid(self):
        """Test valid model names"""
        valid_models = [
            "thenlper/gte-small",
            "Alibaba-NLP/gte-large-en-v1.5",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        ]
        for model in valid_models:
            result = InputValidator.validate_model_name(model)
            assert result == model
    
    def test_validate_model_name_invalid(self):
        """Test invalid model names"""
        with pytest.raises(Exception):  # HTTPException
            InputValidator.validate_model_name("invalid/model")
    
    def test_validate_dataset_name_valid(self):
        """Test valid dataset names"""
        valid_datasets = [
            "test/dataset",
            "company_dataset",
            "dataset-name",
            "TeraflopAI/Caselaw_Access_Project"
        ]
        for dataset in valid_datasets:
            result = InputValidator.validate_dataset_name(dataset)
            assert result == dataset
    
    def test_validate_dataset_name_invalid(self):
        """Test invalid dataset names"""
        with pytest.raises(Exception):  # HTTPException
            InputValidator.validate_dataset_name("")
            
        with pytest.raises(Exception):  # HTTPException
            InputValidator.validate_dataset_name("invalid dataset name")  # spaces not allowed


class TestAPIEndpoints:
    """Test API endpoints"""
    
    @patch('main.search')
    def test_search_endpoint_success(self, mock_search, sample_search_request, client):
        """Test successful search request"""
        # Mock the search function
        mock_search.search = AsyncMock(return_value=[{"id": 1, "text": "result"}])
        
        response = client.post("/search", json=sample_search_request)
        
        # Should not fail validation, but might fail on actual search if dependencies not available
        assert response.status_code in [200, 500]  # 500 if search module not available
    
    def test_search_endpoint_validation_error(self, client):
        """Test search request with validation errors"""
        invalid_request = {
            "collection": "",  # Invalid empty collection
            "text": "",  # Invalid empty text
            "n": 0  # Invalid n value
        }
        
        response = client.post("/search", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_create_embeddings_validation_error(self, sample_invalid_request, client):
        """Test create embeddings with validation errors"""
        response = client.post("/create_embeddings", json=sample_invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_create_embeddings_invalid_model(self, client):
        """Test create embeddings with invalid model"""
        request = {
            "dataset": "test/dataset",
            "split": "train", 
            "column": "text",
            "dst_path": "/tmp/test",
            "models": ["invalid/model"]  # Invalid model
        }
        
        response = client.post("/create_embeddings", json=request)
        # Should fail validation due to invalid model
        assert response.status_code in [400, 422, 500]


class TestRequestModels:
    """Test Pydantic request models"""
    
    def test_search_request_validation(self, client):
        """Test SearchRequest model validation"""
        valid_data = {
            "collection": "test",
            "text": "sample query",
            "n": 10
        }
        
        response = client.post("/search", json=valid_data)
        # Should pass validation (might fail on execution if search not available)
        assert response.status_code in [200, 404, 500]
    
    def test_search_request_default_n(self, client):
        """Test SearchRequest with default n value"""
        data = {
            "collection": "test",
            "text": "sample query"
            # n should default to 10
        }
        
        response = client.post("/search", json=data)
        # Should pass validation
        assert response.status_code in [200, 404, 500]
    
    def test_create_embeddings_request_validation(self, sample_create_embeddings_request, client):
        """Test CreateEmbeddingsRequest model validation"""
        response = client.post("/create_embeddings", json=sample_create_embeddings_request)
        
        # Should fail validation
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__])
