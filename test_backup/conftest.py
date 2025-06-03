import pytest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_search_request():
    """Sample search request for testing"""
    return {
        "collection": "test_collection",
        "text": "sample query text",
        "n": 5
    }

@pytest.fixture
def sample_create_embeddings_request():
    """Sample create embeddings request for testing"""
    return {
        "dataset": "test/dataset",
        "split": "train",
        "column": "text",
        "dst_path": "/tmp/test_embeddings",
        "models": ["thenlper/gte-small"]
    }

@pytest.fixture
def sample_invalid_request():
    """Sample invalid request for testing validation"""
    return {
        "dataset": "",  # Invalid empty dataset
        "split": "train",
        "column": "text",
        "dst_path": "/tmp/test",
        "models": []  # Invalid empty models
    }
