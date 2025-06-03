"""
Test configuration for integration tests.

This conftest provides test fixtures and configuration without importing
the main application to avoid dependency conflicts.
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Configure test environment
@pytest.fixture(scope="session", autouse=True)
def setup_testing_environment():
    """Set up the testing environment for integration tests."""
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['DISABLE_TELEMETRY'] = 'true'
    
    # Set up test data directory
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after tests
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir, ignore_errors=True)


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.random((100, 384)).astype(np.float32)


@pytest.fixture
def sample_texts():
    """Generate sample texts corresponding to vectors."""
    return [f"Sample text {i} for vector embeddings" for i in range(100)]


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for vectors."""
    return [{'id': f'doc_{i}', 'source': 'test', 'category': f'cat_{i % 5}'} for i in range(100)]


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as a slow test")
    config.addinivalue_line("markers", "unit: mark test as a unit test")


# Mock problematic dependencies at import time
@pytest.fixture(scope="session", autouse=True)
def mock_problematic_imports():
    """Mock problematic imports that cause torchvision/transformers issues."""
    
    # Mock torchvision to avoid operator issues
    torchvision_mock = MagicMock()
    torchvision_mock.ops = MagicMock()
    torchvision_mock.ops.nms = MagicMock()
    
    # Mock transformers components that cause issues
    transformers_mock = MagicMock()
    transformers_mock.WhisperProcessor = MagicMock()
    transformers_mock.WhisperTokenizer = MagicMock()
    transformers_mock.WhisperFeatureExtractor = MagicMock()
    
    with patch.dict('sys.modules', {
        'torchvision': torchvision_mock,
        'torchvision.ops': torchvision_mock.ops,
        'transformers': transformers_mock,
    }):
        yield
