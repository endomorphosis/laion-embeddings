"""
Simple test to verify import functionality.
"""

import pytest


class TestBasicFunctionality:
    """Test basic functionality."""
    
    def test_simple_assertion(self):
        """Test a simple assertion."""
        assert 1 + 1 == 2
    
    def test_vector_config_import(self):
        """Test importing VectorConfig."""
        from services.vector_service import VectorConfig
        config = VectorConfig()
        assert config.dimension == 768
        assert config.metric == "L2"
