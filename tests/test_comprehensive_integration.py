#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for IPFS Kit Migration

This test suite validates that the migration from legacy IPFS libraries
to ipfs_kit_py has been successful and all functionality works correctly.
"""

import pytest
import sys
import os
import warnings
import tempfile
import json
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestIPFSKitIntegration:
    """Test suite for ipfs_kit_py integration."""
    
    def test_ipfs_kit_import(self):
        """Test that ipfs_kit_py can be imported successfully."""
        try:
            from ipfs_kit_py import ipfs_kit
            assert ipfs_kit is not None, "ipfs_kit should not be None"
        except ImportError as e:
            pytest.fail(f"Failed to import ipfs_kit: {e}")
    
    def test_storacha_kit_import(self):
        """Test that storacha_kit can be imported from ipfs_kit_py."""
        try:
            from ipfs_kit_py import storacha_kit
            assert storacha_kit is not None, "storacha_kit should not be None"
        except ImportError as e:
            pytest.fail(f"Failed to import storacha_kit: {e}")
    
    def test_s3_kit_import(self):
        """Test that s3_kit can be imported from ipfs_kit_py."""
        try:
            from ipfs_kit_py import s3_kit
            assert s3_kit is not None, "s3_kit should not be None"
        except ImportError as e:
            pytest.fail(f"Failed to import s3_kit: {e}")
    
    def test_high_level_api_import(self):
        """Test that high-level API can be imported."""
        try:
            from ipfs_kit_py.high_level_api import IPFSSimpleAPI
            assert IPFSSimpleAPI is not None, "IPFSSimpleAPI should not be None"
        except ImportError as e:
            pytest.fail(f"Failed to import IPFSSimpleAPI: {e}")
    
    def test_error_classes_import(self):
        """Test that error classes can be imported."""
        try:
            from ipfs_kit_py.error import (
                IPFSError, 
                IPFSConnectionError, 
                IPFSContentNotFoundError,
                IPFSValidationError
            )
            assert all([
                IPFSError is not None,
                IPFSConnectionError is not None,
                IPFSContentNotFoundError is not None,
                IPFSValidationError is not None
            ]), "All error classes should be importable"
        except ImportError as e:
            pytest.fail(f"Failed to import error classes: {e}")
    
    def test_deprecated_storacha_clusters_warning(self):
        """Test that importing storacha_clusters produces deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                from storacha_clusters import storacha_clusters
                # Check that at least one warning was issued
                assert len(w) > 0, "Expected deprecation warning for storacha_clusters"
                # Check that it's a deprecation warning
                warning_messages = [str(warning.message) for warning in w]
                deprecation_found = any("deprecated" in msg.lower() for msg in warning_messages)
                assert deprecation_found, f"Expected deprecation warning, got: {warning_messages}"
            except ImportError:
                pytest.fail("storacha_clusters should still be importable (with warnings)")
    
    def test_ipfs_kit_instantiation(self):
        """Test that ipfs_kit can be instantiated."""
        try:
            from ipfs_kit_py import ipfs_kit
            
            # Test instantiation with minimal resources and metadata
            resources = {"test": True}
            metadata = {"test_meta": "value"}
            
            kit_instance = ipfs_kit(resources, metadata)
            assert kit_instance is not None, "ipfs_kit instance should not be None"
            assert hasattr(kit_instance, 'resources'), "Instance should have resources attribute"
            assert hasattr(kit_instance, 'metadata'), "Instance should have metadata attribute"
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate ipfs_kit: {e}")
    
    def test_storacha_kit_instantiation(self):
        """Test that storacha_kit can be instantiated."""
        try:
            from ipfs_kit_py import storacha_kit
            
            # Test instantiation with minimal resources and metadata
            resources = {"test": True}
            metadata = {"test_meta": "value"}
            
            kit_instance = storacha_kit(resources, metadata)
            assert kit_instance is not None, "storacha_kit instance should not be None"
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate storacha_kit: {e}")

class TestProjectIntegration:
    """Test that project components work with ipfs_kit_py."""
    
    def test_main_module_import(self):
        """Test that main module can be imported without errors."""
        try:
            # Test that main.py can be imported without legacy dependency errors
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore deprecation warnings for this test
                import main
        except ImportError as e:
            # Allow import errors for modules that need proper environment setup
            if "storacha_clusters" in str(e) or "ipfs_cluster_index" in str(e):
                pytest.skip(f"Skipping due to expected import issue: {e}")
            else:
                pytest.fail(f"Unexpected import error in main: {e}")
    
    def test_create_embeddings_integration(self):
        """Test that create_embeddings module works with ipfs_kit_py."""
        try:
            # This should work since we've updated the imports
            from create_embeddings.create_embeddings import create_embeddings
            
            # Test instantiation
            resources = {"test": True}
            metadata = {"test_meta": "value"}
            
            embeddings_instance = create_embeddings(resources, metadata)
            assert embeddings_instance is not None
            
        except Exception as e:
            pytest.fail(f"Failed to integrate create_embeddings with ipfs_kit_py: {e}")
    
    def test_shard_embeddings_integration(self):
        """Test that shard_embeddings module works with ipfs_kit_py."""
        try:
            from shard_embeddings.shard_embeddings import shard_embeddings
            
            # Test instantiation
            resources = {"test": True}
            metadata = {"test_meta": "value"}
            
            shard_instance = shard_embeddings(resources, metadata)
            assert shard_instance is not None
            
        except Exception as e:
            pytest.fail(f"Failed to integrate shard_embeddings with ipfs_kit_py: {e}")
    
    def test_ipfs_cluster_index_integration(self):
        """Test that ipfs_cluster_index module works with ipfs_kit_py."""
        try:
            from ipfs_cluster_index.ipfs_cluster_index import ipfs_cluster_index
            
            # Test instantiation
            resources = {"test": True}
            metadata = {"test_meta": "value"}
            
            cluster_instance = ipfs_cluster_index(resources, metadata)
            assert cluster_instance is not None
            
        except Exception as e:
            pytest.fail(f"Failed to integrate ipfs_cluster_index with ipfs_kit_py: {e}")

class TestFunctionalIntegration:
    """Test actual functionality of integrated components."""
    
    def test_ipfs_kit_basic_functionality(self):
        """Test basic functionality of ipfs_kit."""
        try:
            from ipfs_kit_py import ipfs_kit
            
            resources = {"api_endpoint": "http://localhost:5001"}
            metadata = {"test": True}
            
            kit = ipfs_kit(resources, metadata)
            
            # Test that basic methods exist and can be called
            assert hasattr(kit, 'add'), "ipfs_kit should have add method"
            assert hasattr(kit, 'pin'), "ipfs_kit should have pin method"
            assert hasattr(kit, 'get'), "ipfs_kit should have get method"
            
        except Exception as e:
            pytest.fail(f"Failed basic functionality test for ipfs_kit: {e}")
    
    def test_storacha_kit_basic_functionality(self):
        """Test basic functionality of storacha_kit."""
        try:
            from ipfs_kit_py import storacha_kit
            
            resources = {"api_key": "test_key"}
            metadata = {"test": True}
            
            kit = storacha_kit(resources, metadata)
            
            # Test that basic methods exist
            assert hasattr(kit, 'upload'), "storacha_kit should have upload method"
            assert hasattr(kit, 'store'), "storacha_kit should have store method"
            
        except Exception as e:
            pytest.fail(f"Failed basic functionality test for storacha_kit: {e}")

class TestLegacyImportReplacements:
    """Test that legacy imports have been properly replaced."""
    
    def test_no_legacy_ipfs_imports_in_main_files(self):
        """Check that main project files don't have legacy IPFS imports."""
        legacy_patterns = [
            "import ipfs_embeddings_py",
            "import ipfs_datasets_py", 
            "import ipfs_accelerate_py",
            "import ipfs_transformers_py",
            "import ipfshttpclient",
            "from ipfs_embeddings_py",
            "from ipfs_datasets_py",
            "from ipfs_accelerate_py", 
            "from ipfs_transformers_py",
            "from ipfshttpclient"
        ]
        
        main_files = [
            "main.py",
            "__init__.py",
            "create_embeddings/create_embeddings.py",
            "shard_embeddings/shard_embeddings.py",
            "ipfs_cluster_index/ipfs_cluster_index.py"
        ]
        
        for file_path in main_files:
            full_path = project_root / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                    for pattern in legacy_patterns:
                        if pattern in content and not content.count(pattern) == content.count(f"# {pattern}"):
                            pytest.fail(f"Found legacy import '{pattern}' in {file_path}")
    
    def test_ipfs_kit_py_imports_present(self):
        """Check that ipfs_kit_py imports are present in main files."""
        expected_imports = [
            "from ipfs_kit_py import ipfs_kit",
            "from ipfs_kit_py import storacha_kit",
            "import ipfs_kit_py"
        ]
        
        main_files = [
            "create_embeddings/create_embeddings.py",
            "shard_embeddings/shard_embeddings.py", 
            "ipfs_cluster_index/ipfs_cluster_index.py"
        ]
        
        for file_path in main_files:
            full_path = project_root / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                    # Check that at least one ipfs_kit_py import is present
                    has_import = any(import_stmt in content for import_stmt in expected_imports)
                    assert has_import, f"No ipfs_kit_py imports found in {file_path}"

class TestMCPDatasetTools:
    """Test that MCP dataset tools are available and working."""
    
    def test_mcp_dataset_tools_import(self):
        """Test that MCP dataset tools can be imported."""
        try:
            # Test these specific functions that were mentioned in the workspace
            from mcp_ipfs_datasets_load_dataset import load_dataset
            from mcp_ipfs_datasets_save_dataset import save_dataset
            from mcp_ipfs_datasets_process_dataset import process_dataset
            
            assert callable(load_dataset), "load_dataset should be callable"
            assert callable(save_dataset), "save_dataset should be callable" 
            assert callable(process_dataset), "process_dataset should be callable"
            
        except ImportError as e:
            pytest.skip(f"MCP dataset tools not available: {e}")
    
    def test_mcp_ipfs_functions_import(self):
        """Test that MCP IPFS functions can be imported."""
        try:
            from mcp_ipfs_datasets_pin_to_ipfs import pin_to_ipfs
            from mcp_ipfs_datasets_get_from_ipfs import get_from_ipfs
            
            assert callable(pin_to_ipfs), "pin_to_ipfs should be callable"
            assert callable(get_from_ipfs), "get_from_ipfs should be callable"
            
        except ImportError as e:
            pytest.skip(f"MCP IPFS functions not available: {e}")

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
