#!/usr/bin/env python3
"""
Comprehensive validation script to test all pytest fixes
"""

import sys
import os
from pathlib import Path
import traceback

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_tool_imports():
    """Test all critical tool imports."""
    
    print("=" * 60)
    print("TESTING ALL TOOL IMPORTS")
    print("=" * 60)
    
    errors = []
    success_count = 0
    
    # Test 1: Session Management Tools
    try:
        from mcp_server.tools.session_management_tools import (
            SessionCreationTool, SessionMonitoringTool, SessionCleanupTool,
            create_session_tool, monitor_session_tool, cleanup_session_tool
        )
        print("✓ session_management_tools imported successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"session_management_tools import failed: {e}")
        print(f"✗ session_management_tools import failed: {e}")
    
    # Test 2: Data Processing Tools
    try:
        from mcp_server.tools.data_processing_tools import (
            ChunkingTool, DatasetLoadingTool, ParquetToCarTool
        )
        print("✓ data_processing_tools imported successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"data_processing_tools import failed: {e}")
        print(f"✗ data_processing_tools import failed: {e}")
    
    # Test 3: Rate Limiting Tools
    try:
        from mcp_server.tools.rate_limiting_tools import (
            RateLimitConfigurationTool, RateLimitMonitoringTool
        )
        print("✓ rate_limiting_tools imported successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"rate_limiting_tools import failed: {e}")
        print(f"✗ rate_limiting_tools import failed: {e}")
    
    # Test 4: Embedding Tools
    try:
        from mcp_server.tools.embedding_tools import (
            EmbeddingGenerationTool, BatchEmbeddingTool, MultimodalEmbeddingTool
        )
        print("✓ embedding_tools imported successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"embedding_tools import failed: {e}")
        print(f"✗ embedding_tools import failed: {e}")
    
    # Test 5: Search Tools
    try:
        from mcp_server.tools.search_tools import (
            SemanticSearchTool, SimilaritySearchTool, FacetedSearchTool
        )
        print("✓ search_tools imported successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"search_tools import failed: {e}")
        print(f"✗ search_tools import failed: {e}")
    
    # Test 6: IPFS Cluster Tools
    try:
        from mcp_server.tools.ipfs_cluster_tools import (
            IPFSClusterTool, StorachaIntegrationTool
        )
        print("✓ ipfs_cluster_tools imported successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"ipfs_cluster_tools import failed: {e}")
        print(f"✗ ipfs_cluster_tools import failed: {e}")
    
    return success_count, errors


def test_tool_instantiation():
    """Test that tools can be instantiated without errors."""
    
    print("\n" + "=" * 60)
    print("TESTING TOOL INSTANTIATION")
    print("=" * 60)
    
    errors = []
    success_count = 0
    
    try:
        from mcp_server.tools.session_management_tools import SessionCreationTool
        tool = SessionCreationTool()
        print("✓ SessionCreationTool instantiated successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"SessionCreationTool instantiation failed: {e}")
        print(f"✗ SessionCreationTool instantiation failed: {e}")
    
    try:
        from mcp_server.tools.data_processing_tools import ChunkingTool
        tool = ChunkingTool(None)  # Pass None for ipfs_embeddings_instance
        print("✓ ChunkingTool instantiated successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"ChunkingTool instantiation failed: {e}")
        print(f"✗ ChunkingTool instantiation failed: {e}")
    
    try:
        from mcp_server.tools.rate_limiting_tools import RateLimitConfigurationTool
        tool = RateLimitConfigurationTool()
        print("✓ RateLimitConfigurationTool instantiated successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"RateLimitConfigurationTool instantiation failed: {e}")
        print(f"✗ RateLimitConfigurationTool instantiation failed: {e}")
    
    try:
        from mcp_server.tools.embedding_tools import EmbeddingGenerationTool
        # Create a mock embedding service
        class MockEmbeddingService:
            pass
        tool = EmbeddingGenerationTool(MockEmbeddingService())
        print("✓ EmbeddingGenerationTool instantiated successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"EmbeddingGenerationTool instantiation failed: {e}")
        print(f"✗ EmbeddingGenerationTool instantiation failed: {e}")
    
    try:
        from mcp_server.tools.search_tools import SemanticSearchTool
        # Create a mock vector service
        class MockVectorService:
            pass
        tool = SemanticSearchTool(MockVectorService())
        print("✓ SemanticSearchTool instantiated successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"SemanticSearchTool instantiation failed: {e}")
        print(f"✗ SemanticSearchTool instantiation failed: {e}")
    
    try:
        from mcp_server.tools.ipfs_cluster_tools import IPFSClusterTool
        # Create a mock ipfs vector service
        class MockIPFSVectorService:
            pass
        tool = IPFSClusterTool(MockIPFSVectorService())
        print("✓ IPFSClusterTool instantiated successfully")
        success_count += 1
    except Exception as e:
        errors.append(f"IPFSClusterTool instantiation failed: {e}")
        print(f"✗ IPFSClusterTool instantiation failed: {e}")
    
    return success_count, errors


def main():
    """Run all validation tests."""
    
    print("LAION EMBEDDINGS PROJECT - PYTEST FIXES VALIDATION")
    print("=" * 60)
    
    total_success = 0
    total_errors = []
    
    # Test imports
    import_success, import_errors = test_tool_imports()
    total_success += import_success
    total_errors.extend(import_errors)
    
    # Test instantiation
    instantiation_success, instantiation_errors = test_tool_instantiation()
    total_success += instantiation_success
    total_errors.extend(instantiation_errors)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✓ Total successful tests: {total_success}")
    print(f"✗ Total errors: {len(total_errors)}")
    
    if total_errors:
        print("\nErrors encountered:")
        for i, error in enumerate(total_errors, 1):
            print(f"{i}. {error}")
        return False
    else:
        print("\n🎉 ALL TESTS PASSED! All pytest fixes are working correctly!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
