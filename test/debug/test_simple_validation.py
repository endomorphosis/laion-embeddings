#!/usr/bin/env python3
"""
Simple validation test for MCP server components.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports of MCP components."""
    print("=" * 50)
    print("BASIC IMPORT VALIDATION TEST")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Import MCPConfig
    tests_total += 1
    try:
        from src.mcp_server.config import MCPConfig
        print("✓ MCPConfig import successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ MCPConfig import failed: {e}")
    
    # Test 2: Import ServiceFactory
    tests_total += 1
    try:
        from src.mcp_server.service_factory import ServiceFactory
        print("✓ ServiceFactory import successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ServiceFactory import failed: {e}")
    
    # Test 3: Import core services
    tests_total += 1
    try:
        from services.vector_service import VectorService
        from services.embedding_service import EmbeddingService
        from services.clustering_service import VectorClusterer
        print("✓ Core services import successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Core services import failed: {e}")
    
    # Test 4: Import MCP tools
    tests_total += 1
    try:
        from src.mcp_server.tools.embedding_tools import EmbeddingGenerationTool
        from src.mcp_server.tools.search_tools import SemanticSearchTool
        from src.mcp_server.tools.storage_tools import StorageManagementTool
        print("✓ MCP tools import successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ MCP tools import failed: {e}")
    
    # Test 5: Basic config creation
    tests_total += 1
    try:
        config = MCPConfig()
        print(f"✓ MCPConfig creation successful: {config.server_name}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ MCPConfig creation failed: {e}")
    
    print(f"\nResults: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("🎉 All basic validation tests passed!")
        return True
    else:
        print("❌ Some validation tests failed!")
        return False

def main():
    """Main function."""
    try:
        success = test_basic_imports()
        return 0 if success else 1
    except Exception as e:
        print(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
