#!/usr/bin/env python3
"""Test script to verify our new MCP tools are working"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tool_imports():
    """Test importing our new tool modules"""
    print("Testing tool imports...")
    
    try:
        from src.mcp_server.tools.sparse_embedding_tools import (
            SparseEmbeddingGenerationTool,
            SparseIndexingTool, 
            SparseSearchTool
        )
        print("✅ Sparse embedding tools imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import sparse embedding tools: {e}")
        return False

def test_ipfs_tools():
    """Test importing IPFS cluster tools"""
    try:
        from src.mcp_server.tools.ipfs_cluster_tools import (
            IPFSClusterManagementTool,
            StorachaIntegrationTool,
            IPFSPinningTool
        )
        print("✅ IPFS cluster tools imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import IPFS cluster tools: {e}")
        return False

def test_session_tools():
    """Test importing session management tools"""
    try:
        from src.mcp_server.tools.session_management_tools import (
            SessionCreationTool,
            SessionMonitoringTool,
            SessionCleanupTool
        )
        print("✅ Session management tools imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import session management tools: {e}")
        return False

def test_tool_registry():
    """Test the tool registry integration"""
    try:
        from src.mcp_server.tool_registry import ToolRegistry
        registry = ToolRegistry()
        print("✅ Tool registry created successfully")
        
        # Test individual tool registrations
        from src.mcp_server.tools.sparse_embedding_tools import SparseEmbeddingGenerationTool
        tool = SparseEmbeddingGenerationTool()
        registry.register_tool("test_sparse", tool)
        
        tools = registry.get_all_tools()
        if "test_sparse" in tools:
            print("✅ Tool registration works")
            return True
        else:
            print("❌ Tool registration failed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test tool registry: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing LAION MCP Tools Implementation")
    print("=" * 50)
    
    results = []
    results.append(test_tool_imports())
    results.append(test_ipfs_tools()) 
    results.append(test_session_tools())
    results.append(test_tool_registry())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ ALL TESTS PASSED!")
        print("The new MCP tools are ready for use.")
    else:
        print("❌ Some tests failed. Check the output above.")
    print("=" * 50)
