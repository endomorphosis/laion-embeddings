#!/usr/bin/env python3
"""Direct test of new MCP tools."""

import sys
import os
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

def test_tool_imports():
    """Test if all new tool modules can be imported."""
    print("Testing tool imports...")
    
    # Test sparse embedding tools
    try:
        from src.mcp_server.tools.sparse_embedding_tools import (
            SparseEmbeddingGenerationTool,
            SparseIndexingTool, 
            SparseSearchTool
        )
        print("✅ sparse_embedding_tools imported successfully")
        
        # Test tool instantiation
        tool1 = SparseEmbeddingGenerationTool()
        tool2 = SparseIndexingTool()
        tool3 = SparseSearchTool()
        print(f"  - Tool names: {tool1.name}, {tool2.name}, {tool3.name}")
        
    except Exception as e:
        print(f"❌ sparse_embedding_tools import failed: {e}")
    
    # Test IPFS cluster tools
    try:
        from src.mcp_server.tools.ipfs_cluster_tools import (
            IPFSClusterManagementTool,
            StorachaIntegrationTool,
            IPFSPinningTool
        )
        print("✅ ipfs_cluster_tools imported successfully")
        
        # Test tool instantiation
        tool1 = IPFSClusterManagementTool()
        tool2 = StorachaIntegrationTool()
        tool3 = IPFSPinningTool()
        print(f"  - Tool names: {tool1.name}, {tool2.name}, {tool3.name}")
        
    except Exception as e:
        print(f"❌ ipfs_cluster_tools import failed: {e}")
    
    # Test session management tools
    try:
        from src.mcp_server.tools.session_management_tools import (
            SessionCreationTool,
            SessionMonitoringTool,
            SessionCleanupTool
        )
        print("✅ session_management_tools imported successfully")
        
        # Test tool instantiation
        tool1 = SessionCreationTool()
        tool2 = SessionMonitoringTool()
        tool3 = SessionCleanupTool()
        print(f"  - Tool names: {tool1.name}, {tool2.name}, {tool3.name}")
        
    except Exception as e:
        print(f"❌ session_management_tools import failed: {e}")

def test_tool_registry():
    """Test if the tool registry loads all tools."""
    print("\nTesting tool registry...")
    
    try:
        from src.mcp_server.tool_registry import initialize_laion_tools
        tools = initialize_laion_tools()
        print(f"✅ Tool registry loaded {len(tools)} tools")
        
        # Check for our specific tools
        missing_tools = [
            'generate_sparse_embedding', 'index_sparse_embeddings', 'sparse_search',
            'ipfs_cluster_management', 'storacha_integration', 'ipfs_pinning_management',
            'create_session', 'monitor_sessions', 'manage_session_cleanup'
        ]
        
        loaded_tool_names = [tool.name for tool in tools]
        
        print("\nChecking for missing tools:")
        all_found = True
        for tool_name in missing_tools:
            if tool_name in loaded_tool_names:
                print(f"✅ {tool_name} - FOUND")
            else:
                print(f"❌ {tool_name} - MISSING")
                all_found = False
        
        if all_found:
            print("\n🎉 ALL NEW TOOLS SUCCESSFULLY REGISTERED!")
        else:
            print("\n⚠️  Some tools are still missing from registry")
            
        return all_found
        
    except Exception as e:
        print(f"❌ Tool registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("LAION EMBEDDINGS MCP TOOLS TEST")
    print("=" * 60)
    
    test_tool_imports()
    success = test_tool_registry()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! MCP server should have all tools.")
    else:
        print("❌ Some tests failed. Check the output above.")
    print("=" * 60)
