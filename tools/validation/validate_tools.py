#!/usr/bin/env python3
"""Validate MCP tool registration"""

import os
import sys

# Ensure correct path
project_root = '/home/barberb/laion-embeddings-1'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    print("=" * 60)
    print("MCP TOOL REGISTRATION VALIDATION")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path includes: {project_root}")
    
    # Test basic imports
    print("\n1. Testing basic imports...")
    try:
        from src.mcp_server import tool_registry
        print("✅ tool_registry module imported")
    except Exception as e:
        print(f"❌ tool_registry import failed: {e}")
        return False
    
    # Test tool registry function
    print("\n2. Testing tool registry initialization...")
    try:
        tools = tool_registry.initialize_laion_tools()
        print(f"✅ Loaded {len(tools)} tools total")
        
        # List first few tools
        print("\nFirst 10 tools:")
        for i, tool in enumerate(tools[:10]):
            print(f"  {i+1:2d}. {tool.name}")
            
    except Exception as e:
        print(f"❌ Tool registry initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check for our specific new tools
    print("\n3. Checking for new tools...")
    missing_tools = [
        'generate_sparse_embedding', 'index_sparse_embeddings', 'sparse_search',
        'ipfs_cluster_management', 'storacha_integration', 'ipfs_pinning_management',
        'create_session', 'monitor_sessions', 'manage_session_cleanup'
    ]
    
    loaded_names = {tool.name for tool in tools}
    found_count = 0
    
    for tool_name in missing_tools:
        if tool_name in loaded_names:
            print(f"✅ {tool_name}")
            found_count += 1
        else:
            print(f"❌ {tool_name}")
    
    print(f"\nResult: Found {found_count}/{len(missing_tools)} new tools")
    
    if found_count == len(missing_tools):
        print("\n🎉 SUCCESS: All new tools are registered!")
        return True
    else:
        print(f"\n⚠️  INCOMPLETE: {len(missing_tools) - found_count} tools still missing")
        return False

if __name__ == "__main__":
    success = main()
    print("=" * 60)
    sys.exit(0 if success else 1)
