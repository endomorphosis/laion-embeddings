#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

print("Testing tool registry...")

try:
    from src.mcp_server.tool_registry import initialize_laion_tools
    tools = initialize_laion_tools()
    
    missing_tools = ['generate_sparse_embedding', 'index_sparse_embeddings', 'sparse_search', 'ipfs_cluster_management', 'storacha_integration', 'ipfs_pinning_management', 'create_session', 'monitor_sessions', 'manage_session_cleanup']
    
    loaded_tool_names = [tool.name for tool in tools]
    
    print(f"Total tools: {len(tools)}")
    print("Missing tools check:")
    
    for tool_name in missing_tools:
        if tool_name in loaded_tool_names:
            print(f"✅ {tool_name}")
        else:
            print(f"❌ {tool_name}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
