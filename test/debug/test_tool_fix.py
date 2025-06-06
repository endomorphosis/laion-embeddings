#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

print("Testing tool registry fix...")

try:
    from src.mcp_server.tool_registry import initialize_laion_tools
    tools = initialize_laion_tools()
    
    print(f"Total tools loaded: {len(tools)}")
    
    # Check for our new tools
    missing_tools = [
        'generate_sparse_embedding', 'index_sparse_embeddings', 'sparse_search',
        'ipfs_cluster_management', 'storacha_integration', 'ipfs_pinning_management',
        'create_session', 'monitor_sessions', 'manage_session_cleanup'
    ]
    
    tool_names = [t.name for t in tools]
    found_count = 0
    
    print("\nChecking for new tools:")
    for name in missing_tools:
        if name in tool_names:
            print(f"✅ {name}")
            found_count += 1
        else:
            print(f"❌ {name}")
    
    print(f"\nResult: {found_count}/{len(missing_tools)} new tools found")
    
    # Write result to file
    with open('/home/barberb/laion-embeddings-1/tool_test_result.txt', 'w') as f:
        f.write(f"Tool test result: {found_count}/{len(missing_tools)} new tools found\n")
        f.write(f"Total tools: {len(tools)}\n")
        if found_count == len(missing_tools):
            f.write("SUCCESS: All new tools registered!\n")
        else:
            f.write("INCOMPLETE: Some tools missing\n")
    
    print("Results written to tool_test_result.txt")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Write error to file
    with open('/home/barberb/laion-embeddings-1/tool_test_result.txt', 'w') as f:
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())

print("Test completed.")
