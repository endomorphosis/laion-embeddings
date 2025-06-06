#!/usr/bin/env python3
"""Debug tool registration issue by testing imports step by step."""

def test_step_by_step():
    import sys
    import os
    
    # Set up path
    project_root = '/home/barberb/laion-embeddings-1'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    results = []
    
    # Step 1: Test basic tool registry import
    try:
        from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
        results.append("✅ Tool registry import successful")
    except Exception as e:
        results.append(f"❌ Tool registry import failed: {e}")
        return results
    
    # Step 2: Test individual new tool imports
    tool_modules = [
        ('sparse_embedding_tools', 'SparseEmbeddingGenerationTool'),
        ('ipfs_cluster_tools', 'IPFSClusterManagementTool'), 
        ('session_management_tools', 'SessionCreationTool')
    ]
    
    for module_name, class_name in tool_modules:
        try:
            module = __import__(f'src.mcp_server.tools.{module_name}', fromlist=[class_name])
            tool_class = getattr(module, class_name)
            tool_instance = tool_class(None)
            results.append(f"✅ {module_name}.{class_name} -> {tool_instance.name}")
        except Exception as e:
            results.append(f"❌ {module_name}.{class_name} failed: {e}")
    
    # Step 3: Test full tool initialization
    try:
        tools = initialize_laion_tools()
        results.append(f"✅ Full initialization: {len(tools)} tools")
        
        # Check for our specific tools
        missing_tools = [
            'generate_sparse_embedding', 'index_sparse_embeddings', 'sparse_search',
            'ipfs_cluster_management', 'storacha_integration', 'ipfs_pinning_management',
            'create_session', 'monitor_sessions', 'manage_session_cleanup'
        ]
        
        tool_names = {t.name for t in tools}
        found = sum(1 for name in missing_tools if name in tool_names)
        results.append(f"✅ Found {found}/{len(missing_tools)} new tools")
        
        if found < len(missing_tools):
            results.append("❌ Some tools missing from registry")
            # Show which ones are missing
            for name in missing_tools:
                if name not in tool_names:
                    results.append(f"  Missing: {name}")
        
    except Exception as e:
        results.append(f"❌ Full initialization failed: {e}")
        import traceback
        results.append(f"Traceback: {traceback.format_exc()}")
    
    return results

# Write results to file instead of printing
if __name__ == "__main__":
    results = test_step_by_step()
    
    with open('/home/barberb/laion-embeddings-1/debug_results.txt', 'w') as f:
        f.write("MCP Tool Registration Debug Results\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(result + "\n")
        f.write("\nTest completed.\n")
        
    print("Debug results written to debug_results.txt")
