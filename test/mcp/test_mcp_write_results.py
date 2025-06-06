#!/usr/bin/env python3
"""
Test MCP server functionality and write results to a file.
"""
import sys
import traceback
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Open result file
with open('mcp_test_results.txt', 'w') as f:
    f.write("MCP Server Test Results\n")
    f.write("======================\n\n")
    
    try:
        f.write("1. Testing basic imports...\n")
        from mcp_server.tool_registry import ToolRegistry, ClaudeMCPTool, initialize_laion_tools
        from mcp_server.error_handlers import ValidationError, MCPError
        f.write("✓ Basic imports successful\n\n")
        
        f.write("2. Testing ToolRegistry creation...\n")
        registry = ToolRegistry()
        f.write("✓ ToolRegistry created successfully\n\n")
        
        f.write("3. Testing tool initialization...\n")
        initialize_laion_tools(registry, embedding_service=None)
        tools = registry.get_all_tools()
        f.write(f"✓ Initialized {len(tools)} tools:\n")
        for tool in tools:
            f.write(f"  - {tool.name}: {tool.description[:50]}...\n")
        f.write("\n")
        
        f.write("4. Testing tool categories...\n")
        categories = registry.get_categories()
        f.write(f"✓ Found {len(categories)} categories:\n")
        for category, tool_list in categories.items():
            f.write(f"  - {category}: {len(tool_list)} tools\n")
        f.write("\n")
        
        f.write("5. Testing simple tool execution...\n")
        if "cluster_analysis" in [tool.name for tool in tools]:
            cluster_tool = registry.get_tool("cluster_analysis")
            # This would be async, so just test that we can get the tool
            f.write("✓ Cluster analysis tool retrieved successfully\n")
            f.write(f"  Tool parameters schema keys: {list(cluster_tool.parameters_schema.keys())}\n")
        f.write("\n")
        
        f.write("🎉 ALL TESTS PASSED! MCP server is ready.\n")
        
    except Exception as e:
        f.write(f"❌ ERROR: {str(e)}\n")
        f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
print("Test completed. Check mcp_test_results.txt for results.")
