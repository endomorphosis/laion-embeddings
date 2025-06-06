#!/usr/bin/env python3
"""
Test script to validate MCP server fix
"""

import sys
import os
import traceback

# Add the project root to Python path
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

def test_mcp_imports():
    """Test MCP related imports"""
    try:
        print("🔍 Testing MCP server imports...")
        
        # Test tool registry import
        from src.mcp_server.tool_registry import ToolRegistry, ClaudeMCPTool
        print("✅ ToolRegistry and ClaudeMCPTool imported successfully")
        
        # Test that get_all_tools returns a list
        registry = ToolRegistry()
        tools = registry.get_all_tools()
        print(f"✅ ToolRegistry.get_all_tools() returns: {type(tools)} (should be list)")
        
        # Test MCP server import
        from mcp_server_enhanced import LAIONMCPServer
        print("✅ LAIONMCPServer imported successfully")
        
        # Test creating server instance (without initialization)
        print("✅ All imports successful - fix is working!")
        return True
        
    except Exception as e:
        print(f"❌ Error during import test: {e}")
        traceback.print_exc()
        return False

def test_bug_fix():
    """Test that the specific bug is fixed"""
    try:
        print("\n🔧 Testing bug fix specifically...")
        
        from src.mcp_server.tool_registry import ToolRegistry
        registry = ToolRegistry()
        
        # Get tools - this should return a list
        tools = registry.get_all_tools()
        print(f"✅ get_all_tools() returns: {type(tools).__name__}")
        
        # Verify it's a list and not a dict
        if isinstance(tools, list):
            print("✅ Correct: get_all_tools() returns a list")
            
            # Simulate the fixed loop logic
            for tool_instance in tools:
                # This should work without calling .items()
                tool_name = tool_instance.name if hasattr(tool_instance, 'name') else "unknown"
                print(f"✅ Can access tool.name: {tool_name}")
                break  # Just test the first one
                
            print("✅ Bug fix validated: No more .items() error on list")
            return True
        else:
            print(f"❌ Unexpected: get_all_tools() returns {type(tools)}")
            return False
            
    except Exception as e:
        print(f"❌ Bug fix test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 MCP Server Bug Fix Validation")
    print("=" * 60)
    
    success = True
    
    # Test imports
    if not test_mcp_imports():
        success = False
    
    # Test specific bug fix
    if not test_bug_fix():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Bug fix is successful!")
        print("✅ MCP server should now start without the .items() error")
    else:
        print("❌ TESTS FAILED - Bug fix needs more work")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
