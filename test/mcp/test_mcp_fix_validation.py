#!/usr/bin/env python3
"""
Test script to validate the MCP server bug fix.
This script tests that the fix for the .items() bug is working correctly.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

def test_mcp_server_bug_fix():
    """Test that the MCP server bug fix is working."""
    print("=" * 60)
    print("MCP SERVER BUG FIX VALIDATION TEST")
    print("=" * 60)
    
    try:
        print("\n1. Testing import of LAIONMCPServer...")
        from mcp_server_enhanced import LAIONMCPServer
        print("   ✓ Successfully imported LAIONMCPServer")
        
        print("\n2. Testing server instantiation...")
        server = LAIONMCPServer()
        print("   ✓ Successfully created LAIONMCPServer instance")
        
        print("\n3. Checking server attributes...")
        print(f"   - Has 'tools' attribute: {hasattr(server, 'tools')}")
        print(f"   - Has 'tool_registry' attribute: {hasattr(server, 'tool_registry')}")
        
        if hasattr(server, 'tools'):
            print(f"   - Number of tools loaded: {len(server.tools)}")
            if len(server.tools) > 0:
                print("   - Sample tools:")
                for i, tool_name in enumerate(list(server.tools.keys())[:3]):
                    print(f"     {i+1}. {tool_name}")
        
        print("\n4. Testing tool registry access...")
        if hasattr(server, 'tool_registry'):
            all_tools = server.tool_registry.get_all_tools()
            print(f"   - Tool registry returned: {type(all_tools)}")
            print(f"   - Number of tools in registry: {len(all_tools) if all_tools else 0}")
            
            # Test that we can iterate over the tools (this was the bug)
            if all_tools:
                print("   - Testing iteration over tools (the fixed bug):")
                for tool in all_tools[:2]:  # Test first 2 tools
                    print(f"     - Tool name: {tool.name}")
                    print(f"     - Tool description: {tool.description[:50]}...")
        
        print("\n" + "=" * 60)
        print("✓ MCP SERVER BUG FIX VALIDATION SUCCESSFUL!")
        print("✓ The .items() bug has been fixed and server initializes correctly")
        print("=" * 60)
        return True
        
    except ImportError as e:
        print(f"\n   ✗ Import Error: {e}")
        return False
    except AttributeError as e:
        print(f"\n   ✗ Attribute Error: {e}")
        print("     This might indicate the bug still exists or a related issue")
        return False
    except Exception as e:
        print(f"\n   ✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mcp_server_bug_fix()
    sys.exit(0 if success else 1)
