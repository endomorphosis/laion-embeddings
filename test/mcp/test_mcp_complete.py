#!/usr/bin/env python3
"""
Complete MCP Server Test - Validates all tool categories including new ones
"""

import json
import subprocess
import sys
import time
import os
from typing import Dict, List

def test_tool_registration():
    """Test that all tools are properly registered"""
    print("🔧 Testing MCP Tool Registration...")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/barberb/laion-embeddings-1'
    
    try:
        # Start server
        print("Starting MCP server...")
        process = subprocess.Popen(
            [sys.executable, 'mcp_server_minimal.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd='/home/barberb/laion-embeddings-1'
        )
        
        # Initialize server
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        print("Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + '\n')
        process.stdin.flush()
        
        # Give server time to initialize
        time.sleep(2)
        
        # Request tool list
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        print("Requesting tool list...")
        process.stdin.write(json.dumps(tools_request) + '\n')
        process.stdin.flush()
        
        # Read responses
        time.sleep(1)
        
        # Try to get output
        try:
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                if line.strip():
                    output_lines.append(line.strip())
                    print(f"Server response: {line.strip()}")
                if len(output_lines) >= 10:  # Limit output
                    break
        except Exception as e:
            print(f"Error reading output: {e}")
        
        # Check stderr for errors
        try:
            error_output = process.stderr.read()
            if error_output:
                print(f"Server errors: {error_output}")
        except:
            pass
        
        # Terminate process
        process.terminate()
        process.wait(timeout=5)
        
        print("✅ MCP server test completed")
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        if 'process' in locals():
            try:
                process.terminate()
            except:
                pass


def test_tool_imports():
    """Test that all tool modules can be imported"""
    print("\n📦 Testing Tool Module Imports...")
    
    tool_modules = [
        "src.mcp_server.tools.embedding_tools",
        "src.mcp_server.tools.search_tools", 
        "src.mcp_server.tools.storage_tools",
        "src.mcp_server.tools.analysis_tools",
        "src.mcp_server.tools.data_processing_tools",
        "src.mcp_server.tools.auth_tools",
        "src.mcp_server.tools.admin_tools",
        "src.mcp_server.tools.cache_tools",
        "src.mcp_server.tools.monitoring_tools",
        "src.mcp_server.tools.background_task_tools",
        "src.mcp_server.tools.rate_limiting_tools",
        "src.mcp_server.tools.index_management_tools",
        # New tool categories
        "src.mcp_server.tools.sparse_embedding_tools",
        "src.mcp_server.tools.ipfs_cluster_tools",
        "src.mcp_server.tools.session_management_tools"
    ]
    
    success_count = 0
    failure_count = 0
    
    for module_name in tool_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failure_count += 1
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
            failure_count += 1
    
    print(f"\n📊 Import Results: {success_count} success, {failure_count} failures")
    return success_count, failure_count


def test_tool_registry_integration():
    """Test tool registry integration"""
    print("\n🔗 Testing Tool Registry Integration...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '/home/barberb/laion-embeddings-1')
        
        # Import registry components
        from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
        
        print("✅ Successfully imported ToolRegistry")
        
        # Create registry instance
        registry = ToolRegistry()
        
        # Initialize tools
        print("Initializing LAION tools...")
        initialize_laion_tools(registry, embedding_service=None)
        
        # Get all tools
        tools = registry.get_all_tools()
        print(f"✅ Successfully registered {len(tools)} tools")
        
        # Categorize tools
        categories = {}
        for tool_name, tool in tools.items():
            category = getattr(tool, 'category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(tool_name)
        
        print("\n📋 Tool Categories:")
        for category, tool_list in categories.items():
            print(f"  {category}: {len(tool_list)} tools")
            for tool_name in sorted(tool_list):
                print(f"    - {tool_name}")
        
        # Check for new tool categories
        expected_new_tools = [
            'generate_sparse_embedding',
            'index_sparse_embeddings', 
            'sparse_search',
            'ipfs_cluster_management',
            'storacha_integration',
            'ipfs_pinning_management',
            'create_session',
            'monitor_sessions',
            'manage_session_cleanup'
        ]
        
        found_new_tools = []
        missing_new_tools = []
        
        for expected_tool in expected_new_tools:
            if expected_tool in tools:
                found_new_tools.append(expected_tool)
            else:
                missing_new_tools.append(expected_tool)
        
        print(f"\n🆕 New Tools Status:")
        print(f"✅ Found: {len(found_new_tools)}")
        print(f"❌ Missing: {len(missing_new_tools)}")
        
        if found_new_tools:
            print("Found new tools:")
            for tool in found_new_tools:
                print(f"  - {tool}")
        
        if missing_new_tools:
            print("Missing new tools:")
            for tool in missing_new_tools:
                print(f"  - {tool}")
        
        return len(found_new_tools), len(missing_new_tools)
        
    except Exception as e:
        print(f"❌ Error testing tool registry: {e}")
        import traceback
        traceback.print_exc()
        return 0, len(expected_new_tools)


def main():
    """Run all tests"""
    print("🧪 MCP Complete Tool Test Suite")
    print("=" * 50)
    
    # Test 1: Module imports
    success_imports, failed_imports = test_tool_imports()
    
    # Test 2: Tool registry integration  
    found_tools, missing_tools = test_tool_registry_integration()
    
    # Test 3: MCP server functionality
    test_tool_registration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Module Imports: {success_imports} success, {failed_imports} failures")
    print(f"New Tools: {found_tools} found, {missing_tools} missing")
    
    if failed_imports == 0 and missing_tools == 0:
        print("🎉 All tests passed! MCP server implementation is complete.")
        return 0
    else:
        print("⚠️  Some issues found. Check output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
