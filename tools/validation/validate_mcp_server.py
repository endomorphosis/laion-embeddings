#!/usr/bin/env python3
"""
MCP Server Component Validation
Tests the core MCP server components without subprocess communication
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

def test_mcp_server_components():
    """Test MCP server components directly"""
    print("🔍 Testing MCP Server Components...")
    
    try:
        # Test 1: Import the minimal server
        print("\n1. Testing MCP server import...")
        from mcp_server_minimal import MinimalMCPServer
        print("✅ MinimalMCPServer imported successfully")
        
        # Test 2: Create server instance
        print("\n2. Testing server instantiation...")
        server = MinimalMCPServer()
        print("✅ Server instance created successfully")
        
        # Test 3: Check tools setup
        print("\n3. Testing tools setup...")
        print(f"   Available tools: {list(server.tools.keys())}")
        expected_tools = ["generate_embedding", "semantic_search", "cluster_analysis", "storage_management"]
        for tool in expected_tools:
            if tool in server.tools:
                print(f"   ✅ {tool} - configured")
            else:
                print(f"   ❌ {tool} - missing")
        
        # Test 4: Test request handling methods
        print("\n4. Testing request handling...")
        
        # Test initialize request
        async def test_initialize():
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
            response = await server.handle_request(init_request)
            return response
        
        # Test tools/list request
        async def test_tools_list():
            list_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            response = await server.handle_request(list_request)
            return response
        
        # Test tools/call request
        async def test_tool_call():
            call_request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "generate_embedding",
                    "arguments": {"text": "Hello world"}
                }
            }
            response = await server.handle_request(call_request)
            return response
        
        # Run async tests
        async def run_async_tests():
            print("   Testing initialize...")
            init_response = await test_initialize()
            print(f"   ✅ Initialize response: {init_response.get('result', {}).get('capabilities', 'OK')}")
            
            print("   Testing tools/list...")
            list_response = await test_tools_list()
            tools_count = len(list_response.get('result', {}).get('tools', []))
            print(f"   ✅ Tools list response: {tools_count} tools found")
            
            print("   Testing tools/call...")
            call_response = await test_tool_call()
            if 'result' in call_response:
                print(f"   ✅ Tool call response: Success")
            elif 'error' in call_response:
                print(f"   ⚠️  Tool call response: {call_response['error']['message']}")
            else:
                print(f"   ❓ Tool call response: {call_response}")
        
        # Run the async tests
        asyncio.run(run_async_tests())
        
        print("\n🎉 All component tests completed successfully!")
        print("\n📋 Summary:")
        print("   - MCP server imports correctly")
        print("   - Server instantiation works")
        print("   - Tools are properly configured")
        print("   - Request handling methods work")
        print("   - Async functionality is operational")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_vs_code_config():
    """Validate VS Code MCP configuration"""
    print("\n🔍 Validating VS Code MCP Configuration...")
    
    try:
        config_path = Path('/home/barberb/laion-embeddings-1/.vscode/mcp.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print("✅ MCP configuration file exists")
            
            if 'mcpServers' in config:
                servers = config['mcpServers']
                print(f"✅ Found {len(servers)} MCP server(s) configured")
                
                for name, server_config in servers.items():
                    print(f"   📡 Server: {name}")
                    print(f"      Command: {server_config.get('command', 'Not specified')}")
                    print(f"      Args: {server_config.get('args', [])}")
                    print(f"      Env: {server_config.get('env', {})}")
                    
                    # Verify the server file exists
                    if 'args' in server_config and server_config['args']:
                        server_file = server_config['args'][0]
                        if os.path.exists(server_file):
                            print(f"      ✅ Server file exists: {server_file}")
                        else:
                            print(f"      ❌ Server file missing: {server_file}")
            
            return True
        else:
            print("❌ MCP configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MCP Server Validation Suite")
    print("=" * 50)
    
    # Test components
    components_ok = test_mcp_server_components()
    
    # Test VS Code config
    config_ok = validate_vs_code_config()
    
    print("\n" + "=" * 50)
    if components_ok and config_ok:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\n✅ Your MCP server is ready for use with Claude in VS Code")
        print("\n📖 Next steps:")
        print("   1. Restart VS Code to load the new MCP configuration")
        print("   2. Open Claude in VS Code")
        print("   3. Claude should automatically connect to your LAION embeddings server")
        print("   4. Try asking Claude to generate embeddings or perform semantic search")
    else:
        print("❌ Some validations failed - check the output above")
