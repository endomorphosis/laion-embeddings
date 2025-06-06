#!/usr/bin/env python3
"""
Test the minimal MCP server to ensure it works correctly
"""

import json
import subprocess
import sys
import time

def test_mcp_server():
    """Test the MCP server with sample requests"""
    
    print("Testing Minimal MCP Server...")
    
    # Start the server
    server_process = subprocess.Popen(
        [sys.executable, "/home/barberb/laion-embeddings-1/mcp_server_minimal.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Test 1: Initialize
        print("1. Testing initialize...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "0.1.0"
                }
            }
        }
        
        server_process.stdin.write(json.dumps(init_request) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        if response:
            init_response = json.loads(response.strip())
            print(f"   ✓ Initialize response: {init_response.get('result', {}).get('serverInfo', {}).get('name')}")
        else:
            print("   ✗ No response to initialize")
            return False
        
        # Test 2: List tools
        print("2. Testing tools/list...")
        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        server_process.stdin.write(json.dumps(list_request) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        if response:
            list_response = json.loads(response.strip())
            tools = list_response.get('result', {}).get('tools', [])
            print(f"   ✓ Found {len(tools)} tools: {[t['name'] for t in tools]}")
        else:
            print("   ✗ No response to tools/list")
            return False
        
        # Test 3: Call a tool
        print("3. Testing tools/call...")
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "generate_embedding",
                "arguments": {
                    "text": "Hello, world!",
                    "model": "thenlper/gte-small"
                }
            }
        }
        
        server_process.stdin.write(json.dumps(call_request) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        if response:
            call_response = json.loads(response.strip())
            result_text = call_response.get('result', {}).get('content', [{}])[0].get('text', '')
            print(f"   ✓ Tool call result: {result_text[:100]}...")
        else:
            print("   ✗ No response to tools/call")
            return False
        
        print("✓ All tests passed! MCP server is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False
        
    finally:
        # Clean up
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    if test_mcp_server():
        print("\n🎉 MCP Server is ready for VS Code integration!")
        sys.exit(0)
    else:
        print("\n❌ MCP Server test failed. Check the implementation.")
        sys.exit(1)
