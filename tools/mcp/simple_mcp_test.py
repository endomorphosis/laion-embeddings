#!/usr/bin/env python3
"""
Simple MCP Server Test - Quick validation
"""

import json
import subprocess
import sys
import time
import threading
import os

def test_mcp_server():
    """Test the MCP server with a simple request"""
    print("🚀 Testing MCP Server...")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/barberb/laion-embeddings-1'
    
    # Start server
    print("Starting server...")
    process = subprocess.Popen(
        [sys.executable, 'mcp_server_minimal.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd='/home/barberb/laion-embeddings-1'
    )
    
    # Test requests
    test_requests = [
        # List tools
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        },
        # Initialize
        {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
    ]
    
    try:
        for i, request in enumerate(test_requests):
            print(f"\n📤 Sending request {i+1}: {request['method']}")
            
            # Send request
            request_line = json.dumps(request) + "\n"
            process.stdin.write(request_line)
            process.stdin.flush()
            
            # Wait for response with timeout
            import select
            if select.select([process.stdout], [], [], 5.0)[0]:
                response_line = process.stdout.readline().strip()
                if response_line:
                    try:
                        response = json.loads(response_line)
                        print(f"📥 Response: {json.dumps(response, indent=2)}")
                    except json.JSONDecodeError:
                        print(f"📥 Raw response: {response_line}")
                else:
                    print("📥 No response received")
            else:
                print("⏰ Timeout waiting for response")
                
        # Check for errors
        if process.stderr:
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"\n🔍 Server stderr: {stderr_output}")
    
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
            
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_mcp_server()
