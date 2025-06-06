#!/usr/bin/env python3
"""
Proper MCP Server Test - Spawns server as subprocess and tests JSON-RPC communication
"""

import json
import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path

class MCPServerTester:
    def __init__(self, server_path):
        self.server_path = server_path
        self.process = None
        
    def start_server(self):
        """Start the MCP server as a subprocess"""
        print("🚀 Starting MCP server subprocess...")
        
        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = '/home/barberb/laion-embeddings-1'
        
        self.process = subprocess.Popen(
            [sys.executable, self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered
            env=env
        )
        
        # Give server time to start
        time.sleep(2)
        
        if self.process.poll() is not None:
            # Process has already terminated
            stderr = self.process.stderr.read()
            print(f"❌ Server failed to start. Error: {stderr}")
            return False
            
        print("✅ Server started successfully")
        return True
    
    def send_request(self, request, timeout=10):
        """Send a JSON-RPC request to the server"""
        if not self.process or self.process.poll() is not None:
            raise Exception("Server not running")
            
        request_json = json.dumps(request) + '\n'
        print(f"📤 Sending: {request}")
        
        try:
            # Send request
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response with timeout
            def read_with_timeout():
                return self.process.stdout.readline()
            
            # Use threading for timeout
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = read_with_timeout()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                raise TimeoutError(f"Response timeout after {timeout}s")
                
            if exception[0]:
                raise exception[0]
                
            if not result[0]:
                raise Exception("Empty response")
                
            response_line = result[0].strip()
            if not response_line:
                raise Exception("Empty response line")
                
            response = json.loads(response_line)
            print(f"📥 Received: {response}")
            return response
            
        except Exception as e:
            print(f"❌ Request failed: {e}")
            # Check if server is still alive and get stderr
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                print(f"Server stderr: {stderr}")
            raise
    
    def test_initialize(self):
        """Test server initialization"""
        print("\n🔧 Testing initialization...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = self.send_request(request)
        
        # Validate response
        if response.get("id") != 1:
            raise Exception(f"Wrong response ID: {response.get('id')}")
            
        result = response.get("result", {})
        server_info = result.get("serverInfo", {})
        
        if server_info.get("name") != "laion-embeddings-mcp":
            raise Exception(f"Wrong server name: {server_info.get('name')}")
            
        print("✅ Initialize test passed")
        return True
    
    def test_list_tools(self):
        """Test listing tools"""
        print("\n🛠️ Testing tools list...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = self.send_request(request)
        
        # Validate response
        if response.get("id") != 2:
            raise Exception(f"Wrong response ID: {response.get('id')}")
            
        result = response.get("result", {})
        tools = result.get("tools", [])
        
        if not tools:
            raise Exception("No tools returned")
            
        tool_names = [tool["name"] for tool in tools]
        expected_tools = ["generate_embedding", "semantic_search", "cluster_analysis"]
        
        for expected_tool in expected_tools:
            if expected_tool not in tool_names:
                raise Exception(f"Missing tool: {expected_tool}")
        
        print(f"✅ Tools list test passed. Found tools: {tool_names}")
        return True
    
    def test_call_tool(self):
        """Test calling a tool"""
        print("\n⚡ Testing tool call...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "generate_embedding",
                "arguments": {
                    "text": "Hello world test",
                    "model": "thenlper/gte-small"
                }
            }
        }
        
        response = self.send_request(request)
        
        # Validate response
        if response.get("id") != 3:
            raise Exception(f"Wrong response ID: {response.get('id')}")
            
        result = response.get("result", {})
        content = result.get("content", [])
        
        if not content:
            raise Exception("No content in tool response")
            
        if content[0].get("type") != "text":
            raise Exception("Wrong content type")
            
        text_result = content[0].get("text", "")
        if "Hello world test" not in text_result:
            raise Exception("Tool result doesn't contain input text")
        
        print(f"✅ Tool call test passed. Result: {text_result[:100]}...")
        return True
    
    def stop_server(self):
        """Stop the MCP server"""
        if self.process:
            print("🛑 Stopping server...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️ Server didn't stop gracefully, killing...")
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None
            print("✅ Server stopped")
    
    def run_full_test(self):
        """Run complete test suite"""
        try:
            print("=== MCP Server Subprocess Test ===")
            
            # Start server
            if not self.start_server():
                return False
                
            # Run tests
            self.test_initialize()
            self.test_list_tools()
            self.test_call_tool()
            
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ MCP Server is working correctly with subprocess communication")
            print("✅ VS Code MCP integration should work!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            
            # Get server stderr for debugging
            if self.process and self.process.stderr:
                try:
                    stderr = self.process.stderr.read()
                    if stderr:
                        print(f"Server stderr: {stderr}")
                except:
                    pass
            
            return False
            
        finally:
            self.stop_server()

def main():
    server_path = "/home/barberb/laion-embeddings-1/mcp_server_minimal.py"
    
    if not Path(server_path).exists():
        print(f"❌ Server file not found: {server_path}")
        return False
    
    tester = MCPServerTester(server_path)
    success = tester.run_full_test()
    
    if success:
        print("\n📋 VS Code Configuration Status:")
        config_path = "/home/barberb/laion-embeddings-1/.vscode/mcp.json"
        if Path(config_path).exists():
            print(f"✅ MCP config file exists: {config_path}")
            with open(config_path) as f:
                config = json.load(f)
                if "laion-embeddings" in config.get("mcpServers", {}):
                    print("✅ LAION embeddings server configured")
                else:
                    print("⚠️ LAION embeddings server not found in config")
        else:
            print(f"❌ MCP config file missing: {config_path}")
        
        print("\n🚀 Ready for Claude integration!")
        return True
    else:
        print("\n❌ MCP Server test failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
