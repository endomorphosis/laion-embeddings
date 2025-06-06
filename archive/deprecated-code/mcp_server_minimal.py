#!/usr/bin/env python3
"""
Minimal MCP Server for LAION Embeddings
Using basic JSON-RPC over stdio
"""

import json
import sys
import asyncio
import logging
from typing import Dict, Any, List

# Configure logging to stderr so it doesn't interfere with stdio
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class MinimalMCPServer:
    """Minimal MCP server implementation using JSON-RPC over stdio"""
    
    def __init__(self):
        self.tools = {}
        self.setup_tools()
    
    def setup_tools(self):
        """Setup available tools"""
        self.tools = {
            "generate_embedding": {
                "description": "Generate embeddings for text using LAION models",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to embed"},
                        "model": {"type": "string", "description": "Model to use", "default": "thenlper/gte-small"}
                    },
                    "required": ["text"]
                }
            },
            "semantic_search": {
                "description": "Perform semantic search using embeddings",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Number of results", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            "cluster_analysis": {
                "description": "Perform clustering analysis on embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Data to cluster"},
                        "n_clusters": {"type": "integer", "description": "Number of clusters", "default": 5}
                    },
                    "required": ["data"]
                }
            }
        }
        logger.info(f"Setup {len(self.tools)} tools")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC request"""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            logger.info(f"Handling request: {method}")
            
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": name,
                                "description": tool["description"],
                                "inputSchema": tool["parameters"]
                            }
                            for name, tool in self.tools.items()
                        ]
                    }
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                if tool_name not in self.tools:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": f"Tool '{tool_name}' not found"
                        }
                    }
                
                # Execute tool (mock implementation for now)
                result = await self.execute_tool(tool_name, tool_args)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": str(result)
                            }
                        ]
                    }
                }
            
            elif method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {
                                "listChanged": False
                            }
                        },
                        "serverInfo": {
                            "name": "laion-embeddings-mcp",
                            "version": "0.1.0"
                        }
                    }
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not found"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with given arguments"""
        logger.info(f"Executing tool {tool_name} with args: {args}")
        
        # Mock implementations for demonstration
        if tool_name == "generate_embedding":
            text = args.get("text", "")
            model = args.get("model", "thenlper/gte-small")
            return f"Generated embedding for '{text[:50]}...' using model {model} (mock result: [0.1, 0.2, 0.3, ...])"
        
        elif tool_name == "semantic_search":
            query = args.get("query", "")
            limit = args.get("limit", 10)
            return f"Semantic search results for '{query}' (top {limit} results - mock data)"
        
        elif tool_name == "cluster_analysis":
            data = args.get("data", [])
            n_clusters = args.get("n_clusters", 5)
            return f"Cluster analysis completed: {len(data)} items clustered into {n_clusters} groups (mock result)"
        
        else:
            return f"Tool {tool_name} executed successfully (mock result)"
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting minimal MCP server on stdio...")
        
        try:
            # Read from stdin and write to stdout
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    response = await self.handle_request(request)
                    
                    # Write response to stdout
                    response_line = json.dumps(response)
                    sys.stdout.write(response_line + "\n")
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    sys.stdout.write(json.dumps(error_response) + "\n")
                    sys.stdout.flush()
                    
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise

async def main():
    """Main entry point"""
    server = MinimalMCPServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
