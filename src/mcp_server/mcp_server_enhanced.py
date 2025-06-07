#!/usr/bin/env python3
"""
Enhanced LAION Embeddings MCP Server
Integrates with real LAION embeddings tools and infrastructure
"""

import json
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Configure logging to stderr so it doesn't interfere with stdio
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class LAIONEmbeddingsMCPServer:
    """Enhanced MCP server with real LAION embeddings integration"""
    
    def __init__(self):
        self.tools = {}
        self.tool_registry = None
        self.setup_tools()
    
    def setup_tools(self):
        """Setup LAION tools with real implementations"""
        try:
            # Try to import and initialize real tools
            from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
            from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
            
            # Initialize components
            self.tool_registry = ToolRegistry()
            ipfs_embeddings_instance = ipfs_embeddings_py(resources={}, metadata={})
            
            # Initialize real LAION tools
            initialize_laion_tools(self.tool_registry, ipfs_embeddings_instance)
            
            # Get tools from registry
            real_tools = self.tool_registry.get_all_tools()
            
            # Convert to MCP format
            for tool_instance in real_tools:
                tool_name = tool_instance.name
                self.tools[tool_name] = {
                    "description": tool_instance.description,
                    "parameters": tool_instance.input_schema,
                    "instance": tool_instance
                }
            
            logger.info(f"Successfully loaded {len(self.tools)} real LAION tools")
            
        except Exception as e:
            logger.warning(f"Failed to load real LAION tools, using fallback: {e}")
            self.setup_fallback_tools()
    
    def setup_fallback_tools(self):
        """Setup fallback mock tools if real tools fail to load"""
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
                },
                "instance": None
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
                },
                "instance": None
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
                },
                "instance": None
            },
            "storage_management": {
                "description": "Manage IPFS storage for embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "Storage action", "enum": ["list", "add", "remove"]},
                        "path": {"type": "string", "description": "Storage path"}
                    },
                    "required": ["action"]
                },
                "instance": None
            },
            "collection_management": {
                "description": "Manage embedding collections",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "description": "Collection operation", "enum": ["create", "list", "delete"]},
                        "collection_name": {"type": "string", "description": "Collection name"}
                    },
                    "required": ["operation"]
                },
                "instance": None
            }
        }
        logger.info(f"Setup {len(self.tools)} fallback tools")
    
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
                
                # Execute tool
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
        
        tool_info = self.tools.get(tool_name)
        if not tool_info:
            return f"Tool {tool_name} not found"
        
        tool_instance = tool_info.get("instance")
        
        # Try to use real tool implementation
        if tool_instance and self.tool_registry:
            try:
                result = await self.tool_registry.execute_tool(tool_name, args)
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Real tool execution failed for {tool_name}: {e}")
                # Fall back to mock implementation
        
        # Fallback mock implementations
        if tool_name == "generate_embedding":
            text = args.get("text", "")
            model = args.get("model", "thenlper/gte-small")
            return f"Generated embedding for '{text[:100]}...' using model {model}\n[Mock result: 768-dimensional vector]"
        
        elif tool_name == "semantic_search":
            query = args.get("query", "")
            limit = args.get("limit", 10)
            return f"Semantic search results for '{query}':\n1. Result 1 (similarity: 0.95)\n2. Result 2 (similarity: 0.87)\n...\n[{limit} results total - mock data]"
        
        elif tool_name == "cluster_analysis":
            data = args.get("data", [])
            n_clusters = args.get("n_clusters", 5)
            return f"Cluster analysis completed:\n- Input: {len(data)} data points\n- Clusters: {n_clusters}\n- Silhouette score: 0.72 (mock)\n- Centroids calculated"
        
        elif tool_name == "storage_management":
            action = args.get("action", "list")
            path = args.get("path", "/")
            return f"Storage {action} operation on path '{path}'\nStatus: Success\n[Mock IPFS storage result]"
        
        elif tool_name == "collection_management":
            operation = args.get("operation", "list")
            collection_name = args.get("collection_name", "default")
            return f"Collection {operation} for '{collection_name}'\nStatus: Success\n[Mock collection management result]"
        
        else:
            return f"Tool {tool_name} executed successfully (fallback mock result)"
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Enhanced LAION MCP server on stdio...")
        
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
    server = LAIONEmbeddingsMCPServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
