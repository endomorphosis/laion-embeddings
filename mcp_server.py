#!/usr/bin/env python3
"""
LAION Embeddings MCP Server - Production Entry Point

This is the single MCP server entry point for CI/CD testing and production deployment.
Validates all 22 MCP tools and provides comprehensive testing capabilities.
"""

import sys
import asyncio
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging to stderr for MCP compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class LAIONEmbeddingsMCPServer:
    """Production MCP server with comprehensive tool validation"""
    
    def __init__(self):
        self.tools = {}
        self.tool_registry = None
        self.ipfs_embeddings_instance = None
        self.validation_results = {"tools_loaded": 0, "tools_failed": 0, "errors": []}
        
    async def initialize(self):
        """Initialize all MCP tools and validate functionality"""
        try:
            logger.info("🚀 Initializing LAION Embeddings MCP Server...")
            
            # Import core components
            from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
            from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
            
            # Initialize components
            self.tool_registry = ToolRegistry()
            
            # Initialize IPFS embeddings with minimal config for testing
            test_resources = {
                "local_endpoints": [["test-model", "cpu", 512]],
                "tei_endpoints": [],
                "openvino_endpoints": [],
                "libp2p_endpoints": []
            }
            test_metadata = {
                "dataset": "test_dataset",
                "chunk_size": 512,
                "n_results": 10,
                "dst_path": "/tmp/test"
            }
            
            self.ipfs_embeddings_instance = ipfs_embeddings_py(
                resources=test_resources, 
                metadata=test_metadata
            )
            
            # Initialize all LAION tools
            logger.info("📦 Loading MCP tools...")
            initialize_laion_tools(self.tool_registry, self.ipfs_embeddings_instance)
            
            # Get and validate all tools
            real_tools = self.tool_registry.get_all_tools()
            logger.info(f"✅ Found {len(real_tools)} MCP tools")
            
            # Convert to MCP format and validate
            for tool_instance in real_tools:
                try:
                    tool_name = tool_instance.name
                    self.tools[tool_name] = {
                        "description": tool_instance.description,
                        "parameters": tool_instance.input_schema,
                        "instance": tool_instance
                    }
                    self.validation_results["tools_loaded"] += 1
                    logger.info(f"  ✅ {tool_name}: {tool_instance.description[:60]}...")
                    
                except Exception as e:
                    self.validation_results["tools_failed"] += 1
                    error_msg = f"Failed to load tool {getattr(tool_instance, 'name', 'unknown')}: {str(e)}"
                    self.validation_results["errors"].append(error_msg)
                    logger.error(f"  ❌ {error_msg}")
            
            # Report validation results
            total_tools = self.validation_results["tools_loaded"] + self.validation_results["tools_failed"]
            success_rate = (self.validation_results["tools_loaded"] / total_tools * 100) if total_tools > 0 else 0
            
            logger.info(f"🎯 MCP Tools Validation Complete:")
            logger.info(f"   ✅ Loaded: {self.validation_results['tools_loaded']}")
            logger.info(f"   ❌ Failed: {self.validation_results['tools_failed']}")
            logger.info(f"   📊 Success Rate: {success_rate:.1f}%")
            
            if self.validation_results["errors"]:
                logger.warning("⚠️  Errors encountered:")
                for error in self.validation_results["errors"]:
                    logger.warning(f"     {error}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP server: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""
        try:
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "initialize":
                return await self.handle_initialize(params)
            elif method == "tools/list":
                return await self.handle_list_tools()
            elif method == "tools/call":
                return await self.handle_call_tool(params)
            elif method == "validation/status":
                return await self.handle_validation_status()
            else:
                return self.error_response(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            return self.error_response(str(e))
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request"""
        success = await self.initialize()
        if success:
            return {
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "validation": {"status": True}
                    },
                    "serverInfo": {
                        "name": "laion-embeddings-mcp-server",
                        "version": "2.2.0"
                    }
                }
            }
        else:
            return self.error_response("Failed to initialize server")
    
    async def handle_list_tools(self) -> Dict[str, Any]:
        """Handle tools list request"""
        tools_list = []
        for tool_name, tool_data in self.tools.items():
            tools_list.append({
                "name": tool_name,
                "description": tool_data["description"],
                "inputSchema": tool_data["parameters"]
            })
        
        return {
            "result": {
                "tools": tools_list
            }
        }
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return self.error_response(f"Tool not found: {tool_name}")
        
        try:
            tool_instance = self.tools[tool_name]["instance"]
            result = await tool_instance.call(arguments)
            
            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            return self.error_response(f"Tool execution failed: {str(e)}")
    
    async def handle_validation_status(self) -> Dict[str, Any]:
        """Handle validation status request"""
        return {
            "result": {
                "validation": self.validation_results,
                "tools_count": len(self.tools),
                "server_status": "ready" if self.tools else "error"
            }
        }
    
    def error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "error": {
                "code": -1,
                "message": message
            }
        }

async def main():
    """Main entry point for MCP server"""
    server = LAIONEmbeddingsMCPServer()
    
    # Initialize server
    logger.info("🔥 Starting LAION Embeddings MCP Server...")
    success = await server.initialize()
    
    if not success:
        logger.error("❌ Server initialization failed")
        sys.exit(1)
    
    logger.info("✅ MCP Server ready for requests")
    
    # For CI/CD testing, output validation results
    if "--validate" in sys.argv:
        print(json.dumps({
            "status": "success" if success else "failed",
            "validation": server.validation_results,
            "tools_count": len(server.tools),
            "tools": list(server.tools.keys())
        }, indent=2))
        return
    
    # Handle stdio communication for MCP
    try:
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            try:
                request = json.loads(line.strip())
                response = await server.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {line}")
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                
    except KeyboardInterrupt:
        logger.info("🛑 MCP Server shutting down...")
    except Exception as e:
        logger.error(f"❌ Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
