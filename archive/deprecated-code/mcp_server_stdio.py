#!/usr/bin/env python3
"""
MCP Server for LAION Embeddings
Implements the Model Context Protocol (MCP) for Claude integration
"""

import asyncio
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Sequence

# MCP Protocol imports
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Local imports
from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
from src.mcp_server.config import MCPConfig
from src.mcp_server.session_manager import SessionManager
from src.mcp_server.error_handlers import handle_tool_error, ValidationError
from src.mcp_server.monitoring import MetricsCollector
from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/laion_mcp_server.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class LAIONMCPServer:
    """LAION Embeddings MCP Server"""
    
    def __init__(self):
        self.config = MCPConfig()
        self.session_manager = SessionManager(self.config)
        self.metrics_collector = MetricsCollector(config=self.config)
        self.tool_registry = ToolRegistry()
        
        # Initialize IPFS embeddings
        self.ipfs_embeddings_instance = ipfs_embeddings_py(resources={}, metadata={})
        
        # Initialize tools
        initialize_laion_tools(self.tool_registry, self.ipfs_embeddings_instance)
        
        # Create MCP server
        self.server = Server("laion-embeddings-mcp")
        
        logger.info(f"Initialized LAION MCP Server with {len(self.tool_registry.list_tools())} tools")
    
    def setup_handlers(self):
        """Set up MCP protocol handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List all available tools"""
            try:
                tools_data = self.tool_registry.get_all_tools()
                mcp_tools = []
                
                for tool_name, tool_instance in tools_data.items():
                    # Convert our tool format to MCP Tool format
                    tool = types.Tool(
                        name=tool_name,
                        description=tool_instance.description,
                        inputSchema=tool_instance.parameters_schema
                    )
                    mcp_tools.append(tool)
                
                logger.info(f"Listed {len(mcp_tools)} tools for MCP client")
                return mcp_tools
            
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                return []
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
            """Execute a tool with given arguments"""
            try:
                logger.info(f"Calling tool: {name} with arguments: {arguments}")
                
                # Check if tool exists
                if not self.tool_registry.has_tool(name):
                    error_msg = f"Tool '{name}' not found"
                    logger.error(error_msg)
                    return [types.TextContent(type="text", text=f"Error: {error_msg}")]
                
                # Validate arguments
                if not self.tool_registry.validate_tool_parameters(name, arguments):
                    error_msg = f"Invalid arguments for tool '{name}'"
                    logger.error(error_msg)
                    return [types.TextContent(type="text", text=f"Error: {error_msg}")]
                
                # Execute the tool
                result = await self.tool_registry.execute_tool(name, arguments)
                
                # Format result for MCP response
                if isinstance(result, dict):
                    response_text = json.dumps(result, indent=2, default=str)
                elif isinstance(result, (list, tuple)):
                    response_text = json.dumps(result, indent=2, default=str)
                else:
                    response_text = str(result)
                
                logger.info(f"Tool {name} executed successfully")
                return [types.TextContent(type="text", text=response_text)]
            
            except Exception as e:
                error_msg = f"Error executing tool '{name}': {str(e)}"
                logger.error(error_msg)
                return [types.TextContent(type="text", text=f"Error: {error_msg}")]
        
        @self.server.list_resources()
        async def list_resources() -> List[types.Resource]:
            """List available resources"""
            # For now, return empty list. Could be extended to include datasets, models, etc.
            return []
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource by URI"""
            # For now, not implemented
            raise NotImplementedError("Resource reading not yet implemented")
    
    async def run(self):
        """Run the MCP server"""
        try:
            # Set up all handlers
            self.setup_handlers()
            
            logger.info("Starting LAION MCP Server...")
            logger.info(f"Available tools: {list(self.tool_registry.get_all_tools().keys())}")
            
            # Run the server with stdio transport
            async with stdio_server() as streams:
                await self.server.run(
                    read_stream=streams[0],
                    write_stream=streams[1],
                    init_options={}
                )
                
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            raise

def main():
    """Main entry point"""
    try:
        server = LAIONMCPServer()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
