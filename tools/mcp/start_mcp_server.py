#!/usr/bin/env python3
"""
LAION Embeddings MCP Server
A Model Context Protocol server for Claude integration with LAION embeddings functionality
"""

import sys
import logging
from pathlib import Path

# MCP imports
from mcp.server import FastMCP

# Local imports  
from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
from src.mcp_server.config import MCPConfig
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

# Initialize FastMCP server
mcp = FastMCP("laion-embeddings")

def setup_laion_tools():
    """Set up LAION embedding tools for the MCP server"""
    try:
        # Initialize components
        config = MCPConfig()
        tool_registry = ToolRegistry()
        ipfs_embeddings_instance = ipfs_embeddings_py(resources={}, metadata={})
        
        # Initialize tools
        initialize_laion_tools(tool_registry, ipfs_embeddings_instance)
        
        # Get all tools from registry
        tools_data = tool_registry.get_all_tools()
        
        logger.info(f"Setting up {len(tools_data)} LAION tools for MCP server")
        
        # Register each tool with FastMCP
        for tool_name, tool_instance in tools_data.items():
            
            def create_tool_function(tool_name, tool_registry):
                async def tool_function(**kwargs):
                    """Dynamic tool function"""
                    try:
                        logger.info(f"Executing tool: {tool_name} with args: {kwargs}")
                        result = await tool_registry.execute_tool(tool_name, kwargs)
                        return str(result)
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        logger.error(error_msg)
                        return error_msg
                
                return tool_function
            
            # Create the tool function
            tool_func = create_tool_function(tool_name, tool_registry)
            tool_func.__name__ = tool_name
            tool_func.__doc__ = tool_instance.description
            
            # Register with FastMCP
            mcp.tool(name=tool_name, description=tool_instance.description)(tool_func)
            
        logger.info(f"Successfully registered {len(tools_data)} tools with FastMCP")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up LAION tools: {e}")
        return False

if __name__ == "__main__":
    try:
        logger.info("Starting LAION Embeddings MCP Server...")
        
        # Setup tools
        if not setup_laion_tools():
            logger.error("Failed to setup LAION tools")
            sys.exit(1)
            
        logger.info("Running LAION MCP server with stdio transport...")
        mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
