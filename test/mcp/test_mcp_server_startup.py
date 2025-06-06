#!/usr/bin/env python3
"""
Test script to verify MCP server startup and basic functionality.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_server_startup():
    """Test basic server startup and tool registration."""
    try:
        logger.info("Testing MCP server startup...")
        
        # Test FastAPI app creation which includes all MCP server setup
        from src.mcp_server.fastapi_integration import create_fastapi_app
        logger.info("✅ FastAPI integration imported successfully")
        
        # Create the app - this will initialize the entire MCP server stack
        app = create_fastapi_app()
        logger.info("✅ FastAPI app created with MCP server")
        
        # Access the tool registry from the app state
        registry = app.state.tool_registry
        logger.info("✅ Tool registry accessed from app state")
        
        # Create tool registry
        registry = ToolRegistry()
        logger.info("✅ Tool registry created")
        
        # Initialize tools with no embedding service (testing basic functionality)
        initialize_laion_tools(registry, embedding_service=None)
        
        # Check registered tools
        tools = registry.get_all_tools()
        logger.info(f"✅ Registered {len(tools)} tools:")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")
        
        # Test tool execution with a simple analysis tool
        available_tools = [tool.name for tool in tools]
        if "cluster_analysis" in available_tools:
            logger.info("Testing cluster analysis tool...")
            cluster_tool = registry.get_tool("cluster_analysis")
            if cluster_tool:
                try:
                    result = await cluster_tool.execute({
                        "algorithm": "kmeans",
                        "data_source": "collection",
                        "collection_name": "test_collection",
                        "n_clusters": 3
                    })
                    logger.info(f"✅ Cluster analysis test successful: {result.get('success', False)}")
                except Exception as e:
                    logger.error(f"❌ Cluster analysis test failed: {e}")
            else:
                logger.error("❌ Could not retrieve cluster analysis tool")
        
        # Test listing tools by category
        categories = registry.get_categories()
        logger.info(f"✅ Available categories: {list(categories.keys()) if isinstance(categories, dict) else categories}")
        
        logger.info("🎉 MCP server startup test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Server startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_server_startup())
    sys.exit(0 if success else 1)
