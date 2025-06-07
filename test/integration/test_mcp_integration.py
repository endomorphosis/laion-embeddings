#!/usr/bin/env python3
"""
Test the updated MCP server with proper service integration.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_service_integration():
    """Test the MCP server with service integration."""
    try:
        logger.info("Testing MCP server service integration...")
        
        # Import the main application
        from src.mcp_server.main import MCPServerApplication
        
        logger.info("✓ Successfully imported MCPServerApplication")
        
        # Try to create the application (but don't run it)
        app = MCPServerApplication()
        logger.info("✓ Successfully created MCPServerApplication instance")
        
        # Check if config is loaded
        logger.info(f"✓ Server name: {app.config.server_name}")
        logger.info(f"✓ Server version: {app.config.server_version}")
        
        # Try to initialize components without running
        try:
            await app._initialize_components()
            logger.info("✓ Successfully initialized all components")
            
            # Check service factory
            if app.service_factory:
                logger.info("✓ Service factory initialized")
                
                # Test service retrieval
                embedding_service = app.service_factory.get_embedding_service()
                vector_service = app.service_factory.get_vector_service()
                clustering_service = app.service_factory.get_clustering_service()
                ipfs_vector_service = app.service_factory.get_ipfs_vector_service()
                distributed_vector_service = app.service_factory.get_distributed_vector_service()
                
                logger.info(f"✓ Embedding service: {type(embedding_service).__name__}")
                logger.info(f"✓ Vector service: {type(vector_service).__name__}")
                logger.info(f"✓ Clustering service: {type(clustering_service).__name__}")
                logger.info(f"✓ IPFS vector service: {type(ipfs_vector_service).__name__}")
                logger.info(f"✓ Distributed vector service: {type(distributed_vector_service).__name__}")
                
            # Check tool registry
            if app.tool_registry:
                logger.info(f"✓ Tool registry with {len(app.tool_registry.tools)} tools")
                
                # List all registered tools
                for tool_name, tool in app.tool_registry.tools.items():
                    logger.info(f"  - {tool_name}: {type(tool).__name__}")
            
            # Clean shutdown
            await app._shutdown_components()
            logger.info("✓ Successfully shut down all components")
            
        except Exception as init_error:
            logger.error(f"✗ Component initialization failed: {init_error}")
            return False
        
        logger.info("🎉 All tests passed! MCP server integration is working.")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False

async def main():
    """Main test function."""
    success = await test_service_integration()
    if not success:
        sys.exit(1)
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
