#!/usr/bin/env python3
"""
Debug script to test MCP server startup and identify issues.
"""
import sys
import traceback
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test each import individually to identify issues."""
    logger.info("Testing imports...")
    
    try:
        logger.info("Importing tool_registry...")
        from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
        logger.info("✓ tool_registry imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import tool_registry: {e}")
        traceback.print_exc()
        return False
    
    try:
        logger.info("Importing analysis_tools...")
        from src.mcp_server.tools.analysis_tools import ClusterAnalysisTool, QualityAssessmentTool, DimensionalityReductionTool
        logger.info("✓ analysis_tools imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import analysis_tools: {e}")
        traceback.print_exc()
        return False
    
    try:
        logger.info("Creating ToolRegistry...")
        registry = ToolRegistry()
        logger.info("✓ ToolRegistry created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create ToolRegistry: {e}")
        traceback.print_exc()
        return False
    
    try:
        logger.info("Testing tool initialization...")
        initialize_laion_tools(registry, None)
        logger.info("✓ Tools initialized successfully")
        
        tools = registry.get_all_tools()
        logger.info(f"✓ Registered {len(tools)} tools:")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")
            
    except Exception as e:
        logger.error(f"✗ Failed to initialize tools: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_fastapi_creation():
    """Test FastAPI app creation."""
    logger.info("Testing FastAPI app creation...")
    
    try:
        from src.mcp_server.fastapi_integration import create_fastapi_app
        logger.info("✓ FastAPI integration imported successfully")
        
        app = create_fastapi_app()
        logger.info("✓ FastAPI app created successfully")
        
        # Test accessing the registry
        tools_count = len(app.state.tool_registry.get_all_tools())
        logger.info(f"✓ App has {tools_count} tools registered")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create FastAPI app: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("Starting MCP server startup debug tests...")
    
    success = True
    
    # Test individual components
    if not test_imports():
        success = False
    
    if not test_fastapi_creation():
        success = False
    
    if success:
        logger.info("🎉 All tests passed! MCP server should start successfully.")
    else:
        logger.error("❌ Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
