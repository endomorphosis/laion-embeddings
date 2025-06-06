#!/usr/bin/env python3
"""
Test script to validate MCP server setup
"""

import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mcp_imports():
    """Test MCP package imports"""
    try:
        from mcp.server import FastMCP
        logger.info("✓ FastMCP import successful")
        return True
    except ImportError as e:
        logger.error(f"✗ FastMCP import failed: {e}")
        return False

def test_laion_imports():
    """Test LAION embeddings imports"""
    try:
        from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
        from src.mcp_server.config import MCPConfig
        from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
        logger.info("✓ LAION imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ LAION imports failed: {e}")
        return False

def test_tool_registry():
    """Test tool registry initialization"""
    try:
        from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
        from src.mcp_server.config import MCPConfig
        from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
        
        config = MCPConfig()
        tool_registry = ToolRegistry()
        ipfs_embeddings_instance = ipfs_embeddings_py(resources={}, metadata={})
        
        # Initialize tools
        initialize_laion_tools(tool_registry, ipfs_embeddings_instance)
        
        # Get tools
        tools_data = tool_registry.get_all_tools()
        logger.info(f"✓ Tool registry initialized with {len(tools_data)} tools")
        logger.info(f"  Available tools: {list(tools_data.keys())}")
        return True
    except Exception as e:
        logger.error(f"✗ Tool registry test failed: {e}")
        return False

def test_fastmcp_setup():
    """Test FastMCP server setup"""
    try:
        from mcp.server import FastMCP
        
        # Create a simple FastMCP instance
        mcp = FastMCP("test-server")
        
        # Add a simple test tool
        @mcp.tool()
        def test_tool(message: str) -> str:
            """A simple test tool"""
            return f"Test response: {message}"
        
        logger.info("✓ FastMCP server setup successful")
        return True
    except Exception as e:
        logger.error(f"✗ FastMCP setup failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing MCP server components...")
    
    tests = [
        ("MCP Imports", test_mcp_imports),
        ("LAION Imports", test_laion_imports), 
        ("Tool Registry", test_tool_registry),
        ("FastMCP Setup", test_fastmcp_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
        else:
            logger.error(f"{test_name} test failed")
    
    logger.info(f"\nTest Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ All tests passed! MCP server should be ready.")
        sys.exit(0)
    else:
        logger.error("✗ Some tests failed. Check the errors above.")
        sys.exit(1)
