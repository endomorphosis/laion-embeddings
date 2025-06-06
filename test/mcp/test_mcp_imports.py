#!/usr/bin/env python3
"""
Simple test to check MCP imports and basic functionality
"""

import sys
import traceback

def test_basic_imports():
    """Test basic imports"""
    try:
        print("Testing basic imports...")
        
        # Test MCP import
        try:
            import mcp
            print(f"✓ MCP imported successfully: {mcp.__version__ if hasattr(mcp, '__version__') else 'unknown version'}")
        except ImportError as e:
            print(f"✗ MCP import failed: {e}")
            return False
            
        # Test FastMCP import
        try:
            from mcp.server import FastMCP
            print("✓ FastMCP imported successfully")
        except ImportError as e:
            print(f"✗ FastMCP import failed: {e}")
            try:
                from mcp.server.fastmcp import FastMCP
                print("✓ FastMCP imported from alternative path")
            except ImportError as e2:
                print(f"✗ FastMCP import failed from both paths: {e}, {e2}")
                return False
        
        # Test creating a FastMCP instance
        try:
            mcp_server = FastMCP("test-server")
            print("✓ FastMCP instance created successfully")
        except Exception as e:
            print(f"✗ Failed to create FastMCP instance: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_laion_imports():
    """Test LAION-specific imports"""
    try:
        print("\nTesting LAION imports...")
        
        # Test tool registry import
        try:
            from src.mcp_server.tool_registry import ToolRegistry
            print("✓ ToolRegistry imported successfully")
        except ImportError as e:
            print(f"✗ ToolRegistry import failed: {e}")
            return False
            
        # Test config import
        try:
            from src.mcp_server.config import MCPConfig
            print("✓ MCPConfig imported successfully")
        except ImportError as e:
            print(f"✗ MCPConfig import failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error in LAION imports: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== MCP Server Import Test ===")
    
    basic_ok = test_basic_imports()
    laion_ok = test_laion_imports()
    
    if basic_ok and laion_ok:
        print("\n✓ All imports successful! MCP server should work.")
        sys.exit(0)
    else:
        print("\n✗ Some imports failed. Check the errors above.")
        sys.exit(1)
