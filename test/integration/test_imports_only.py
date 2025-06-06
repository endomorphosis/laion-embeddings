#!/usr/bin/env python3
"""
Simple import test for MCP components
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    print("Testing basic imports...")
    
    try:
        print("1. Testing config import...")
        from src.mcp_server.config import MCPConfig
        print("✓ MCPConfig imported")
        
        print("2. Testing service factory import...")
        from src.mcp_server.service_factory import ServiceFactory
        print("✓ ServiceFactory imported")
        
        print("3. Testing main import...")
        from src.mcp_server.main import MCPServerApplication
        print("✓ MCPServerApplication imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    print(f"Test result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
