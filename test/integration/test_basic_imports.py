#!/usr/bin/env python3
"""
Test basic imports to verify the MCP server can be imported
"""

import sys
import os
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

try:
    print("Testing basic Python...")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    print("Testing imports...")
    import json
    print("✅ json imported")
    
    import asyncio
    print("✅ asyncio imported")
    
    import logging
    print("✅ logging imported")
    
    # Test if our MCP server can be imported
    print("Testing MCP server import...")
    try:
        import mcp_server_minimal
        print("✅ mcp_server_minimal imported successfully")
    except Exception as e:
        print(f"❌ Error importing mcp_server_minimal: {e}")
        import traceback
        traceback.print_exc()
    
    print("All basic tests completed!")
    
except Exception as e:
    print(f"❌ Error during basic tests: {e}")
    import traceback
    traceback.print_exc()
