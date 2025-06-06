#!/usr/bin/env python3
"""
Simple MCP Server Test
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def simple_test():
    print("Starting simple MCP test...")
    
    try:
        from src.mcp_server.service_factory import ServiceFactory
        from src.mcp_server.config import MCPConfig
        print("✓ Imports successful")
        
        config = MCPConfig()
        print(f"✓ MCPConfig created")
        
        factory = ServiceFactory(config)
        print("✓ ServiceFactory initialized")
        
        services = await factory.initialize_services()
        print(f"✓ Services initialized: {list(services.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(simple_test())
    print(f"Test result: {'PASS' if result else 'FAIL'}")
