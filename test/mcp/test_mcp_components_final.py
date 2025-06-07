#!/usr/bin/env python3
"""
MCP Server Component Test

Tests individual components of the MCP server.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_components():
    """Test MCP server components step by step."""
    
    print("=" * 60)
    print("MCP SERVER COMPONENT TEST")
    print("=" * 60)
    
    # Test 1: Basic imports
    print("\n1. Testing imports...")
    try:
        from src.mcp_server.config import MCPConfig
        from src.mcp_server.service_factory import ServiceFactory
        from src.mcp_server.main import MCPServerApplication
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Config creation
    print("\n2. Testing config creation...")
    try:
        config = MCPConfig()
        print(f"✓ Config created: {config.server_name}")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False
    
    # Test 3: Service factory
    print("\n3. Testing service factory...")
    try:
        factory = ServiceFactory(config)
        print("✓ Service factory created")
        
        services = await factory.initialize_services()
        print(f"✓ Services initialized: {list(services.keys())}")
    except Exception as e:
        print(f"✗ Service factory failed: {e}")
        return False
    
    # Test 4: MCP Application (creation only)
    print("\n4. Testing MCP application creation...")
    try:
        app = MCPServerApplication()
        print("✓ MCP application created")
        print(f"  - Server name: {app.config.server_name}")
        print(f"  - Version: {app.config.server_version}")
    except Exception as e:
        print(f"✗ MCP application creation failed: {e}")
        return False
    
    # Test 5: Component initialization
    print("\n5. Testing component initialization...")
    try:
        await app._initialize_components()
        print("✓ Components initialized")
        
        # Check components
        if app.service_factory:
            print("  ✓ Service factory initialized")
        if app.tool_registry:
            print("  ✓ Tool registry initialized")
            tools = app.tool_registry.get_all_tools()
            print(f"  ✓ Tools registered: {len(tools)}")
            for tool_name in sorted(tools.keys())[:5]:  # Show first 5
                print(f"    - {tool_name}")
            if len(tools) > 5:
                print(f"    ... and {len(tools) - 5} more")
        if app.mcp_server:
            print("  ✓ MCP server initialized")
            
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL COMPONENT TESTS PASSED")
    print("=" * 60)
    
    return True

async def main():
    """Main test function."""
    try:
        success = await test_mcp_components()
        if success:
            print("\n🎉 MCP component test completed successfully!")
            return 0
        else:
            print("\n❌ MCP component test failed!")
            return 1
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
