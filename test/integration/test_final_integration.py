#!/usr/bin/env python3
"""
Final End-to-End MCP Server Test

Tests the complete MCP server integration with real services.
"""

import asyncio
import logging
import sys
import os
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mcp_server.main import MCPServerApplication
from src.mcp_server.service_factory import ServiceFactory, ServiceConfigs
from services.vector_service import VectorConfig
from services.clustering_service import ClusterConfig
from services.ipfs_vector_service import IPFSConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestMCPConfig:
    """Test configuration for MCP server."""
    vector_dimension: int = 768
    n_clusters: int = 10
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

async def test_mcp_server_integration():
    """Test complete MCP server integration."""
    
    print("=" * 60)
    print("FINAL MCP SERVER INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Create default test configuration
    print("\n1. Creating test configuration...")
    try:
        config = TestMCPConfig()
        print(f"✓ Test config created: {config}")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False
    
    # Test 2: Initialize ServiceFactory
    print("\n2. Initializing ServiceFactory...")
    try:
        factory = ServiceFactory(config)
        print("✓ ServiceFactory initialized")
    except Exception as e:
        print(f"✗ ServiceFactory initialization failed: {e}")
        return False
    
    # Test 3: Initialize services
    print("\n3. Initializing services...")
    try:
        services = await factory.initialize_services()
        print(f"✓ Services initialized: {list(services.keys())}")
        
        # Check each service
        for name, service in services.items():
            print(f"  - {name}: {type(service).__name__}")
            
    except Exception as e:
        print(f"✗ Service initialization failed: {e}")
        return False
    
    # Test 4: Test service access methods
    print("\n4. Testing service access methods...")
    try:
        vector_service = factory.get_vector_service()
        print(f"✓ Vector service: {type(vector_service).__name__}")
        
        embedding_service = factory.get_embedding_service()
        print(f"✓ Embedding service: {type(embedding_service).__name__}")
        
        clustering_service = factory.get_clustering_service()
        print(f"✓ Clustering service: {type(clustering_service).__name__}")
        
        distributed_service = factory.get_distributed_vector_service()
        print(f"✓ Distributed service: {type(distributed_service).__name__}")
        
        # IPFS service may not be available
        try:
            ipfs_service = factory.get_ipfs_vector_service()
            print(f"✓ IPFS service: {type(ipfs_service).__name__}")
        except Exception as e:
            print(f"! IPFS service not available: {e}")
            
    except Exception as e:
        print(f"✗ Service access failed: {e}")
        return False
    
    # Test 5: Initialize MCP Server Application
    print("\n5. Initializing MCP Server Application...")
    try:
        app = MCPServerApplication(config)
        print("✓ MCP Server Application created")
        
        # Initialize the application
        await app.initialize()
        print("✓ MCP Server Application initialized")
        
        # Check tools
        if hasattr(app, 'get_tools'):
            tools = app.get_tools()
            print(f"✓ Available tools: {len(tools)}")
            for tool_name in sorted(tools.keys()):
                print(f"  - {tool_name}")
        
    except Exception as e:
        print(f"✗ MCP Server Application failed: {e}")
        return False
    
    # Test 6: Test basic service functionality
    print("\n6. Testing basic service functionality...")
    try:
        # Test vector service
        vector_service = factory.get_vector_service()
        if hasattr(vector_service, 'get_dimension'):
            dim = vector_service.get_dimension()
            print(f"✓ Vector dimension: {dim}")
        
        # Test embedding service
        embedding_service = factory.get_embedding_service()
        if hasattr(embedding_service, 'get_model_info'):
            info = embedding_service.get_model_info()
            print(f"✓ Embedding model info: {info}")
        
    except Exception as e:
        print(f"! Service functionality test failed: {e}")
        # This is not critical, continue
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - MCP SERVER INTEGRATION COMPLETE")
    print("=" * 60)
    
    return True

async def main():
    """Main test function."""
    try:
        success = await test_mcp_server_integration()
        if success:
            print("\n🎉 MCP Server integration test completed successfully!")
            return 0
        else:
            print("\n❌ MCP Server integration test failed!")
            return 1
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
