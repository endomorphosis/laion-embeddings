#!/usr/bin/env python3
"""
Quick MCP Integration Test
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    try:
        print("Testing MCP Integration...")
        
        # Test service factory
        from src.mcp_server.service_factory import ServiceFactory
        from src.mcp_server.config import MCPConfig
        
        config = MCPConfig()
        factory = ServiceFactory(config)
        print("✅ Service factory created")
        
        # Initialize services
        await factory.initialize()
        print("✅ Services initialized")
        
        # Get services
        embedding_service = factory.get_embedding_service()
        vector_service = factory.get_vector_service()
        clustering_service = factory.get_clustering_service()
        ipfs_service = factory.get_ipfs_vector_service()
        distributed_service = factory.get_distributed_vector_service()
        
        print(f"✅ Services retrieved: {len([s for s in [embedding_service, vector_service, clustering_service, ipfs_service, distributed_service] if s is not None])}/5")
        
        # Test tool imports
        from src.mcp_server.tools.embedding_tools import EmbeddingGenerationTool
        from src.mcp_server.tools.search_tools import SemanticSearchTool
        from src.mcp_server.tools.analysis_tools import ClusterAnalysisTool
        print("✅ Tool imports working")
        
        # Test tool creation with services
        embedding_tool = EmbeddingGenerationTool(embedding_service)
        search_tool = SemanticSearchTool(vector_service)
        cluster_tool = ClusterAnalysisTool(clustering_service)
        print("✅ Tools created with services")
        
        # Cleanup
        await factory.shutdown()
        print("✅ Services shutdown")
        
        print("\n🎉 MCP Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
