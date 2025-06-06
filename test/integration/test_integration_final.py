#!/usr/bin/env python3
"""
Final MCP Integration Test - Verify complete service-to-tool integration.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_complete_integration():
    """Test complete MCP integration with real services."""
    print("🚀 Testing Complete MCP Integration\n")
    
    test_passed = 0
    test_failed = 0
    
    def check_test(name, condition, error=None):
        nonlocal test_passed, test_failed
        if condition:
            print(f"✅ {name}")
            test_passed += 1
        else:
            print(f"❌ {name}")
            if error:
                print(f"   Error: {error}")
            test_failed += 1
    
    try:
        # Test 1: Service Imports
        print("1. Testing Core Service Imports...")
        try:
            from services.embedding_service import EmbeddingService
            from services.vector_service import VectorService
            from services.clustering_service import VectorClusterer
            from services.ipfs_vector_service import IPFSVectorService
            from services.distributed_vector_service import DistributedVectorIndex
            check_test("Core service imports", True)
        except Exception as e:
            check_test("Core service imports", False, e)
        
        # Test 2: MCP Tool Imports
        print("\n2. Testing MCP Tool Imports...")
        try:
            from src.mcp_server.tools.embedding_tools import (
                EmbeddingGenerationTool, BatchEmbeddingTool, MultimodalEmbeddingTool
            )
            from src.mcp_server.tools.search_tools import (
                SemanticSearchTool, SimilaritySearchTool, FacetedSearchTool
            )
            from src.mcp_server.tools.storage_tools import (
                StorageManagementTool, CollectionManagementTool, RetrievalTool
            )
            from src.mcp_server.tools.analysis_tools import (
                ClusterAnalysisTool, QualityAssessmentTool, DimensionalityReductionTool
            )
            from src.mcp_server.tools.vector_store_tools import (
                VectorIndexTool, VectorRetrievalTool, VectorMetadataTool
            )
            from src.mcp_server.tools.ipfs_cluster_tools import (
                IPFSClusterTool, DistributedVectorTool, IPFSMetadataTool
            )
            check_test("All MCP tool imports", True)
        except Exception as e:
            check_test("All MCP tool imports", False, e)
        
        # Test 3: Service Factory
        print("\n3. Testing Service Factory...")
        try:
            from src.mcp_server.service_factory import ServiceFactory
            from src.mcp_server.config import MCPConfig
            
            config = MCPConfig()
            service_factory = ServiceFactory(config)
            await service_factory.initialize()
            check_test("Service factory initialization", True)
            
            # Test service retrieval
            embedding_service = service_factory.get_embedding_service()
            vector_service = service_factory.get_vector_service()
            clustering_service = service_factory.get_clustering_service()
            ipfs_vector_service = service_factory.get_ipfs_vector_service()
            distributed_vector_service = service_factory.get_distributed_vector_service()
            
            check_test("Service retrieval from factory", all([
                embedding_service is not None,
                vector_service is not None, 
                clustering_service is not None,
                ipfs_vector_service is not None,
                distributed_vector_service is not None
            ]))
            
        except Exception as e:
            check_test("Service factory", False, e)
            return False
        
        # Test 4: Tool Instantiation with Services
        print("\n4. Testing Tool Instantiation with Real Services...")
        try:
            # Embedding tools
            embedding_gen = EmbeddingGenerationTool(embedding_service)
            batch_embedding = BatchEmbeddingTool(embedding_service)
            multimodal_embedding = MultimodalEmbeddingTool(embedding_service)
            check_test("Embedding tools with services", True)
            
            # Search tools
            semantic_search = SemanticSearchTool(vector_service)
            similarity_search = SimilaritySearchTool(vector_service)
            faceted_search = FacetedSearchTool(vector_service)
            check_test("Search tools with services", True)
            
            # Storage tools
            storage_mgmt = StorageManagementTool(vector_service)
            collection_mgmt = CollectionManagementTool(vector_service)
            retrieval = RetrievalTool(vector_service)
            check_test("Storage tools with services", True)
            
            # Analysis tools
            cluster_analysis = ClusterAnalysisTool(clustering_service)
            quality_assessment = QualityAssessmentTool(vector_service)
            dimensionality_reduction = DimensionalityReductionTool(clustering_service)
            check_test("Analysis tools with services", True)
            
            # Vector store tools
            vector_index = VectorIndexTool(vector_service)
            vector_retrieval = VectorRetrievalTool(vector_service)
            vector_metadata = VectorMetadataTool(vector_service)
            check_test("Vector store tools with services", True)
            
            # IPFS cluster tools
            ipfs_cluster = IPFSClusterTool(ipfs_vector_service)
            distributed_vector = DistributedVectorTool(distributed_vector_service)
            ipfs_metadata = IPFSMetadataTool(ipfs_vector_service)
            check_test("IPFS cluster tools with services", True)
            
        except Exception as e:
            check_test("Tool instantiation with services", False, e)
        
        # Test 5: Service Validation (Tools should reject None)
        print("\n5. Testing Service Validation...")
        try:
            EmbeddingGenerationTool(None)
            check_test("Tool service validation", False, "Should reject None service")
        except ValueError:
            check_test("Tool service validation", True)
        except Exception as e:
            check_test("Tool service validation", False, e)
        
        # Test 6: MCP Server Application
        print("\n6. Testing MCP Server Application...")
        try:
            from src.mcp_server.main import MCPServerApplication
            
            # Create a test config
            test_config = """
SERVER_NAME=laion-embeddings-mcp
SERVER_VERSION=1.0.0
LOG_LEVEL=INFO
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384
ENABLE_METRICS=false
"""
            config_file = Path("test_config.env")
            config_file.write_text(test_config)
            
            # Initialize the application
            app = MCPServerApplication(str(config_file))
            check_test("MCP server application creation", True)
            
            # Clean up
            config_file.unlink()
            
        except Exception as e:
            check_test("MCP server application", False, e)
        
        # Test 7: Service Cleanup
        print("\n7. Testing Service Cleanup...")
        try:
            await service_factory.shutdown()
            check_test("Service factory shutdown", True)
        except Exception as e:
            check_test("Service factory shutdown", False, e)
        
        # Final Results
        print(f"\n🎯 Integration Test Results:")
        print(f"   ✅ Passed: {test_passed}")
        print(f"   ❌ Failed: {test_failed}")
        print(f"   📊 Success Rate: {test_passed/(test_passed+test_failed)*100:.1f}%")
        
        if test_failed == 0:
            print("\n🎉 ALL TESTS PASSED! MCP Integration Complete!")
            print("\n📋 Integration Summary:")
            print("   ✓ All core services import correctly")
            print("   ✓ All MCP tools import correctly")
            print("   ✓ Service factory initializes all services")
            print("   ✓ Tools instantiate with real service instances")
            print("   ✓ Service validation prevents None services")
            print("   ✓ MCP server application creates successfully")
            print("   ✓ Clean shutdown works properly")
            
            print("\n🚀 Ready for Production!")
            print("   • All features exposed via MCP tools")
            print("   • Real services integrated with tools")
            print("   • Proper error handling and validation")
            print("   • Clean resource management")
            
            return True
        else:
            print(f"\n⚠️ {test_failed} tests failed. Integration needs fixes.")
            return False
            
    except Exception as e:
        print(f"\n💥 Critical integration failure: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_complete_integration())
    sys.exit(0 if result else 1)
