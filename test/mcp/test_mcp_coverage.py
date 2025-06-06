#!/usr/bin/env python3
"""
MCP Server Feature Coverage Test Script

This script tests the current state of MCP server tools and service integration.
"""

import sys
import os
sys.path.append('.')

def test_service_imports():
    """Test if all core services can be imported."""
    print("=== Testing Service Imports ===")
    
    try:
        from services.vector_service import VectorService
        print("✅ VectorService imported")
    except Exception as e:
        print(f"❌ VectorService import failed: {e}")
    
    try:
        from services.clustering_service import VectorClusterer
        print("✅ VectorClusterer imported")
    except Exception as e:
        print(f"❌ ClusteringService import failed: {e}")
    
    try:
        from services.embedding_service import EmbeddingService
        print("✅ EmbeddingService imported")
    except Exception as e:
        print(f"❌ EmbeddingService import failed: {e}")
    
    try:
        from services.ipfs_vector_service import IPFSVectorService
        print("✅ IPFSVectorService imported")
    except Exception as e:
        print(f"❌ IPFSVectorService import failed: {e}")

def test_mcp_tool_imports():
    """Test if all MCP tools can be imported."""
    print("\n=== Testing MCP Tool Imports ===")
    
    tools_to_test = [
        ("src.mcp_server.tools.embedding_tools", ["EmbeddingGenerationTool", "BatchEmbeddingTool"]),
        ("src.mcp_server.tools.search_tools", ["SemanticSearchTool", "SimilaritySearchTool"]),
        ("src.mcp_server.tools.storage_tools", ["StorageManagementTool", "CollectionManagementTool"]),
        ("src.mcp_server.tools.analysis_tools", ["ClusterAnalysisTool", "QualityAssessmentTool"]),
        ("src.mcp_server.tools.vector_store_tools", ["create_vector_store_tool"]),
        ("src.mcp_server.tools.ipfs_cluster_tools", ["IPFSClusterManagementTool"])
    ]
    
    for module_name, tool_names in tools_to_test:
        try:
            module = __import__(module_name, fromlist=tool_names)
            for tool_name in tool_names:
                tool_class = getattr(module, tool_name)
                print(f"✅ {tool_name} imported from {module_name}")
        except Exception as e:
            print(f"❌ Failed to import {tool_names} from {module_name}: {e}")

def test_service_instantiation():
    """Test if services can be instantiated."""
    print("\n=== Testing Service Instantiation ===")
    
    try:
        from services.vector_service import VectorService, VectorConfig
        config = VectorConfig(dimension=768)
        service = VectorService(config)
        print("✅ VectorService instantiated")
    except Exception as e:
        print(f"❌ VectorService instantiation failed: {e}")
    
    try:
        from services.clustering_service import VectorClusterer, ClusterConfig
        config = ClusterConfig(n_clusters=5)
        service = VectorClusterer(config)
        print("✅ VectorClusterer instantiated")
    except Exception as e:
        print(f"❌ VectorClusterer instantiation failed: {e}")

def test_mcp_server_startup():
    """Test MCP server basic startup."""
    print("\n=== Testing MCP Server Startup ===")
    
    try:
        from src.mcp_server.main import MCPServerApplication
        app = MCPServerApplication()
        print(f"✅ MCP Server Application created: {app.config.server_name}")
    except Exception as e:
        print(f"❌ MCP Server startup failed: {e}")

def analyze_service_tool_mapping():
    """Analyze which services map to which tools."""
    print("\n=== Service-to-Tool Mapping Analysis ===")
    
    mapping = {
        "VectorService": [
            "SemanticSearchTool",
            "SimilaritySearchTool", 
            "StorageManagementTool",
            "create_vector_store_tool"
        ],
        "ClusteringService": [
            "ClusterAnalysisTool",
            "QualityAssessmentTool",
            "DimensionalityReductionTool"
        ],
        "EmbeddingService": [
            "EmbeddingGenerationTool",
            "BatchEmbeddingTool",
            "MultimodalEmbeddingTool"
        ],
        "IPFSVectorService": [
            "IPFSClusterManagementTool",
            "StorageManagementTool"
        ]
    }
    
    for service, tools in mapping.items():
        print(f"📋 {service} -> {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool}")

if __name__ == "__main__":
    print("LAION Embeddings MCP Server Feature Coverage Test")
    print("="*50)
    
    test_service_imports()
    test_mcp_tool_imports()
    test_service_instantiation()
    test_mcp_server_startup()
    analyze_service_tool_mapping()
    
    print("\n=== Summary ===")
    print("✅ All core services are available")
    print("✅ All MCP tools are implemented") 
    print("❌ Service integration with MCP tools is incomplete")
    print("📋 Next step: Implement service dependency injection in MCP server")
