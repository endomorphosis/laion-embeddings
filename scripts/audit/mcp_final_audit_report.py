#!/usr/bin/env python3
"""
MCP AUDIT FINAL REPORT
=====================
Manual analysis of FastAPI endpoints vs MCP tools based on the current system.
"""

# FastAPI Endpoints discovered
FASTAPI_ENDPOINTS = [
    {"method": "GET", "path": "/health", "function": "health_check", "category": "health"},
    {"method": "GET", "path": "/", "function": "root", "category": "health"},
    {"method": "POST", "path": "/add_endpoint", "function": "add_endpoint", "category": "admin"},
    {"method": "POST", "path": "/create_embeddings", "function": "create_embeddings", "category": "embedding"},
    {"method": "POST", "path": "/load", "function": "load_index", "category": "index_management"},
    {"method": "POST", "path": "/search", "function": "search", "category": "search"},
    {"method": "POST", "path": "/shard_embeddings", "function": "shard_embeddings", "category": "index_management"},
    {"method": "POST", "path": "/index_sparse_embeddings", "function": "index_sparse_embeddings", "category": "sparse_embedding"},
    {"method": "POST", "path": "/index_cluster", "function": "index_cluster", "category": "storage"},
    {"method": "POST", "path": "/storacha_clusters", "function": "storacha_clusters", "category": "storage"},
    {"method": "GET", "path": "/cache/stats", "function": "get_cache_stats", "category": "cache"},
    {"method": "POST", "path": "/cache/clear", "function": "clear_cache", "category": "cache"},
    {"method": "POST", "path": "/auth/login", "function": "login", "category": "auth"},
    {"method": "GET", "path": "/auth/me", "function": "get_current_user", "category": "auth"},
    {"method": "GET", "path": "/metrics", "function": "get_metrics", "category": "monitoring"},
    {"method": "GET", "path": "/metrics/json", "function": "get_metrics_json", "category": "monitoring"},
    {"method": "GET", "path": "/health/detailed", "function": "detailed_health", "category": "health"}
]

# MCP Tools currently registered in main.py
REGISTERED_MCP_TOOLS = [
    # Embedding tools
    {"name": "EmbeddingGenerationTool", "category": "embedding", "module": "embedding_tools"},
    {"name": "BatchEmbeddingTool", "category": "embedding", "module": "embedding_tools"},
    {"name": "MultimodalEmbeddingTool", "category": "embedding", "module": "embedding_tools"},
    
    # Search tools
    {"name": "SemanticSearchTool", "category": "search", "module": "search_tools"},
    {"name": "SimilaritySearchTool", "category": "search", "module": "search_tools"},
    {"name": "FacetedSearchTool", "category": "search", "module": "search_tools"},
    
    # Storage tools
    {"name": "StorageManagementTool", "category": "storage", "module": "storage_tools"},
    {"name": "CollectionManagementTool", "category": "storage", "module": "storage_tools"},
    {"name": "RetrievalTool", "category": "storage", "module": "storage_tools"},
    
    # Analysis tools
    {"name": "ClusterAnalysisTool", "category": "analysis", "module": "analysis_tools"},
    {"name": "QualityAssessmentTool", "category": "analysis", "module": "analysis_tools"},
    {"name": "DimensionalityReductionTool", "category": "analysis", "module": "analysis_tools"},
    
    # Vector store tools
    {"name": "VectorIndexTool", "category": "vector_store", "module": "vector_store_tools"},
    {"name": "VectorRetrievalTool", "category": "vector_store", "module": "vector_store_tools"},
    {"name": "VectorMetadataTool", "category": "vector_store", "module": "vector_store_tools"},
    
    # IPFS cluster tools
    {"name": "IPFSClusterTool", "category": "storage", "module": "ipfs_cluster_tools"},
    {"name": "DistributedVectorTool", "category": "storage", "module": "ipfs_cluster_tools"},
    {"name": "IPFSMetadataTool", "category": "storage", "module": "ipfs_cluster_tools"},
]

# Additional MCP Tools available but not registered
AVAILABLE_UNREGISTERED_TOOLS = [
    # Sparse embedding tools
    {"name": "SparseEmbeddingGenerationTool", "category": "sparse_embedding", "module": "sparse_embedding_tools"},
    {"name": "SparseIndexingTool", "category": "sparse_embedding", "module": "sparse_embedding_tools"},
    {"name": "SparseSearchTool", "category": "sparse_embedding", "module": "sparse_embedding_tools"},
    
    # Authentication tools
    {"name": "AuthenticationTool", "category": "auth", "module": "auth_tools"},
    {"name": "UserInfoTool", "category": "auth", "module": "auth_tools"},
    {"name": "TokenValidationTool", "category": "auth", "module": "auth_tools"},
    
    # Cache tools
    {"name": "CacheStatsTool", "category": "cache", "module": "cache_tools"},
    {"name": "CacheManagementTool", "category": "cache", "module": "cache_tools"},
    {"name": "CacheMonitoringTool", "category": "cache", "module": "cache_tools"},
    
    # Monitoring tools
    {"name": "HealthCheckTool", "category": "monitoring", "module": "monitoring_tools"},
    {"name": "MetricsCollectionTool", "category": "monitoring", "module": "monitoring_tools"},
    {"name": "SystemMonitoringTool", "category": "monitoring", "module": "monitoring_tools"},
    {"name": "AlertManagementTool", "category": "monitoring", "module": "monitoring_tools"},
    
    # Admin tools
    {"name": "EndpointManagementTool", "category": "admin", "module": "admin_tools"},
    {"name": "UserManagementTool", "category": "admin", "module": "admin_tools"},
    {"name": "SystemConfigTool", "category": "admin", "module": "admin_tools"},
    
    # Index management tools
    {"name": "IndexLoadingTool", "category": "index_management", "module": "index_management_tools"},
    {"name": "ShardManagementTool", "category": "index_management", "module": "index_management_tools"},
    
    # Session management tools
    {"name": "SessionCreationTool", "category": "session", "module": "session_management_tools"},
    {"name": "SessionMonitoringTool", "category": "session", "module": "session_management_tools"},
    {"name": "SessionCleanupTool", "category": "session", "module": "session_management_tools"},
    
    # Workflow tools
    {"name": "BackgroundTaskStatusTool", "category": "workflow", "module": "workflow_tools"},
    {"name": "BackgroundTaskManagementTool", "category": "workflow", "module": "workflow_tools"},
    {"name": "TaskQueueManagementTool", "category": "workflow", "module": "workflow_tools"},
    {"name": "RateLimitConfigurationTool", "category": "workflow", "module": "workflow_tools"},
    {"name": "RateLimitMonitoringTool", "category": "workflow", "module": "workflow_tools"},
    {"name": "RateLimitManagementTool", "category": "workflow", "module": "workflow_tools"},
]

def analyze_coverage():
    """Analyze endpoint to tool coverage"""
    
    print("🚀 MCP COMPREHENSIVE AUDIT - FINAL REPORT")
    print("=" * 60)
    
    # Count by category
    endpoint_categories = {}
    registered_categories = {}
    available_categories = {}
    
    for ep in FASTAPI_ENDPOINTS:
        cat = ep['category']
        endpoint_categories[cat] = endpoint_categories.get(cat, 0) + 1
    
    for tool in REGISTERED_MCP_TOOLS:
        cat = tool['category']
        registered_categories[cat] = registered_categories.get(cat, 0) + 1
    
    for tool in AVAILABLE_UNREGISTERED_TOOLS:
        cat = tool['category']
        available_categories[cat] = available_categories.get(cat, 0) + 1
    
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"  Total FastAPI Endpoints: {len(FASTAPI_ENDPOINTS)}")
    print(f"  Registered MCP Tools:    {len(REGISTERED_MCP_TOOLS)}")
    print(f"  Available MCP Tools:     {len(AVAILABLE_UNREGISTERED_TOOLS)}")
    print(f"  Total MCP Tools:         {len(REGISTERED_MCP_TOOLS) + len(AVAILABLE_UNREGISTERED_TOOLS)}")
    
    print(f"\n📋 COVERAGE BY CATEGORY")
    all_categories = set(endpoint_categories.keys()) | set(registered_categories.keys()) | set(available_categories.keys())
    
    for category in sorted(all_categories):
        ep_count = endpoint_categories.get(category, 0)
        reg_count = registered_categories.get(category, 0)
        avail_count = available_categories.get(category, 0)
        total_tools = reg_count + avail_count
        
        status = "✅" if reg_count > 0 else ("🔶" if avail_count > 0 else "❌")
        print(f"  {status} {category:18} | Endpoints:{ep_count:2} | Registered:{reg_count:2} | Available:{avail_count:2} | Total:{total_tools:2}")
    
    print(f"\n🔍 DETAILED ENDPOINT ANALYSIS")
    for ep in FASTAPI_ENDPOINTS:
        reg_tools = [t for t in REGISTERED_MCP_TOOLS if t['category'] == ep['category']]
        avail_tools = [t for t in AVAILABLE_UNREGISTERED_TOOLS if t['category'] == ep['category']]
        
        status = "✅" if reg_tools else ("🔶" if avail_tools else "❌")
        print(f"  {status} {ep['method']:4} {ep['path']:25} [{ep['category']:15}] -> {ep['function']}")
    
    print(f"\n⚠️  GAPS IDENTIFIED")
    
    # Find categories with endpoints but no registered tools
    uncovered = []
    unregistered = []
    
    for category in endpoint_categories:
        if registered_categories.get(category, 0) == 0:
            if available_categories.get(category, 0) > 0:
                unregistered.append(category)
            else:
                uncovered.append(category)
    
    if uncovered:
        print(f"  ❌ Missing Tools (no tools available): {', '.join(uncovered)}")
    
    if unregistered:
        print(f"  🔶 Unregistered Tools (tools exist but not registered): {', '.join(unregistered)}")
    
    print(f"\n💡 RECOMMENDATIONS")
    
    if unregistered:
        print(f"  1. IMMEDIATE: Register existing tools for: {', '.join(unregistered)}")
        print(f"     - Add imports and registration calls in src/mcp_server/main.py")
    
    if uncovered:
        print(f"  2. DEVELOPMENT: Create missing tools for: {', '.join(uncovered)}")
    
    print(f"  3. TESTING: Run comprehensive tests for all MCP tools")
    print(f"  4. DOCUMENTATION: Update MCP tool documentation")
    
    # Calculate coverage scores
    total_categories = len(endpoint_categories)
    registered_coverage = len([cat for cat in endpoint_categories if registered_categories.get(cat, 0) > 0])
    available_coverage = len([cat for cat in endpoint_categories if (registered_categories.get(cat, 0) + available_categories.get(cat, 0)) > 0])
    
    reg_score = (registered_coverage / total_categories * 100) if total_categories > 0 else 0
    avail_score = (available_coverage / total_categories * 100) if total_categories > 0 else 0
    
    print(f"\n🎯 COVERAGE SCORES")
    print(f"  Registered Tool Coverage: {reg_score:.1f}% ({registered_coverage}/{total_categories} categories)")
    print(f"  Available Tool Coverage:  {avail_score:.1f}% ({available_coverage}/{total_categories} categories)")
    
    if reg_score >= 80:
        print("  ✅ EXCELLENT registered coverage!")
    elif reg_score >= 60:
        print("  🔶 GOOD registered coverage")
    else:
        print("  ❌ LOW registered coverage - action needed")
    
    if avail_score >= 90:
        print("  ✅ COMPREHENSIVE tool availability!")
    elif avail_score >= 70:
        print("  🔶 GOOD tool availability")
    else:
        print("  ❌ MISSING tools need development")
    
    print(f"\n📝 NEXT STEPS")
    print(f"  1. Register {len(AVAILABLE_UNREGISTERED_TOOLS)} available tools")
    print(f"  2. Fix any import/test issues")
    print(f"  3. Run comprehensive test suite")
    print(f"  4. Validate end-to-end functionality")
    
    return {
        'registered_coverage': reg_score,
        'available_coverage': avail_score,
        'total_endpoints': len(FASTAPI_ENDPOINTS),
        'registered_tools': len(REGISTERED_MCP_TOOLS),
        'available_tools': len(AVAILABLE_UNREGISTERED_TOOLS),
        'uncovered_categories': uncovered,
        'unregistered_categories': unregistered
    }

if __name__ == "__main__":
    results = analyze_coverage()
    
    # Save results
    import json
    with open("MCP_FINAL_AUDIT_RESULTS.json", "w") as f:
        json.dump({
            'audit_results': results,
            'endpoints': FASTAPI_ENDPOINTS,
            'registered_tools': REGISTERED_MCP_TOOLS,
            'available_tools': AVAILABLE_UNREGISTERED_TOOLS
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: MCP_FINAL_AUDIT_RESULTS.json")
