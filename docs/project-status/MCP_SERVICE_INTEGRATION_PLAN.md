# Service Integration Implementation Plan for LAION Embeddings MCP Server

## Issue Summary

The LAION Embeddings MCP server has **complete tool coverage** but suffers from a **critical service integration gap**. All MCP tools are initialized with `None` instead of actual service instances, making them non-functional.

## Current State

```python
# In src/mcp_server/main.py line 125-159
# TODO: Pass actual embedding service
embedding_gen = EmbeddingGenerationTool(None)  
batch_embedding = BatchEmbeddingTool(None)
multimodal_embedding = MultimodalEmbeddingTool(None)
# ... all tools initialized with None
```

## Solution: Service Factory and Dependency Injection

### 1. Create Service Factory (`src/mcp_server/service_factory.py`)

```python
"""
Service Factory for MCP Server

Provides centralized service initialization and dependency injection.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from services.vector_service import VectorService, VectorConfig
from services.clustering_service import ClusteringService, ClusterConfig  
from services.embedding_service import EmbeddingService
from services.ipfs_vector_service import IPFSVectorService, IPFSConfig
from services.distributed_vector_service import DistributedVectorService

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfigs:
    """Consolidated service configurations."""
    vector_config: VectorConfig
    cluster_config: ClusterConfig
    ipfs_config: IPFSConfig
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class ServiceFactory:
    """Factory for creating and managing service instances."""
    
    def __init__(self, configs: ServiceConfigs):
        self.configs = configs
        self._services: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize_services(self) -> Dict[str, Any]:
        """Initialize all services."""
        if self._initialized:
            return self._services
        
        logger.info("Initializing core services...")
        
        # Initialize Vector Service
        logger.info("Initializing VectorService...")
        self._services['vector'] = VectorService(self.configs.vector_config)
        await self._services['vector'].initialize()
        
        # Initialize Clustering Service  
        logger.info("Initializing ClusteringService...")
        self._services['clustering'] = ClusteringService(self.configs.cluster_config)
        await self._services['clustering'].initialize()
        
        # Initialize Embedding Service
        logger.info("Initializing EmbeddingService...")
        self._services['embedding'] = EmbeddingService()
        await self._services['embedding'].initialize()
        
        # Initialize IPFS Service
        logger.info("Initializing IPFSVectorService...")
        self._services['ipfs'] = IPFSVectorService(self.configs.ipfs_config)
        await self._services['ipfs'].initialize()
        
        # Initialize Distributed Service
        logger.info("Initializing DistributedVectorService...")
        self._services['distributed'] = DistributedVectorService()
        await self._services['distributed'].initialize()
        
        self._initialized = True
        logger.info(f"All {len(self._services)} services initialized successfully")
        
        return self._services
    
    def get_service(self, service_name: str) -> Any:
        """Get a specific service instance."""
        if not self._initialized:
            raise RuntimeError("Services not initialized. Call initialize_services() first.")
        return self._services.get(service_name)
    
    async def shutdown_services(self):
        """Shutdown all services gracefully."""
        logger.info("Shutting down services...")
        for name, service in self._services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                logger.info(f"Service {name} shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down service {name}: {e}")
```

### 2. Update MCP Server Main (`src/mcp_server/main.py`)

```python
# Replace the _register_tools method with proper service integration

async def _register_tools(self):
    """Register all available tools with actual service instances."""
    if not self.tool_registry:
        raise InternalError("Tool registry not initialized")
    
    try:
        # Initialize services first
        logger.info("Initializing services for MCP tools...")
        service_configs = ServiceConfigs(
            vector_config=VectorConfig(dimension=768),
            cluster_config=ClusterConfig(n_clusters=10),
            ipfs_config=IPFSConfig()
        )
        
        self.service_factory = ServiceFactory(service_configs)
        services = await self.service_factory.initialize_services()
        
        # Embedding tools with actual embedding service
        logger.info("Registering embedding tools...")
        embedding_service = services['embedding']
        
        embedding_gen = EmbeddingGenerationTool(embedding_service)
        batch_embedding = BatchEmbeddingTool(embedding_service)
        multimodal_embedding = MultimodalEmbeddingTool(embedding_service)
        
        await self.tool_registry.register_tool(embedding_gen)
        await self.tool_registry.register_tool(batch_embedding)
        await self.tool_registry.register_tool(multimodal_embedding)
        
        # Search tools with vector service
        logger.info("Registering search tools...")
        vector_service = services['vector']
        
        semantic_search = SemanticSearchTool(vector_service)
        similarity_search = SimilaritySearchTool(vector_service)
        faceted_search = FacetedSearchTool(vector_service)
        
        await self.tool_registry.register_tool(semantic_search)
        await self.tool_registry.register_tool(similarity_search)
        await self.tool_registry.register_tool(faceted_search)
        
        # Storage tools with vector and IPFS services
        logger.info("Registering storage tools...")
        ipfs_service = services['ipfs']
        
        storage_mgmt = StorageManagementTool(vector_service, ipfs_service)
        collection_mgmt = CollectionManagementTool(vector_service)
        retrieval = RetrievalTool(vector_service, ipfs_service)
        
        await self.tool_registry.register_tool(storage_mgmt)
        await self.tool_registry.register_tool(collection_mgmt)
        await self.tool_registry.register_tool(retrieval)
        
        # Analysis tools with clustering service
        logger.info("Registering analysis tools...")
        clustering_service = services['clustering']
        
        cluster_analysis = ClusterAnalysisTool(clustering_service)
        quality_assessment = QualityAssessmentTool(clustering_service, vector_service)
        dimensionality_reduction = DimensionalityReductionTool(clustering_service)
        
        await self.tool_registry.register_tool(cluster_analysis)
        await self.tool_registry.register_tool(quality_assessment)
        await self.tool_registry.register_tool(dimensionality_reduction)
        
        logger.info(f"Registered {len(self.tool_registry.tools)} tools with actual services")
        
    except Exception as e:
        logger.error(f"Failed to register tools: {e}")
        raise InternalError(f"Tool registration failed: {e}")

async def _shutdown_components(self):
    """Shutdown all components gracefully."""
    logger.info("Shutting down components...")
    
    try:
        # Shutdown services first
        if hasattr(self, 'service_factory'):
            await self.service_factory.shutdown_services()
            logger.info("Services shutdown complete")
        
        # Shutdown MCP server
        if self.mcp_server:
            await self.mcp_server.shutdown()
            logger.info("MCP server shutdown complete")
        
        # ... rest of shutdown logic
```

### 3. Update Tool Constructors

All MCP tools need to be updated to accept and use actual service instances:

```python
# Example: src/mcp_server/tools/embedding_tools.py

class EmbeddingGenerationTool(ClaudeMCPTool):
    def __init__(self, embedding_service):
        super().__init__()
        self.name = "generate_embedding"
        self.description = "Generates an embedding vector for a given text using specified model."
        # ... schema definition ...
        self.embedding_service = embedding_service
        
        # Validate service
        if embedding_service is None:
            raise ValueError("EmbeddingService instance required")
    
    async def execute(self, text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                     normalize: bool = True) -> Dict[str, Any]:
        """Execute embedding generation with actual service."""
        try:
            # Validate inputs
            text = validator.validate_text_input(text)
            model = validator.validate_model_name(model)
            
            # Call actual service
            embedding = await self.embedding_service.generate_embedding(
                text=text,
                model=model,
                normalize=normalize
            )
            
            return {
                "text": text,
                "model": model,
                "embedding": embedding,
                "dimension": len(embedding),
                "normalized": normalize
            }
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
```

## Implementation Steps

### Phase 1: Core Service Integration (Immediate)

1. **Create `ServiceFactory`**
   - Implement service factory class
   - Add service configuration management
   - Add async service initialization

2. **Update MCP Server Main**
   - Replace tool registration with service-aware version
   - Add service factory integration
   - Update shutdown procedures

3. **Update Tool Constructors**
   - Update all tool classes to accept service instances
   - Add service validation
   - Remove mock implementations

### Phase 2: Configuration Integration (Short-term)

1. **Extend MCP Config**
   - Add service configuration options
   - Add environment variable support
   - Add configuration validation

2. **Add Service Health Checks**
   - Implement service health monitoring
   - Add service status reporting
   - Add failure recovery mechanisms

### Phase 3: Testing and Validation (Short-term)

1. **Integration Testing**
   - Test all service-to-tool connections
   - Validate MCP server functionality
   - Performance testing

2. **Error Handling**
   - Add comprehensive error propagation
   - Implement graceful degradation
   - Add detailed logging

## Expected Outcomes

After implementation:

1. **✅ Functional MCP Server**
   - All tools connected to actual services
   - Full LAION Embeddings functionality exposed via MCP
   - Production-ready deployment

2. **✅ Complete Feature Coverage**
   - Vector operations (search, storage, indexing)
   - Clustering and analysis
   - Embedding generation
   - IPFS distributed storage
   - Administrative operations

3. **✅ Robust Service Management**
   - Proper service lifecycle management
   - Health monitoring and recovery
   - Configuration-driven deployment

## Conclusion

The LAION Embeddings project has **complete MCP tool coverage** but requires **service integration implementation** to become functional. This is a straightforward implementation task that will unlock full MCP server capabilities for all LAION Embeddings features.
