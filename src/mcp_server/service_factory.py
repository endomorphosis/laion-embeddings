"""
Service Factory for MCP Server

Provides centralized service initialization and dependency injection.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from services.vector_service import VectorService, VectorConfig
from services.clustering_service import VectorClusterer, ClusterConfig  
from services.embedding_service import EmbeddingService
from services.ipfs_vector_service import IPFSVectorService, IPFSConfig
from services.distributed_vector_service import DistributedVectorIndex

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
    
    def __init__(self, config):
        """Initialize with either MCPConfig or ServiceConfigs."""
        if hasattr(config, 'vector_config'):
            # Already ServiceConfigs
            self.configs = config
        else:
            # Convert MCPConfig to ServiceConfigs
            self.configs = self._create_service_configs_from_mcp_config(config)
        self._services: Dict[str, Any] = {}
        self._initialized = False
    
    def _create_service_configs_from_mcp_config(self, mcp_config) -> ServiceConfigs:
        """Create ServiceConfigs from MCPConfig."""
        return ServiceConfigs(
            vector_config=VectorConfig(
                dimension=getattr(mcp_config, 'vector_dimension', 768),
                metric="L2",
                index_type="IVF",
                nlist=100,
                nprobe=10
            ),
            cluster_config=ClusterConfig(
                algorithm="kmeans",
                n_clusters=getattr(mcp_config, 'n_clusters', 10),
                random_state=42
            ),
            ipfs_config=IPFSConfig(
                api_url='/ip4/127.0.0.1/tcp/5001',
                gateway_url='http://127.0.0.1:8080',
                timeout=60,
                chunk_size=1000,
                compression=True,
                pin_content=True
            ),
            embedding_provider=getattr(mcp_config, 'embedding_provider', "sentence-transformers"),
            embedding_model=getattr(mcp_config, 'embedding_model', "sentence-transformers/all-MiniLM-L6-v2")
        )
    
    async def initialize_services(self) -> Dict[str, Any]:
        """Initialize all services."""
        if self._initialized:
            return self._services
        
        logger.info("Initializing core services...")
        
        try:
            # Initialize Vector Service
            logger.info("Initializing VectorService...")
            self._services['vector'] = VectorService(self.configs.vector_config)
            if hasattr(self._services['vector'], 'initialize'):
                await self._services['vector'].initialize()
            
            # Initialize Clustering Service  
            logger.info("Initializing VectorClusterer...")
            self._services['clustering'] = VectorClusterer(self.configs.cluster_config)
            if hasattr(self._services['clustering'], 'initialize'):
                await self._services['clustering'].initialize()
            
            # Initialize Embedding Service
            logger.info("Initializing EmbeddingService...")
            self._services['embedding'] = EmbeddingService()
            if hasattr(self._services['embedding'], 'initialize'):
                await self._services['embedding'].initialize()
            
            # Initialize IPFS Service (optional)
            logger.info("Initializing IPFSVectorService...")
            try:
                self._services['ipfs'] = IPFSVectorService(self.configs.vector_config, self.configs.ipfs_config)
                if hasattr(self._services['ipfs'], 'initialize'):
                    await self._services['ipfs'].initialize()
            except ImportError as e:
                logger.warning(f"IPFS service not available: {e}")
                self._services['ipfs'] = None
            
            # Initialize Distributed Vector Service (optional)
            logger.info("Initializing DistributedVectorIndex...")
            try:
                self._services['distributed'] = DistributedVectorIndex()
                if hasattr(self._services['distributed'], 'initialize'):
                    await self._services['distributed'].initialize()
            except Exception as e:
                logger.warning(f"Distributed vector service not available: {e}")
                self._services['distributed'] = None
            
            self._initialized = True
            logger.info(f"All {len(self._services)} services initialized successfully")
            
            return self._services
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            # Cleanup any partially initialized services
            await self.shutdown_services()
            raise
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all services (alias for initialize_services)."""
        return await self.initialize_services()
    
    def get_service(self, service_name: str) -> Any:
        """Get a specific service instance."""
        if not self._initialized:
            raise RuntimeError("Services not initialized. Call initialize_services() first.")
        return self._services.get(service_name)
    
    def get_embedding_service(self) -> EmbeddingService:
        """Get the embedding service instance."""
        return self.get_service('embedding')
    
    def get_vector_service(self) -> VectorService:
        """Get the vector service instance."""
        return self.get_service('vector')
    
    def get_clustering_service(self) -> VectorClusterer:
        """Get the clustering service instance."""
        return self.get_service('clustering')
    
    def get_ipfs_vector_service(self) -> IPFSVectorService:
        """Get the IPFS vector service instance."""
        return self.get_service('ipfs')
    
    def get_distributed_vector_service(self) -> DistributedVectorIndex:
        """Get the distributed vector service instance."""
        return self.get_service('distributed')
    
    async def shutdown_services(self):
        """Shutdown all services gracefully."""
        logger.info("Shutting down services...")
        for name, service in self._services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                elif hasattr(service, 'cleanup'):
                    await service.cleanup()
                logger.info(f"Service {name} shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down service {name}: {e}")
        
        self._services.clear()
        self._initialized = False

    async def shutdown(self):
        """Shutdown all services (alias for shutdown_services)."""
        await self.shutdown_services()

def create_default_service_configs() -> ServiceConfigs:
    """Create default service configurations."""
    return ServiceConfigs(
        vector_config=VectorConfig(
            dimension=768,
            metric="L2",
            index_type="IVF",
            nlist=100,
            nprobe=10
        ),
        cluster_config=ClusterConfig(
            algorithm="kmeans",
            n_clusters=10,
            random_state=42
        ),
        ipfs_config=IPFSConfig(
            api_url='/ip4/127.0.0.1/tcp/5001',
            gateway_url='http://127.0.0.1:8080',
            timeout=60,
            chunk_size=1000,
            compression=True,
            pin_content=True
        )
    )
