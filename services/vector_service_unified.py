"""
Unified Vector Service

This module provides a high-level, unified API for vector operations across all
supported vector databases. It handles embedding generation, search, and management
operations with automatic fallback and load balancing.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

from .vector_config import VectorDBType, get_config_manager
from .vector_store_base import (
    BaseVectorStore, VectorDocument, SearchResult, SearchQuery,
    IndexStats, HealthStatus, VectorStoreStatus
)
from .vector_store_factory import get_vector_store_factory, VectorStoreFactory
from .embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class VectorService:
    """
    Unified vector service providing high-level operations across all vector stores.
    
    Features:
    - Automatic embedding generation
    - Multi-database search with fallback
    - Load balancing across instances
    - Health monitoring
    - Batch operations
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                 factory: Optional[VectorStoreFactory] = None,
                 embedding_service: Optional[EmbeddingService] = None,
                 enable_fallback: bool = True,
                 enable_load_balancing: bool = True):
        """
        Initialize vector service.
        
        Args:
            factory: Vector store factory instance
            embedding_service: Embedding service instance
            enable_fallback: Enable automatic fallback to other databases
            enable_load_balancing: Enable load balancing across instances
        """
        self.factory = factory or get_vector_store_factory()
        self.config_manager = self.factory.config_manager
        self.embedding_service = embedding_service or EmbeddingService()
        self.enable_fallback = enable_fallback
        self.enable_load_balancing = enable_load_balancing
        
        # Store instances and health status
        self._stores: Dict[VectorDBType, BaseVectorStore] = {}
        self._health_status: Dict[VectorDBType, HealthStatus] = {}
        self._last_health_check = None
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the vector service and connect to databases."""
        if self._is_initialized:
            return
        
        try:
            # Create store instances for all enabled databases
            self._stores = await self.factory.create_all_enabled_stores()
            
            # Connect to all stores
            connection_tasks = []
            for db_type, store in self._stores.items():
                connection_tasks.append(self._connect_store(db_type, store))
            
            await asyncio.gather(*connection_tasks, return_exceptions=True)
            
            # Initialize embedding service
            await self.embedding_service.initialize()
            
            # Perform initial health check
            await self.check_health()
            
            self._is_initialized = True
            logger.info(f"Vector service initialized with {len(self._stores)} stores")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector service: {e}")
            raise
    
    async def _connect_store(self, db_type: VectorDBType, store: BaseVectorStore) -> None:
        """Connect to a single store."""
        try:
            await store.connect()
            logger.info(f"Connected to {db_type.value} vector store")
        except Exception as e:
            logger.error(f"Failed to connect to {db_type.value}: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the vector service and disconnect from databases."""
        disconnect_tasks = []
        for store in self._stores.values():
            if store.is_connected:
                disconnect_tasks.append(store.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        await self.embedding_service.cleanup()
        self._stores.clear()
        self._health_status.clear()
        self._is_initialized = False
        logger.info("Vector service shutdown complete")
    
    # Health and Status
    
    async def check_health(self) -> Dict[VectorDBType, HealthStatus]:
        """
        Check health of all vector stores.
        
        Returns:
            Dictionary mapping database types to health status
        """
        health_tasks = []
        db_types = []
        
        for db_type, store in self._stores.items():
            if store.is_connected:
                health_tasks.append(store.get_health_status())
                db_types.append(db_type)
        
        if health_tasks:
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            for i, db_type in enumerate(db_types):
                result = health_results[i]
                if isinstance(result, HealthStatus):
                    self._health_status[db_type] = result
                else:
                    # Create error status
                    self._health_status[db_type] = HealthStatus(
                        status=VectorStoreStatus.UNHEALTHY,
                        message=f"Health check failed: {result}",
                        last_check=datetime.utcnow().isoformat()
                    )
        
        self._last_health_check = datetime.utcnow()
        return self._health_status.copy()
    
    def get_healthy_stores(self) -> List[VectorDBType]:
        """Get list of healthy vector stores."""
        healthy = []
        for db_type, status in self._health_status.items():
            if status.status == VectorStoreStatus.HEALTHY:
                healthy.append(db_type)
        return healthy
    
    def get_primary_store(self) -> Optional[BaseVectorStore]:
        """Get the primary (default) vector store if healthy."""
        default_db = self.config_manager.get_default_database()
        if (default_db in self._stores and 
            default_db in self._health_status and
            self._health_status[default_db].status == VectorStoreStatus.HEALTHY):
            return self._stores[default_db]
        return None
    
    def get_fallback_stores(self) -> List[BaseVectorStore]:
        """Get list of healthy fallback stores."""
        if not self.enable_fallback:
            return []
        
        healthy_types = self.get_healthy_stores()
        default_db = self.config_manager.get_default_database()
        
        # Return healthy stores excluding the primary
        fallback_stores = []
        for db_type in healthy_types:
            if db_type != default_db and db_type in self._stores:
                fallback_stores.append(self._stores[db_type])
        
        return fallback_stores
    
    # Document Operations
    
    async def add_documents(self, 
                          texts: List[str],
                          metadata: Optional[List[Dict[str, Any]]] = None,
                          ids: Optional[List[str]] = None,
                          index_name: Optional[str] = None,
                          db_type: Optional[VectorDBType] = None) -> bool:
        """
        Add documents with automatic embedding generation.
        
        Args:
            texts: List of text documents
            metadata: Optional metadata for each document
            ids: Optional IDs for each document
            index_name: Target index name
            db_type: Target database type (uses primary if None)
            
        Returns:
            True if documents added successfully
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Generate embeddings
        embeddings = await self.embedding_service.embed_documents(texts)
        
        # Create vector documents
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{i}_{datetime.utcnow().timestamp()}"
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            
            documents.append(VectorDocument(
                id=doc_id,
                vector=embedding,
                text=text,
                metadata=doc_metadata,
                timestamp=datetime.utcnow().isoformat()
            ))
        
        return await self.add_vectors(documents, index_name, db_type)
    
    async def add_vectors(self,
                         documents: List[VectorDocument],
                         index_name: Optional[str] = None,
                         db_type: Optional[VectorDBType] = None) -> bool:
        """
        Add pre-computed vectors.
        
        Args:
            documents: List of vector documents
            index_name: Target index name
            db_type: Target database type (uses primary if None)
            
        Returns:
            True if vectors added successfully
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Determine target store
        target_store = await self._get_target_store(db_type, "write")
        if not target_store:
            raise RuntimeError("No available vector store for write operations")
        
        try:
            return await target_store.add_vectors(documents, index_name)
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            
            # Try fallback stores
            if self.enable_fallback and db_type is None:
                for fallback_store in self.get_fallback_stores():
                    try:
                        return await fallback_store.add_vectors(documents, index_name)
                    except Exception as fallback_error:
                        logger.warning(f"Fallback store also failed: {fallback_error}")
                        continue
            
            return False
    
    # Search Operations
    
    async def search_text(self,
                         query: str,
                         limit: int = 10,
                         similarity_threshold: Optional[float] = None,
                         filter_expr: Optional[Dict[str, Any]] = None,
                         index_name: Optional[str] = None,
                         db_type: Optional[VectorDBType] = None) -> List[SearchResult]:
        """
        Search using text query with automatic embedding.
        
        Args:
            query: Text query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            filter_expr: Optional filter expression
            index_name: Target index name
            db_type: Target database type (searches all if None)
            
        Returns:
            List of search results
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_query(query)
        
        return await self.search_vectors(
            query_embedding, limit, similarity_threshold, 
            filter_expr, index_name, db_type
        )
    
    async def search_vectors(self,
                           query_vector: List[float],
                           limit: int = 10,
                           similarity_threshold: Optional[float] = None,
                           filter_expr: Optional[Dict[str, Any]] = None,
                           index_name: Optional[str] = None,
                           db_type: Optional[VectorDBType] = None) -> List[SearchResult]:
        """
        Search using pre-computed vector.
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            filter_expr: Optional filter expression
            index_name: Target index name
            db_type: Target database type (searches all if None)
            
        Returns:
            List of search results
        """
        if not self._is_initialized:
            await self.initialize()
        
        search_query = SearchQuery(
            vector=query_vector,
            limit=limit,
            filter_expr=filter_expr,
            similarity_threshold=similarity_threshold,
            include_metadata=True
        )
        
        # Determine target store(s)
        if db_type:
            target_stores = [self._stores[db_type]] if db_type in self._stores else []
        else:
            # Search using primary store, fall back to others if needed
            target_stores = [self.get_primary_store()]
            if self.enable_fallback:
                target_stores.extend(self.get_fallback_stores())
            target_stores = [store for store in target_stores if store is not None]
        
        if not target_stores:
            return []
        
        # Try each store until we get results
        for store in target_stores:
            try:
                results = await store.search(search_query, index_name)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Search failed on store: {e}")
                continue
        
        return []
    
    async def hybrid_search(self,
                          query: str,
                          vector_weight: float = 0.7,
                          text_weight: float = 0.3,
                          limit: int = 10,
                          index_name: Optional[str] = None,
                          db_type: Optional[VectorDBType] = None) -> List[SearchResult]:
        """
        Perform hybrid vector + text search.
        
        Args:
            query: Text query
            vector_weight: Weight for vector similarity
            text_weight: Weight for text similarity
            limit: Maximum number of results
            index_name: Target index name
            db_type: Target database type (uses primary if None)
            
        Returns:
            List of search results
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_query(query)
        
        # Determine target store
        target_store = await self._get_target_store(db_type, "read")
        if not target_store:
            # Fall back to regular vector search
            return await self.search_text(query, limit, index_name=index_name)
        
        try:
            return await target_store.hybrid_search(
                query_embedding, query, vector_weight, text_weight, limit, index_name
            )
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
            # Fall back to vector search
            return await self.search_vectors(query_embedding, limit, index_name=index_name)
    
    async def batch_search(self,
                          queries: List[str],
                          limit: int = 10,
                          index_name: Optional[str] = None,
                          db_type: Optional[VectorDBType] = None) -> List[List[SearchResult]]:
        """
        Perform batch search operations.
        
        Args:
            queries: List of text queries
            limit: Maximum number of results per query
            index_name: Target index name
            db_type: Target database type (uses primary if None)
            
        Returns:
            List of result lists (one per query)
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Generate embeddings for all queries
        query_embeddings = await self.embedding_service.embed_documents(queries)
        
        # Create search queries
        search_queries = []
        for embedding in query_embeddings:
            search_queries.append(SearchQuery(
                vector=embedding,
                limit=limit,
                include_metadata=True
            ))
        
        # Determine target store
        target_store = await self._get_target_store(db_type, "read")
        if not target_store:
            # Fall back to individual searches
            results = []
            for query in queries:
                result = await self.search_text(query, limit, index_name=index_name)
                results.append(result)
            return results
        
        try:
            return await target_store.batch_search(search_queries, index_name)
        except Exception as e:
            logger.warning(f"Batch search failed: {e}")
            # Fall back to individual searches
            results = []
            for query in queries:
                result = await self.search_text(query, limit, index_name=index_name)
                results.append(result)
            return results
    
    # Index Management
    
    async def create_index(self,
                          index_name: str,
                          dimension: Optional[int] = None,
                          distance_metric: str = "cosine",
                          db_type: Optional[VectorDBType] = None) -> bool:
        """
        Create a new index.
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension (uses default if None)
            distance_metric: Distance metric
            db_type: Target database type (uses primary if None)
            
        Returns:
            True if index created successfully
        """
        if not self._is_initialized:
            await self.initialize()
        
        if dimension is None:
            dimension = self.config_manager.get_embedding_config().get('dimension', 512)
        
        target_store = await self._get_target_store(db_type, "write")
        if not target_store:
            return False
        
        try:
            return await target_store.create_index(index_name, dimension, distance_metric)
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    async def get_index_stats(self,
                            index_name: Optional[str] = None,
                            db_type: Optional[VectorDBType] = None) -> Optional[IndexStats]:
        """
        Get index statistics.
        
        Args:
            index_name: Target index name
            db_type: Target database type (uses primary if None)
            
        Returns:
            Index statistics or None if not available
        """
        if not self._is_initialized:
            await self.initialize()
        
        target_store = await self._get_target_store(db_type, "read")
        if not target_store:
            return None
        
        try:
            return await target_store.get_index_stats(index_name)
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return None
    
    # Utility Methods
    
    async def _get_target_store(self, 
                              db_type: Optional[VectorDBType],
                              operation_type: str) -> Optional[BaseVectorStore]:
        """Get target store for an operation."""
        if db_type:
            return self._stores.get(db_type)
        
        # Use primary store if healthy
        primary = self.get_primary_store()
        if primary:
            return primary
        
        # Use any healthy store
        healthy_stores = self.get_healthy_stores()
        if healthy_stores:
            return self._stores[healthy_stores[0]]
        
        return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get overall service status."""
        return {
            'initialized': self._is_initialized,
            'stores_connected': len([s for s in self._stores.values() if s.is_connected]),
            'total_stores': len(self._stores),
            'healthy_stores': len(self.get_healthy_stores()),
            'primary_store': self.config_manager.get_default_database().value,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'embedding_service_ready': self.embedding_service.is_ready()
        }
    
    # Context Manager Support
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
