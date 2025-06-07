"""
This module provides a factory pattern for creating and managing vector store instances
across all supported databases (Qdrant, Elasticsearch, pgvector, FAISS).
"""

import asyncio
from typing import Dict, Any, Optional, List, Type
import logging

from .vector_config import VectorDBType, get_config_manager, VectorDatabaseConfigManager
from .vector_store_base import BaseVectorStore, VectorStoreError, ConnectionError


logger = logging.getLogger(__name__)


class VectorStoreRegistry:
    """Registry for vector store implementations."""
    
    def __init__(self):
        self._stores: Dict[VectorDBType, Type[BaseVectorStore]] = {}
        self._instances: Dict[str, BaseVectorStore] = {}
    
    def register(self, db_type: VectorDBType, store_class: Type[BaseVectorStore]) -> None:
        """
        Register a vector store implementation.
        
        Args:
            db_type: Database type
            store_class: Vector store implementation class
        """
        self._stores[db_type] = store_class
        logger.info(f"Registered vector store for {db_type.value}")
    
    def get_store_class(self, db_type: VectorDBType) -> Optional[Type[BaseVectorStore]]:
        """
        Get vector store class for database type.
        
        Args:
            db_type: Database type
            
        Returns:
            Vector store class or None if not registered
        """
        return self._stores.get(db_type)
    
    def is_registered(self, db_type: VectorDBType) -> bool:
        """Check if database type is registered."""
        return db_type in self._stores
    
    def get_registered_types(self) -> List[VectorDBType]:
        """Get list of registered database types."""
        return list(self._stores.keys())
    
    def get_instance_key(self, db_type: VectorDBType, config: Dict[str, Any]) -> str:
        """Generate unique key for store instance."""
        # Create a simple hash of key configuration parameters
        key_params = {
            'type': db_type.value,
            'host': config.get('host', ''),
            'port': config.get('port', ''),
            'index': config.get('index_name', config.get('collection_name', '')),
        }
        return str(hash(frozenset(key_params.items())))
    
    def add_instance(self, key: str, instance: BaseVectorStore) -> None:
        """Add vector store instance to registry."""
        self._instances[key] = instance
    
    def get_instance(self, key: str) -> Optional[BaseVectorStore]:
        """Get vector store instance from registry."""
        return self._instances.get(key)
    
    def remove_instance(self, key: str) -> None:
        """Remove vector store instance from registry."""
        if key in self._instances:
            del self._instances[key]
    
    def clear_instances(self) -> None:
        """Clear all instances."""
        self._instances.clear()


# Global registry instance
_registry = VectorStoreRegistry()


def register_vector_store(db_type: VectorDBType, store_class: Type[BaseVectorStore]) -> None:
    """
    Register a vector store implementation.
    
    Args:
        db_type: Database type
        store_class: Vector store implementation class
    """
    _registry.register(db_type, store_class)


class VectorStoreFactory:
    """Factory for creating and managing vector store instances."""
    
    def __init__(self, config_manager: Optional[VectorDatabaseConfigManager] = None):
        """
        Initialize factory.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or get_config_manager()
        self.registry = _registry
        self._auto_register_stores()
    
    def _auto_register_stores(self) -> None:
        """Auto-register available vector store implementations."""
        # Try to import and register each vector store implementation
        self._try_register_qdrant()
        self._try_register_elasticsearch()
        self._try_register_pgvector()
        self._try_register_faiss()
        self._try_register_ipfs()
        self._try_register_duckdb()
    
    def _try_register_qdrant(self) -> None:
        """Try to register Qdrant vector store."""
        try:
            from .providers.qdrant_store import QdrantVectorStore
            self.registry.register(VectorDBType.QDRANT, QdrantVectorStore)
        except ImportError as e:
            logger.warning(f"Could not register Qdrant store: {e}")
    
    def _try_register_elasticsearch(self) -> None:
        """Try to register Elasticsearch vector store."""
        try:
            from .providers.elasticsearch_store import ElasticsearchVectorStore
            self.registry.register(VectorDBType.ELASTICSEARCH, ElasticsearchVectorStore)
        except ImportError as e:
            logger.warning(f"Could not register Elasticsearch store: {e}")
    
    def _try_register_pgvector(self) -> None:
        """Try to register pgvector store."""
        try:
            from .providers.pgvector_store import PgVectorStore
            self.registry.register(VectorDBType.PGVECTOR, PgVectorStore)
        except ImportError as e:
            logger.warning(f"Could not register pgvector store: {e}")
    
    def _try_register_faiss(self) -> None:
        """Try to register FAISS vector store."""
        try:
            from .providers.faiss_store import FAISSVectorStore
            self.registry.register(VectorDBType.FAISS, FAISSVectorStore)
        except ImportError as e:
            logger.warning(f"Could not register FAISS store: {e}")
    
    def _try_register_ipfs(self) -> None:
        """Try to register IPFS vector store."""
        try:
            from .providers.ipfs_store import IPFSVectorStore, IPFS_KIT_AVAILABLE, IPFS_CLIENT_AVAILABLE
            if not (IPFS_KIT_AVAILABLE or IPFS_CLIENT_AVAILABLE):
                logger.warning("Could not register IPFS store: Neither ipfs_kit_py nor ipfshttpclient is available.")
                return
            self.registry.register(VectorDBType.IPFS, IPFSVectorStore)
        except ImportError as e:
            logger.warning(f"Could not register IPFS store: {e}")
    
    def _try_register_duckdb(self) -> None:
        """Try to register DuckDB vector store."""
        try:
            from .providers.duckdb_store import DuckDBVectorStore, DUCKDB_AVAILABLE, ARROW_AVAILABLE
            if not (DUCKDB_AVAILABLE and ARROW_AVAILABLE):
                logger.warning("Could not register DuckDB store: duckdb or pyarrow is not available.")
                return
            self.registry.register(VectorDBType.DUCKDB, DuckDBVectorStore)
        except ImportError as e:
            logger.warning(f"Could not register DuckDB store: {e}")
    
    async def create_store(self, db_type: Optional[VectorDBType] = None,
                          config_override: Optional[Dict[str, Any]] = None,
                          reuse_instance: bool = True) -> BaseVectorStore:
        """
        Create a vector store instance.
        
        Args:
            db_type: Database type (uses default if None)
            config_override: Configuration overrides
            reuse_instance: Whether to reuse existing instances
            
        Returns:
            Vector store instance
            
        Raises:
            VectorStoreError: If store cannot be created
        """
        # Determine database type
        if db_type is None:
            db_type = self.config_manager.get_default_database()
        
        # Check if database is enabled
        if not self.config_manager.is_database_enabled(db_type):
            raise VectorStoreError(f"Database {db_type.value} is not enabled")
        
        # Get store class
        store_class = self.registry.get_store_class(db_type)
        if store_class is None:
            raise VectorStoreError(f"No implementation registered for {db_type.value}")
        
        # Get configuration
        db_config = self.config_manager.get_database_config(db_type)
        if db_config is None:
            raise VectorStoreError(f"No configuration found for {db_type.value}")
        
        # Merge configuration
        config = {
            **db_config.connection_params,
            **db_config.index_params,
            **db_config.search_params,
            **db_config.performance_params,
        }
        
        if config_override:
            config.update(config_override)
        
        # Check for existing instance
        if reuse_instance:
            instance_key = self.registry.get_instance_key(db_type, config)
            existing = self.registry.get_instance(instance_key)
            if existing and existing.is_connected:
                return existing
        
        # Create new instance
        try:
            instance = store_class(config)
            
            # Store instance if reuse is enabled
            if reuse_instance:
                instance_key = self.registry.get_instance_key(db_type, config)
                self.registry.add_instance(instance_key, instance)
            
            return instance
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create {db_type.value} store: {e}")
    
    async def create_all_enabled_stores(self) -> Dict[VectorDBType, BaseVectorStore]:
        """
        Create instances for all enabled databases.
        
        Returns:
            Dictionary mapping database types to store instances
        """
        stores = {}
        enabled_dbs = self.config_manager.get_enabled_databases()
        
        for db_type in enabled_dbs:
            try:
                store = await self.create_store(db_type)
                stores[db_type] = store
            except Exception as e:
                logger.error(f"Failed to create store for {db_type.value}: {e}")
        
        return stores
    
    async def get_or_create_store(self, db_type: Optional[VectorDBType] = None) -> BaseVectorStore:
        """
        Get existing store or create new one.
        
        Args:
            db_type: Database type (uses default if None)
            
        Returns:
            Vector store instance
        """
        return await self.create_store(db_type, reuse_instance=True)
    
    def get_available_stores(self) -> List[VectorDBType]:
        """Get list of available (registered and enabled) store types."""
        available = []
        enabled = self.config_manager.get_enabled_databases()
        
        for db_type in enabled:
            if self.registry.is_registered(db_type):
                available.append(db_type)
        
        return available
    
    def get_registered_stores(self) -> List[VectorDBType]:
        """Get list of registered store types."""
        return self.registry.get_registered_types()
    
    async def test_connection(self, db_type: VectorDBType) -> bool:
        """
        Test connection to a database.
        
        Args:
            db_type: Database type to test
            
        Returns:
            True if connection successful
        """
        try:
            store = await self.create_store(db_type, reuse_instance=False)
            await store.connect()
            result = await store.ping()
            await store.disconnect()
            return result
        except Exception as e:
            logger.error(f"Connection test failed for {db_type.value}: {e}")
            return False
    
    async def test_all_connections(self) -> Dict[VectorDBType, bool]:
        """
        Test connections to all enabled databases.
        
        Returns:
            Dictionary mapping database types to connection status
        """
        results = {}
        enabled_dbs = self.config_manager.get_enabled_databases()
        
        # Test connections concurrently
        tasks = []
        for db_type in enabled_dbs:
            if self.registry.is_registered(db_type):
                tasks.append(self._test_connection_task(db_type))
        
        if tasks:
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, db_type in enumerate([db for db in enabled_dbs if self.registry.is_registered(db)]):
                result = test_results[i]
                results[db_type] = result if isinstance(result, bool) else False
        
        return results
    
    async def _test_connection_task(self, db_type: VectorDBType) -> bool:
        """Task for testing a single connection."""
        return await self.test_connection(db_type)
    
    async def cleanup_instances(self) -> None:
        """Clean up all cached instances."""
        for instance in self.registry._instances.values():
            try:
                if instance.is_connected:
                    await instance.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting instance: {e}")
        
        self.registry.clear_instances()


# Global factory instance
_factory: Optional[VectorStoreFactory] = None


def get_vector_store_factory(config_manager: Optional[VectorDatabaseConfigManager] = None) -> VectorStoreFactory:
    """
    Get or create global vector store factory.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Vector store factory instance
    """
    global _factory
    
    if _factory is None:
        _factory = VectorStoreFactory(config_manager)
    
    return _factory


def reset_factory() -> None:
    """Reset global factory (for testing)."""
    global _factory
    _factory = None


# Convenience functions

async def create_vector_store(db_type: Optional[VectorDBType] = None,
                            config_override: Optional[Dict[str, Any]] = None) -> BaseVectorStore:
    """
    Create a vector store instance using the global factory.
    
    Args:
        db_type: Database type (uses default if None)
        config_override: Configuration overrides
        
    Returns:
        Vector store instance
    """
    factory = get_vector_store_factory()
    return await factory.create_store(db_type, config_override)


async def get_default_vector_store() -> BaseVectorStore:
    """Get the default vector store instance."""
    factory = get_vector_store_factory()
    return await factory.get_or_create_store()


async def test_vector_store_connections() -> Dict[VectorDBType, bool]:
    """Test all enabled vector store connections."""
    factory = get_vector_store_factory()
    return await factory.test_all_connections()
