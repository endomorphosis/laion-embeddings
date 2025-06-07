"""
Elasticsearch vector store implementation for unified vector database support.

This module provides an Elasticsearch-based implementation of the BaseVectorStore interface,
enabling seamless integration with the unified vector service architecture.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from ..vector_store_base import (
    BaseVectorStore, 
    SearchResult, 
    VectorDocument,
    IndexStats,
    HealthStatus
)

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk, scan
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    logger.warning("Elasticsearch client not available. Install with: pip install elasticsearch")
    ELASTICSEARCH_AVAILABLE = False
    # Mock class for when elasticsearch is not available
    class Elasticsearch:
        pass


class ElasticsearchVectorStore(BaseVectorStore):
    """Elasticsearch implementation of BaseVectorStore."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Elasticsearch vector store.
        
        Args:
            config: Configuration dictionary containing:
                - hosts: List of Elasticsearch hosts
                - index_name: Name of the index
                - vector_field: Name of the vector field (default: 'vector')
                - vector_size: Dimensionality of vectors
                - similarity: Similarity function (cosine, dot_product, l2_norm)
                - username: Optional username for authentication
                - password: Optional password for authentication
                - api_key: Optional API key for authentication
                - timeout: Connection timeout
                - max_retries: Maximum retry attempts
        """
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("Elasticsearch client not available. Install with: pip install elasticsearch")
        
        super().__init__(config)
        
        # Elasticsearch-specific configuration
        self.hosts = config.get('hosts', ['localhost:9200'])
        self.index_name = config.get('index_name', 'embeddings')
        self.vector_field = config.get('vector_field', 'vector')
        self.vector_size = config.get('vector_size', 768)
        self.similarity = config.get('similarity', 'cosine')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        # Authentication
        self.username = config.get('username')
        self.password = config.get('password')
        self.api_key = config.get('api_key')
        
        # Initialize client
        self.client: Optional[Elasticsearch] = None
        self._connected = False
        
        # Map similarity functions
        self._similarity_map = {
            'cosine': 'cosine',
            'dot_product': 'dot_product',
            'l2_norm': 'l2_norm',
            'euclidean': 'l2_norm'
        }
    
    async def connect(self) -> None:
        """Connect to Elasticsearch server."""
        try:
            # Prepare connection parameters
            client_config = {
                'hosts': self.hosts,
                'timeout': self.timeout,
                'max_retries': self.max_retries,
                'retry_on_timeout': True
            }
            
            # Add authentication if provided
            if self.api_key:
                client_config['api_key'] = self.api_key
            elif self.username and self.password:
                client_config['basic_auth'] = (self.username, self.password)
            
            # Create client
            self.client = Elasticsearch(**client_config)
            
            # Test connection
            if not await self.ping():
                raise ConnectionError("Unable to connect to Elasticsearch")
            
            # Ensure index exists
            await self._ensure_index_exists()
            
            self._connected = True
            logger.info(f"Connected to Elasticsearch at {self.hosts}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Elasticsearch server."""
        if self.client:
            try:
                await self.client.close()
            except Exception as e:
                logger.warning(f"Error during Elasticsearch disconnect: {e}")
            finally:
                self.client = None
                self._connected = False
    
    async def ping(self) -> bool:
        """Test connection to Elasticsearch server."""
        if not self.client:
            return False
        
        try:
            response = self.client.ping()
            return response
        except Exception as e:
            logger.error(f"Elasticsearch ping failed: {e}")
            return False
    
    async def _ensure_index_exists(self) -> None:
        """Ensure the index exists, create if necessary."""
        try:
            # Check if index exists
            if self.client.indices.exists(index=self.index_name):
                return
            
            # Create index with vector field mapping
            similarity = self._similarity_map.get(self.similarity.lower(), 'cosine')
            
            mapping = {
                "mappings": {
                    "properties": {
                        self.vector_field: {
                            "type": "dense_vector",
                            "dims": self.vector_size,
                            "similarity": similarity
                        },
                        "metadata": {
                            "type": "object",
                            "dynamic": True
                        },
                        "timestamp": {
                            "type": "date"
                        }
                    }
                },
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 1
                    }
                }
            }
            
            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created Elasticsearch index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise
    
    async def add_vectors(self, vectors: List[List[float]], 
                         metadata: Optional[List[Dict[str, Any]]] = None,
                         ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the Elasticsearch index.
        
        Args:
            vectors: List of vector embeddings
            metadata: Optional metadata for each vector
            ids: Optional IDs for vectors
            
        Returns:
            List of assigned vector IDs
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Elasticsearch")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}" 
                   for i in range(len(vectors))]
        
        # Prepare documents for bulk insert
        actions = []
        for i, vector in enumerate(vectors):
            doc = {
                '_index': self.index_name,
                '_id': ids[i],
                '_source': {
                    self.vector_field: vector,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            if metadata and i < len(metadata):
                doc['_source']['metadata'] = metadata[i]
            
            actions.append(doc)
        
        # Bulk insert
        try:
            success, failed = bulk(self.client, actions, refresh=True)
            
            if failed:
                logger.warning(f"Some documents failed to index: {failed}")
            
            logger.info(f"Added {success} vectors to Elasticsearch index {self.index_name}")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors to Elasticsearch: {e}")
            raise
    
    async def search_vectors(self, query_vector: List[float], 
                           limit: int = 10,
                           filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar vectors in Elasticsearch.
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Elasticsearch")
        
        try:
            # Build the query
            query = {
                "knn": {
                    "field": self.vector_field,
                    "query_vector": query_vector,
                    "k": limit,
                    "num_candidates": min(limit * 10, 10000)
                }
            }
            
            # Add filters if provided
            if filters:
                filter_clauses = self._build_es_filter(filters)
                if filter_clauses:
                    query["knn"]["filter"] = {
                        "bool": {
                            "must": filter_clauses
                        }
                    }
            
            # Perform search
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": limit,
                    "_source": {"excludes": [self.vector_field]}  # Don't return vectors for performance
                }
            )
            
            # Convert results
            results = []
            for hit in response['hits']['hits']:
                result = SearchResult(
                    id=hit['_id'],
                    score=float(hit['_score']),
                    metadata=hit['_source'].get('metadata', {}),
                    vector=None  # Not returned for performance
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in Elasticsearch: {e}")
            raise
    
    def _build_es_filter(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build Elasticsearch filter from generic filter dictionary."""
        filter_clauses = []
        
        for key, value in filters.items():
            es_key = f"metadata.{key}"
            
            if isinstance(value, (str, int, float, bool)):
                filter_clauses.append({
                    "term": {es_key: value}
                })
            elif isinstance(value, list):
                filter_clauses.append({
                    "terms": {es_key: value}
                })
            elif isinstance(value, dict):
                # Handle range queries
                range_clause = {}
                if 'gte' in value:
                    range_clause['gte'] = value['gte']
                if 'lte' in value:
                    range_clause['lte'] = value['lte']
                if 'gt' in value:
                    range_clause['gt'] = value['gt']
                if 'lt' in value:
                    range_clause['lt'] = value['lt']
                
                if range_clause:
                    filter_clauses.append({
                        "range": {es_key: range_clause}
                    })
        
        return filter_clauses
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors from Elasticsearch.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Elasticsearch")
        
        try:
            # Build delete actions
            actions = []
            for vector_id in ids:
                actions.append({
                    '_op_type': 'delete',
                    '_index': self.index_name,
                    '_id': vector_id
                })
            
            # Bulk delete
            success, failed = bulk(self.client, actions, refresh=True)
            
            if failed:
                logger.warning(f"Some documents failed to delete: {failed}")
            
            logger.info(f"Deleted {success} vectors from Elasticsearch index {self.index_name}")
            return len(failed) == 0
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Elasticsearch: {e}")
            return False
    
    async def get_vector(self, vector_id: str) -> Optional[VectorDocument]:
        """
        Retrieve a specific vector by ID.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Vector document or None if not found
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Elasticsearch")
        
        try:
            response = self.client.get(
                index=self.index_name,
                id=vector_id,
                _source=True
            )
            
            source = response['_source']
            return VectorDocument(
                id=vector_id,
                vector=source.get(self.vector_field),
                metadata=source.get('metadata', {})
            )
            
        except Exception as e:
            if "NotFoundError" in str(type(e)):
                return None
            logger.error(f"Failed to get vector from Elasticsearch: {e}")
            return None
    
    async def update_vector(self, vector_id: str, 
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a vector's data or metadata.
        
        Args:
            vector_id: Vector ID
            vector: New vector data (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Elasticsearch")
        
        try:
            # Build update document
            doc = {}
            if vector is not None:
                doc[self.vector_field] = vector
            if metadata is not None:
                doc['metadata'] = metadata
            
            if not doc:
                return False
            
            # Update document
            self.client.update(
                index=self.index_name,
                id=vector_id,
                body={
                    "doc": doc,
                    "doc_as_upsert": True
                },
                refresh=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update vector in Elasticsearch: {e}")
            return False
    
    async def get_index_stats(self) -> IndexStats:
        """Get statistics about the Elasticsearch index."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Elasticsearch")
        
        try:
            # Get index stats
            stats = self.client.indices.stats(index=self.index_name)
            index_stats = stats['indices'][self.index_name]
            
            # Get document count
            doc_count = index_stats['primaries']['docs']['count']
            
            # Get index size
            size_bytes = index_stats['primaries']['store']['size_in_bytes']
            
            return IndexStats(
                total_vectors=doc_count,
                index_size_bytes=size_bytes,
                dimensions=self.vector_size,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get Elasticsearch index stats: {e}")
            return IndexStats(
                total_vectors=0,
                index_size_bytes=0,
                dimensions=self.vector_size,
                last_updated=datetime.now()
            )
    
    async def get_health_status(self) -> HealthStatus:
        """Get health status of the Elasticsearch connection."""
        is_healthy = await self.ping()
        
        status = HealthStatus(
            is_healthy=is_healthy,
            database_type="elasticsearch",
            connection_info={
                "hosts": self.hosts,
                "index": self.index_name
            },
            last_check=datetime.now()
        )
        
        if is_healthy:
            try:
                # Get cluster health
                health = self.client.cluster.health()
                stats = await self.get_index_stats()
                
                status.additional_info = {
                    "cluster_status": health.get('status'),
                    "total_vectors": stats.total_vectors,
                    "index_size_bytes": stats.index_size_bytes,
                    "dimensions": stats.dimensions
                }
            except Exception as e:
                status.additional_info = {"stats_error": str(e)}
        
        return status
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Elasticsearch."""
        return self._connected and self.client is not None
    
    @property
    def database_type(self) -> str:
        """Get database type identifier."""
        return "elasticsearch"
