"""
Qdrant vector store implementation for unified vector database support.

This module provides a Qdrant-based implementation of the BaseVectorStore interface,
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
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant client not available. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False
    # Mock classes for when qdrant is not available
    class QdrantClient:
        pass
    
    class models:
        class Distance:
            COSINE = "Cosine"
            EUCLID = "Euclid"
            DOT = "Dot"
        
        class VectorParams:
            pass
        
        class CollectionStatus:
            pass
        
        class Filter:
            pass
        
        class FieldCondition:
            pass
        
        class MatchValue:
            pass
        
        class MatchAny:
            pass
        
        class Range:
            pass
        
        class PointStruct:
            pass
        
        class PointIdsList:
            pass


class QdrantVectorStore(BaseVectorStore):
    """Qdrant implementation of BaseVectorStore."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qdrant vector store.
        
        Args:
            config: Configuration dictionary containing:
                - host: Qdrant server host (default: localhost)
                - port: Qdrant server port (default: 6333)
                - collection_name: Name of the collection
                - vector_size: Dimensionality of vectors
                - distance_metric: Distance metric (cosine, euclidean, dot)
                - timeout: Connection timeout
                - api_key: Optional API key for authentication
                - prefer_grpc: Whether to use gRPC connection
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        super().__init__(config)
        
        # Qdrant-specific configuration
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6333)
        self.collection_name = config.get('collection_name', config.get('index_name', 'embeddings'))
        self.vector_size = config.get('vector_size', 768)
        self.distance_metric = config.get('distance_metric', 'cosine')
        self.timeout = config.get('timeout', 30)
        self.api_key = config.get('api_key')
        self.prefer_grpc = config.get('prefer_grpc', False)
        
        # Initialize client
        self.client: Optional[QdrantClient] = None
        self._connected = False
        
        # Map distance metrics
        self._distance_map = {
            'cosine': Distance.COSINE,
            'euclidean': Distance.EUCLID,
            'dot': Distance.DOT,
            'l2': Distance.EUCLID
        }
    
    async def connect(self) -> None:
        """Connect to Qdrant server."""
        try:
            # Create client
            if self.api_key:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    prefer_grpc=self.prefer_grpc
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    timeout=self.timeout,
                    prefer_grpc=self.prefer_grpc
                )
            
            # Test connection
            await self.ping()
            
            # Ensure collection exists
            await self._ensure_collection_exists()
            
            self._connected = True
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Qdrant server."""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error during Qdrant disconnect: {e}")
            finally:
                self.client = None
                self._connected = False
    
    async def ping(self) -> bool:
        """Test connection to Qdrant server."""
        if not self.client:
            return False
        
        try:
            # Use health check endpoint
            health = self.client.http.health_api.healthz()
            return health.status == "ok"
        except Exception as e:
            logger.error(f"Qdrant ping failed: {e}")
            return False
    
    async def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, create if necessary."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                distance = self._distance_map.get(self.distance_metric.lower(), Distance.COSINE)
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=distance
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    async def add_vectors(self, vectors: List[List[float]], 
                         metadata: Optional[List[Dict[str, Any]]] = None,
                         ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the Qdrant collection.
        
        Args:
            vectors: List of vector embeddings
            metadata: Optional metadata for each vector
            ids: Optional IDs for vectors
            
        Returns:
            List of assigned vector IDs
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}" 
                   for i in range(len(vectors))]
        
        # Prepare points
        points = []
        for i, vector in enumerate(vectors):
            point_data = {
                "id": ids[i],
                "vector": vector,
            }
            
            if metadata and i < len(metadata):
                point_data["payload"] = metadata[i]
            
            points.append(models.PointStruct(**point_data))
        
        # Insert points
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(vectors)} vectors to Qdrant collection {self.collection_name}")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors to Qdrant: {e}")
            raise
    
    async def search_vectors(self, query_vector: List[float], 
                           limit: int = 10,
                           filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar vectors in Qdrant.
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant")
        
        try:
            # Convert filters to Qdrant format
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_qdrant_filter(filters)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert results
            results = []
            for point in search_result:
                result = SearchResult(
                    id=str(point.id),
                    score=float(point.score),
                    metadata=point.payload or {},
                    vector=None  # Not returned by default for performance
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in Qdrant: {e}")
            raise
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from generic filter dictionary."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            elif isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )
            elif isinstance(value, dict):
                # Handle range queries
                if 'gte' in value or 'lte' in value or 'gt' in value or 'lt' in value:
                    range_condition = models.Range()
                    if 'gte' in value:
                        range_condition.gte = value['gte']
                    if 'lte' in value:
                        range_condition.lte = value['lte']
                    if 'gt' in value:
                        range_condition.gt = value['gt']
                    if 'lt' in value:
                        range_condition.lt = value['lt']
                    
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=range_condition
                        )
                    )
        
        return models.Filter(must=conditions) if conditions else None
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors from Qdrant.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant")
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            
            logger.info(f"Deleted {len(ids)} vectors from Qdrant collection {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Qdrant: {e}")
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
            raise RuntimeError("Not connected to Qdrant")
        
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return None
            
            point = points[0]
            return VectorDocument(
                id=str(point.id),
                vector=point.vector,
                metadata=point.payload or {}
            )
            
        except Exception as e:
            logger.error(f"Failed to get vector from Qdrant: {e}")
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
            raise RuntimeError("Not connected to Qdrant")
        
        try:
            # Get current vector if we need to update only metadata
            if metadata and not vector:
                current = await self.get_vector(vector_id)
                if not current:
                    return False
                vector = current.vector
            
            if not vector:
                return False
            
            # Update the point
            point = models.PointStruct(
                id=vector_id,
                vector=vector,
                payload=metadata or {}
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update vector in Qdrant: {e}")
            return False
    
    async def get_index_stats(self) -> IndexStats:
        """Get statistics about the Qdrant collection."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Qdrant")
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return IndexStats(
                total_vectors=collection_info.points_count or 0,
                index_size_bytes=0,  # Qdrant doesn't provide this directly
                dimensions=collection_info.config.params.vectors.size,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get Qdrant index stats: {e}")
            return IndexStats(
                total_vectors=0,
                index_size_bytes=0,
                dimensions=self.vector_size,
                last_updated=datetime.now()
            )
    
    async def get_health_status(self) -> HealthStatus:
        """Get health status of the Qdrant connection."""
        is_healthy = await self.ping()
        
        status = HealthStatus(
            is_healthy=is_healthy,
            database_type="qdrant",
            connection_info={
                "host": self.host,
                "port": self.port,
                "collection": self.collection_name
            },
            last_check=datetime.now()
        )
        
        if is_healthy:
            try:
                stats = await self.get_index_stats()
                status.additional_info = {
                    "total_vectors": stats.total_vectors,
                    "dimensions": stats.dimensions
                }
            except Exception as e:
                status.additional_info = {"stats_error": str(e)}
        
        return status
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        return self._connected and self.client is not None
    
    @property
    def database_type(self) -> str:
        """Get database type identifier."""
        return "qdrant"
