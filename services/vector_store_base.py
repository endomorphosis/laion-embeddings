"""
Base Vector Store Interface

This module defines the unified interface for all vector database implementations.
All vector store providers must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class VectorStoreStatus(Enum):
    """Vector store status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class VectorDocument:
    """Represents a document with vector embedding."""
    id: str
    vector: List[float]
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class SearchResult:
    """Represents a search result."""
    id: str
    score: float
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None


@dataclass
class SearchQuery:
    """Represents a search query."""
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    filter_expr: Optional[Dict[str, Any]] = None
    limit: int = 10
    offset: int = 0
    include_vectors: bool = False
    include_metadata: bool = True
    similarity_threshold: Optional[float] = None


@dataclass
class IndexStats:
    """Statistics about the vector index."""
    total_vectors: int
    index_size_bytes: int
    dimensions: int
    distance_metric: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    additional_stats: Optional[Dict[str, Any]] = None


@dataclass
class HealthStatus:
    """Health status of the vector store."""
    status: VectorStoreStatus
    message: str
    response_time_ms: Optional[float] = None
    last_check: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    This interface ensures consistency across all vector database providers
    (Qdrant, Elasticsearch, pgvector, FAISS).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vector store with configuration.
        
        Args:
            config: Database-specific configuration
        """
        self.config = config
        self._client = None
        self._is_connected = False
    
    # Connection Management
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the vector database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the vector database."""
        pass
    
    @abstractmethod
    async def ping(self) -> bool:
        """Test connection to the vector database."""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to the database."""
        return self._is_connected
    
    # Index Management
    
    @abstractmethod
    async def create_index(self, index_name: str, dimension: int, 
                          distance_metric: str = "cosine",
                          index_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new vector index.
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, euclidean, etc.)
            index_config: Additional index configuration
            
        Returns:
            True if index created successfully
        """
        pass
    
    @abstractmethod
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if index deleted successfully
        """
        pass
    
    @abstractmethod
    async def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.
        
        Args:
            index_name: Name of the index
            
        Returns:
            True if index exists
        """
        pass
    
    @abstractmethod
    async def list_indexes(self) -> List[str]:
        """
        List all available indexes.
        
        Returns:
            List of index names
        """
        pass
    
    # Document Operations
    
    @abstractmethod
    async def add_vectors(self, documents: List[VectorDocument], 
                         index_name: Optional[str] = None) -> bool:
        """
        Add vectors to the index.
        
        Args:
            documents: List of documents to add
            index_name: Target index name (uses default if None)
            
        Returns:
            True if vectors added successfully
        """
        pass
    
    @abstractmethod
    async def update_vectors(self, documents: List[VectorDocument],
                           index_name: Optional[str] = None) -> bool:
        """
        Update existing vectors in the index.
        
        Args:
            documents: List of documents to update
            index_name: Target index name (uses default if None)
            
        Returns:
            True if vectors updated successfully
        """
        pass
    
    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str],
                           index_name: Optional[str] = None) -> bool:
        """
        Delete vectors from the index.
        
        Args:
            vector_ids: List of vector IDs to delete
            index_name: Target index name (uses default if None)
            
        Returns:
            True if vectors deleted successfully
        """
        pass
    
    @abstractmethod
    async def get_vector(self, vector_id: str,
                        index_name: Optional[str] = None) -> Optional[VectorDocument]:
        """
        Retrieve a specific vector by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
            index_name: Target index name (uses default if None)
            
        Returns:
            Vector document or None if not found
        """
        pass
    
    # Search Operations
    
    @abstractmethod
    async def search(self, query: SearchQuery,
                    index_name: Optional[str] = None) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query parameters
            index_name: Target index name (uses default if None)
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def hybrid_search(self, vector_query: List[float], text_query: str,
                          vector_weight: float = 0.7, text_weight: float = 0.3,
                          limit: int = 10, index_name: Optional[str] = None) -> List[SearchResult]:
        """
        Perform hybrid vector + text search.
        
        Args:
            vector_query: Query vector
            text_query: Text query
            vector_weight: Weight for vector similarity
            text_weight: Weight for text similarity
            limit: Maximum number of results
            index_name: Target index name (uses default if None)
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def batch_search(self, queries: List[SearchQuery],
                          index_name: Optional[str] = None) -> List[List[SearchResult]]:
        """
        Perform batch vector searches.
        
        Args:
            queries: List of search queries
            index_name: Target index name (uses default if None)
            
        Returns:
            List of result lists (one per query)
        """
        pass
    
    # Statistics and Monitoring
    
    @abstractmethod
    async def get_index_stats(self, index_name: Optional[str] = None) -> IndexStats:
        """
        Get index statistics.
        
        Args:
            index_name: Target index name (uses default if None)
            
        Returns:
            Index statistics
        """
        pass
    
    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """
        Get health status of the vector store.
        
        Returns:
            Health status information
        """
        pass
    
    # Utility Methods
    
    async def validate_vector(self, vector: List[float], expected_dim: Optional[int] = None) -> bool:
        """
        Validate vector format and dimensions.
        
        Args:
            vector: Vector to validate
            expected_dim: Expected vector dimension
            
        Returns:
            True if vector is valid
        """
        if not isinstance(vector, (list, np.ndarray)):
            return False
        
        if len(vector) == 0:
            return False
        
        if expected_dim is not None and len(vector) != expected_dim:
            return False
        
        # Check if all elements are numeric
        try:
            float_vector = [float(x) for x in vector]
            return True
        except (ValueError, TypeError):
            return False
    
    async def normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize vector to unit length.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector
        """
        np_vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(np_vector)
        if norm == 0:
            return vector
        return (np_vector / norm).tolist()
    
    def get_default_index_name(self) -> str:
        """Get the default index name from configuration."""
        return self.config.get('index_name', 'embeddings')
    
    def get_vector_dimension(self) -> int:
        """Get the vector dimension from configuration."""
        return self.config.get('dimension', 512)
    
    # Context Manager Support
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class ConnectionError(VectorStoreError):
    """Raised when connection to vector store fails."""
    pass


class IndexError(VectorStoreError):
    """Raised when index operations fail."""
    pass


class SearchError(VectorStoreError):
    """Raised when search operations fail."""
    pass


class ValidationError(VectorStoreError):
    """Raised when input validation fails."""
    pass
