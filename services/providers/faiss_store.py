"""
FAISS Vector Store Implementation

This module provides a FAISS-based implementation of the BaseVectorStore interface.
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
and clustering of dense vectors.
"""

import os
import pickle
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..vector_store_base import (
    BaseVectorStore, VectorDocument, SearchResult, SearchQuery,
    IndexStats, HealthStatus, VectorStoreStatus,
    VectorStoreError, ConnectionError, IndexError, SearchError, ValidationError
)

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store implementation.
    
    Features:
    - Multiple index types (Flat, IVF, HNSW)
    - Local file-based persistence
    - High-performance similarity search
    - Batch operations
    - Metadata storage
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FAISS vector store.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Configuration
        self.storage_path = config.get('storage_path', 'data/faiss_indexes')
        self.index_type = config.get('index_type', 'IndexHNSWFlat')
        self.dimension = config.get('dimension', 512)
        self.metric_type = config.get('metric_type', 'METRIC_INNER_PRODUCT')
        self.normalize_vectors = config.get('normalize_vectors', True)
        
        # FAISS settings
        self.nlist = config.get('nlist', 100)
        self.nprobe = config.get('nprobe', 10)
        self.M = config.get('M', 16)
        self.ef_construction = config.get('ef_construction', 200)
        self.ef_search = config.get('ef_search', 50)
        
        # Storage settings
        self.save_interval = config.get('save_interval', 1000)
        self.backup_enabled = config.get('backup_enabled', True)
        
        # Internal state
        self.index = None
        self.metadata_store = {}  # Document ID -> metadata mapping
        self.id_to_index = {}     # Document ID -> FAISS index mapping
        self.index_to_id = {}     # FAISS index -> Document ID mapping
        self.next_index = 0
        self.vectors_added = 0
        
        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)
    
    # Connection Management
    
    async def connect(self) -> None:
        """Initialize FAISS index and load existing data."""
        if self._is_connected:
            return
        
        try:
            # Import FAISS
            import faiss
            self.faiss = faiss
            
            # Create or load index
            await self._initialize_index()
            
            # Load metadata if exists
            await self._load_metadata()
            
            self._is_connected = True
            logger.info(f"Connected to FAISS store at {self.storage_path}")
            
        except ImportError:
            raise ConnectionError("FAISS library not installed. Install with: pip install faiss-cpu or faiss-gpu")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to FAISS store: {e}")
    
    async def disconnect(self) -> None:
        """Save and close FAISS index."""
        if not self._is_connected:
            return
        
        try:
            # Save current state
            await self._save_index()
            await self._save_metadata()
            
            self.index = None
            self._is_connected = False
            logger.info("Disconnected from FAISS store")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def ping(self) -> bool:
        """Test FAISS store availability."""
        return self._is_connected and self.index is not None
    
    # Index Management
    
    async def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        index_path = os.path.join(self.storage_path, f"{self.index_type}.index")
        
        if os.path.exists(index_path):
            # Load existing index
            self.index = self.faiss.read_index(index_path)
            logger.info(f"Loaded existing FAISS index from {index_path}")
        else:
            # Create new index
            self.index = self._create_faiss_index()
            logger.info(f"Created new FAISS index: {self.index_type}")
    
    def _create_faiss_index(self):
        """Create a new FAISS index based on configuration."""
        if self.index_type == "IndexFlatL2":
            return self.faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            return self.faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            quantizer = self.faiss.IndexFlatL2(self.dimension)
            index = self.faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            index.nprobe = self.nprobe
            return index
        elif self.index_type == "IndexHNSWFlat":
            index = self.faiss.IndexHNSWFlat(self.dimension, self.M)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            return index
        else:
            # Default to flat index
            return self.faiss.IndexFlatL2(self.dimension)
    
    async def create_index(self, index_name: str, dimension: int, 
                          distance_metric: str = "cosine",
                          index_config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new FAISS index."""
        try:
            # For FAISS, we reinitialize the index with new parameters
            self.dimension = dimension
            self.index = self._create_faiss_index()
            
            # Clear metadata
            self.metadata_store.clear()
            self.id_to_index.clear()
            self.index_to_id.clear()
            self.next_index = 0
            
            logger.info(f"Created new FAISS index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete FAISS index (reset to empty)."""
        try:
            self.index = self._create_faiss_index()
            self.metadata_store.clear()
            self.id_to_index.clear()
            self.index_to_id.clear()
            self.next_index = 0
            
            # Remove saved files
            index_path = os.path.join(self.storage_path, f"{self.index_type}.index")
            metadata_path = os.path.join(self.storage_path, "metadata.pkl")
            
            for path in [index_path, metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete FAISS index: {e}")
            return False
    
    async def index_exists(self, index_name: str) -> bool:
        """Check if index exists (always True for FAISS)."""
        return self.index is not None and self.index.ntotal > 0
    
    async def list_indexes(self) -> List[str]:
        """List available indexes."""
        return [self.index_type] if self.index is not None else []
    
    # Document Operations
    
    async def add_vectors(self, documents: List[VectorDocument], 
                         index_name: Optional[str] = None) -> bool:
        """Add vectors to FAISS index."""
        if not self._is_connected:
            await self.connect()
        
        try:
            vectors = []
            for doc in documents:
                # Validate vector
                if not await self.validate_vector(doc.vector, self.dimension):
                    raise ValidationError(f"Invalid vector for document {doc.id}")
                
                # Normalize if required
                vector = doc.vector
                if self.normalize_vectors:
                    vector = await self.normalize_vector(vector)
                
                vectors.append(vector)
                
                # Store metadata and ID mapping
                faiss_index = self.next_index
                self.metadata_store[doc.id] = {
                    'text': doc.text,
                    'metadata': doc.metadata or {},
                    'timestamp': doc.timestamp or datetime.utcnow().isoformat(),
                    'faiss_index': faiss_index
                }
                self.id_to_index[doc.id] = faiss_index
                self.index_to_id[faiss_index] = doc.id
                self.next_index += 1
            
            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Train index if needed (for IVF indexes)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if vectors_array.shape[0] >= self.nlist:
                    self.index.train(vectors_array)
                else:
                    logger.warning(f"Not enough vectors to train IVF index. Need at least {self.nlist}, got {vectors_array.shape[0]}")
            
            # Add vectors to index
            self.index.add(vectors_array)
            self.vectors_added += len(documents)
            
            # Save periodically
            if self.vectors_added % self.save_interval == 0:
                await self._save_index()
                await self._save_metadata()
            
            logger.info(f"Added {len(documents)} vectors to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False
    
    async def update_vectors(self, documents: List[VectorDocument],
                           index_name: Optional[str] = None) -> bool:
        """Update vectors (FAISS doesn't support updates, so we skip existing)."""
        # For FAISS, we can only add new vectors
        new_documents = []
        for doc in documents:
            if doc.id not in self.id_to_index:
                new_documents.append(doc)
        
        if new_documents:
            return await self.add_vectors(new_documents, index_name)
        return True
    
    async def delete_vectors(self, vector_ids: List[str],
                           index_name: Optional[str] = None) -> bool:
        """Delete vectors (FAISS doesn't support deletion, so we mark as deleted)."""
        try:
            deleted_count = 0
            for vector_id in vector_ids:
                if vector_id in self.metadata_store:
                    # Mark as deleted in metadata
                    self.metadata_store[vector_id]['deleted'] = True
                    deleted_count += 1
            
            logger.info(f"Marked {deleted_count} vectors as deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    async def get_vector(self, vector_id: str,
                        index_name: Optional[str] = None) -> Optional[VectorDocument]:
        """Retrieve a vector by ID."""
        try:
            if vector_id not in self.metadata_store:
                return None
            
            metadata = self.metadata_store[vector_id]
            if metadata.get('deleted', False):
                return None
            
            faiss_index = metadata['faiss_index']
            
            # FAISS doesn't support retrieving vectors by index easily
            # This is a limitation of FAISS - we'd need to store vectors separately
            # For now, return metadata only
            return VectorDocument(
                id=vector_id,
                vector=[],  # Not available in FAISS
                text=metadata.get('text'),
                metadata=metadata.get('metadata'),
                timestamp=metadata.get('timestamp')
            )
            
        except Exception as e:
            logger.error(f"Failed to get vector: {e}")
            return None
    
    # Search Operations
    
    async def search(self, query: SearchQuery,
                    index_name: Optional[str] = None) -> List[SearchResult]:
        """Perform vector similarity search."""
        if not self._is_connected:
            await self.connect()
        
        try:
            if query.vector is None:
                return []
            
            # Validate and normalize query vector
            if not await self.validate_vector(query.vector, self.dimension):
                raise ValidationError("Invalid query vector")
            
            query_vector = query.vector
            if self.normalize_vectors:
                query_vector = await self.normalize_vector(query_vector)
            
            # Convert to numpy array
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Perform search
            scores, indices = self.index.search(query_array, query.limit)
            
            # Convert results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # No more results
                    break
                
                # Get document ID
                doc_id = self.index_to_id.get(idx)
                if doc_id is None:
                    continue
                
                # Check if deleted
                metadata = self.metadata_store.get(doc_id, {})
                if metadata.get('deleted', False):
                    continue
                
                # Apply similarity threshold
                if query.similarity_threshold and score < query.similarity_threshold:
                    continue
                
                result = SearchResult(
                    id=doc_id,
                    score=float(score),
                    text=metadata.get('text') if query.include_metadata else None,
                    metadata=metadata.get('metadata') if query.include_metadata else None
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"FAISS search failed: {e}")
    
    async def hybrid_search(self, vector_query: List[float], text_query: str,
                          vector_weight: float = 0.7, text_weight: float = 0.3,
                          limit: int = 10, index_name: Optional[str] = None) -> List[SearchResult]:
        """FAISS doesn't support text search, fall back to vector search."""
        search_query = SearchQuery(vector=vector_query, limit=limit)
        return await self.search(search_query, index_name)
    
    async def batch_search(self, queries: List[SearchQuery],
                          index_name: Optional[str] = None) -> List[List[SearchResult]]:
        """Perform batch searches."""
        results = []
        for query in queries:
            result = await self.search(query, index_name)
            results.append(result)
        return results
    
    # Statistics and Monitoring
    
    async def get_index_stats(self, index_name: Optional[str] = None) -> IndexStats:
        """Get FAISS index statistics."""
        try:
            total_vectors = self.index.ntotal if self.index else 0
            active_vectors = len([m for m in self.metadata_store.values() if not m.get('deleted', False)])
            
            return IndexStats(
                total_vectors=active_vectors,
                index_size_bytes=0,  # FAISS doesn't provide this easily
                dimensions=self.dimension,
                distance_metric=self.metric_type,
                additional_stats={
                    'total_in_index': total_vectors,
                    'deleted_vectors': total_vectors - active_vectors,
                    'index_type': self.index_type,
                    'is_trained': getattr(self.index, 'is_trained', True) if self.index else False
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return IndexStats(
                total_vectors=0,
                index_size_bytes=0,
                dimensions=self.dimension,
                distance_metric=self.metric_type
            )
    
    async def get_health_status(self) -> HealthStatus:
        """Get FAISS store health status."""
        try:
            if not self._is_connected or self.index is None:
                return HealthStatus(
                    status=VectorStoreStatus.UNHEALTHY,
                    message="Not connected to FAISS store"
                )
            
            # Check if storage directory is accessible
            if not os.path.exists(self.storage_path):
                return HealthStatus(
                    status=VectorStoreStatus.DEGRADED,
                    message="Storage directory not accessible"
                )
            
            return HealthStatus(
                status=VectorStoreStatus.HEALTHY,
                message="FAISS store is healthy",
                last_check=datetime.utcnow().isoformat(),
                details={
                    'total_vectors': self.index.ntotal,
                    'storage_path': self.storage_path,
                    'index_type': self.index_type
                }
            )
            
        except Exception as e:
            return HealthStatus(
                status=VectorStoreStatus.UNHEALTHY,
                message=f"Health check failed: {e}"
            )
    
    # Utility Methods
    
    async def _save_index(self) -> None:
        """Save FAISS index to disk."""
        try:
            index_path = os.path.join(self.storage_path, f"{self.index_type}.index")
            self.faiss.write_index(self.index, index_path)
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            metadata_path = os.path.join(self.storage_path, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata_store': self.metadata_store,
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'next_index': self.next_index
                }, f)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def _load_metadata(self) -> None:
        """Load metadata from disk."""
        try:
            metadata_path = os.path.join(self.storage_path, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata_store = data.get('metadata_store', {})
                    self.id_to_index = data.get('id_to_index', {})
                    self.index_to_id = data.get('index_to_id', {})
                    self.next_index = data.get('next_index', 0)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
