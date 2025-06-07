"""
IPFS Vector Store Provider

This module implements the IPFS/IPLD vector store provider for the unified vector database architecture.
It leverages IPFS for distributed, content-addressable storage of vector embeddings.
"""

import logging
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import asyncio
import time
from datetime import datetime

from ..vector_store_base import (
    BaseVectorStore,
    VectorDocument,
    SearchResult,
    SearchQuery,
    IndexStats,
    HealthStatus,
    VectorStoreStatus,
    VectorStoreError,
    ConnectionError
)

# Check for IPFS kit availability
try:
    from ipfs_kit_py import ipfs_kit
    IPFS_KIT_AVAILABLE = True
except ImportError:
    IPFS_KIT_AVAILABLE = False

# Try to import direct ipfshttpclient
try:
    import ipfshttpclient
    IPFS_CLIENT_AVAILABLE = True
except ImportError:
    IPFS_CLIENT_AVAILABLE = False

# Import existing IPFSVectorStorage from IPFS vector service
from ..ipfs_vector_service import IPFSVectorStorage, IPFSConfig

logger = logging.getLogger(__name__)


class IPFSVectorStore(BaseVectorStore):
    """
    IPFS Vector Store implementation for distributed vector storage and retrieval.
    
    This implementation leverages IPFS/IPLD for content-addressable storage of vector embeddings,
    enabling distributed and resilient vector databases.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize IPFS Vector Store with configuration.
        
        Args:
            config: IPFS-specific configuration
        """
        super().__init__(config)
        self._client = None
        self._is_connected = False
        self.storage = None
        
        # Extract IPFS specific config
        self.ipfs_gateway = self.config.get('ipfs_gateway', 'localhost:5001')
        self.api_url = f'/ip4/{self.ipfs_gateway.split(":")[0]}/tcp/{self.ipfs_gateway.split(":")[1]}'
        self.sharding_enabled = self.config.get('sharding_enabled', True)
        self.max_shard_size = self.config.get('max_shard_size', 10000)
        self.dimension = self.config.get('dimension', 512)
        self.cache_path = self.config.get('storage', {}).get('cache_path', 'data/ipfs_cache')
        
        # Create cache directory if it doesn't exist
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        
        # Index metadata
        self._index_name = None
        self._index_cid = None
        self._index_metadata = {
            'name': None,
            'dimension': self.dimension,
            'distance_metric': 'cosine',
            'created_at': None,
            'updated_at': None,
            'total_vectors': 0,
            'shard_count': 0,
            'shards': {}
        }
    
    # Connection Management

    async def connect(self) -> None:
        """
        Establish connection to the IPFS node.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Create IPFS config
            ipfs_config = {
                'api_url': self.api_url,
                'gateway_url': f'http://{self.ipfs_gateway.split(":")[0]}:8080',
                'timeout': self.config.get('timeout', 60),
                'chunk_size': self.max_shard_size,
                'compression': self.config.get('settings', {}).get('compression', True),
                'pin_content': self.config.get('settings', {}).get('pin_content', True)
            }
            
            # Initialize IPFS storage
            self.storage = IPFSVectorStorage(ipfs_config)
            self._client = self.storage.client
            
            # Check connection by getting IPFS version
            version = self._client.version()
            logger.info(f"Connected to IPFS node version {version.get('Version', 'unknown')}")
            self._is_connected = True
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to IPFS: {e}")
            raise ConnectionError(f"Failed to connect to IPFS: {e}")
    
    async def disconnect(self) -> None:
        """Close connection to IPFS."""
        self._client = None
        self._is_connected = False
        self.storage = None
    
    async def ping(self) -> bool:
        """
        Test connection to IPFS.
        
        Returns:
            True if connected
        """
        try:
            if not self._client:
                return False
            
            version = self._client.version()
            return 'Version' in version
        except Exception:
            return False
    
    # Index Management
    
    async def create_index(self, index_name: str, dimension: int, 
                          distance_metric: str = "cosine",
                          index_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new vector index in IPFS.
        
        For IPFS, this creates a manifest file that will track shards.
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, euclidean, etc.)
            index_config: Additional index configuration
            
        Returns:
            True if index created successfully
        """
        try:
            if not self._is_connected:
                await self.connect()
            
            # Initialize index metadata
            self._index_name = index_name
            self._index_metadata = {
                'name': index_name,
                'dimension': dimension,
                'distance_metric': distance_metric,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'total_vectors': 0,
                'shard_count': 0,
                'shards': {},
                'config': index_config or {}
            }
            
            # Store index metadata in IPFS
            self._index_cid = await self.storage.store_index_metadata(self._index_metadata)
            logger.info(f"Created index {index_name} with CID {self._index_cid}")
            
            # Create local manifest file for easy recovery
            manifest_path = Path(self.cache_path) / f"{index_name}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump({
                    'index_name': index_name,
                    'cid': self._index_cid,
                    'created_at': self._index_metadata['created_at']
                }, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise VectorStoreError(f"Failed to create index {index_name}: {e}")
    
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index.
        
        For IPFS, this unpins the manifest and all referenced shards.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if index deleted successfully
        """
        try:
            if not self._is_connected:
                await self.connect()
                
            # Check if we're deleting the current index
            if self._index_name == index_name:
                # Load index metadata to get all shards
                metadata = await self.storage.retrieve_index_metadata(self._index_cid)
                
                # Unpin all shards
                for shard_id, shard_info in metadata.get('shards', {}).items():
                    try:
                        cid = shard_info.get('ipfs_hash')
                        if cid:
                            self._client.pin.rm(cid)
                            logger.info(f"Unpinned shard {shard_id} with CID {cid}")
                    except Exception as e:
                        logger.warning(f"Failed to unpin shard {shard_id}: {e}")
                
                # Unpin the manifest
                try:
                    self._client.pin.rm(self._index_cid)
                    logger.info(f"Unpinned index manifest {self._index_cid}")
                except Exception as e:
                    logger.warning(f"Failed to unpin index manifest {self._index_cid}: {e}")
                
                # Clear the index metadata
                self._index_name = None
                self._index_cid = None
                self._index_metadata = {
                    'name': None,
                    'dimension': self.dimension,
                    'distance_metric': 'cosine',
                    'created_at': None,
                    'updated_at': None,
                    'total_vectors': 0,
                    'shard_count': 0,
                    'shards': {}
                }
                
                # Remove local manifest file
                manifest_path = Path(self.cache_path) / f"{index_name}_manifest.json"
                if manifest_path.exists():
                    manifest_path.unlink()
                
                return True
            else:
                # Find the manifest for the specified index
                manifest_path = Path(self.cache_path) / f"{index_name}_manifest.json"
                if not manifest_path.exists():
                    logger.warning(f"No local manifest found for index {index_name}")
                    return False
                
                # Load manifest to get CID
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    
                index_cid = manifest.get('cid')
                if not index_cid:
                    logger.warning(f"No CID found in manifest for index {index_name}")
                    return False
                
                # Load index metadata to get all shards
                metadata = await self.storage.retrieve_index_metadata(index_cid)
                
                # Unpin all shards
                for shard_id, shard_info in metadata.get('shards', {}).items():
                    try:
                        cid = shard_info.get('ipfs_hash')
                        if cid:
                            self._client.pin.rm(cid)
                            logger.info(f"Unpinned shard {shard_id} with CID {cid}")
                    except Exception as e:
                        logger.warning(f"Failed to unpin shard {shard_id}: {e}")
                
                # Unpin the manifest
                try:
                    self._client.pin.rm(index_cid)
                    logger.info(f"Unpinned index manifest {index_cid}")
                except Exception as e:
                    logger.warning(f"Failed to unpin index manifest {index_cid}: {e}")
                
                # Remove local manifest file
                manifest_path.unlink()
                
                return True
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            raise VectorStoreError(f"Failed to delete index {index_name}: {e}")
    
    async def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.
        
        Args:
            index_name: Name of the index
            
        Returns:
            True if index exists
        """
        try:
            if not self._is_connected:
                await self.connect()
                
            # Check if it's the current index
            if self._index_name == index_name and self._index_cid:
                return True
                
            # Check for local manifest file
            manifest_path = Path(self.cache_path) / f"{index_name}_manifest.json"
            return manifest_path.exists()
        except Exception as e:
            logger.error(f"Error checking if index {index_name} exists: {e}")
            return False
    
    async def list_indexes(self) -> List[str]:
        """
        List all available indexes.
        
        Returns:
            List of index names
        """
        indexes = []
        
        try:
            # Scan the cache directory for manifests
            cache_dir = Path(self.cache_path)
            if cache_dir.exists():
                for file_path in cache_dir.glob("*_manifest.json"):
                    try:
                        index_name = file_path.stem.replace("_manifest", "")
                        indexes.append(index_name)
                    except Exception:
                        continue
            
            # Add current index if it exists and is not already in the list
            if self._index_name and self._index_name not in indexes:
                indexes.append(self._index_name)
                
            return indexes
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return []
    
    async def get_index_stats(self, index_name: Optional[str] = None) -> IndexStats:
        """
        Get index statistics.
        
        Args:
            index_name: Name of the index (uses current index if None)
            
        Returns:
            Index statistics
        """
        try:
            if not self._is_connected:
                await self.connect()
                
            if index_name is None:
                index_name = self._index_name
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            # Use current metadata if asking about current index
            if index_name == self._index_name and self._index_metadata.get('name') == index_name:
                metadata = self._index_metadata
            else:
                # Load manifest to get CID
                manifest_path = Path(self.cache_path) / f"{index_name}_manifest.json"
                if not manifest_path.exists():
                    raise VectorStoreError(f"Index {index_name} not found")
                
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    
                index_cid = manifest.get('cid')
                if not index_cid:
                    raise VectorStoreError(f"No CID found for index {index_name}")
                
                # Load index metadata
                metadata = await self.storage.retrieve_index_metadata(index_cid)
                
            # Estimate index size
            index_size_bytes = 0
            for shard_info in metadata.get('shards', {}).values():
                # Assume average vector size is 4 bytes per dimension plus overhead
                vector_count = shard_info.get('vector_count', 0)
                vector_size = vector_count * (metadata.get('dimension', 512) * 4 + 100)  # 4 bytes per float + metadata overhead
                index_size_bytes += vector_size
                
            # Create IndexStats object
            return IndexStats(
                total_vectors=metadata.get('total_vectors', 0),
                index_size_bytes=index_size_bytes,
                dimensions=metadata.get('dimension', 512),
                distance_metric=metadata.get('distance_metric', 'cosine'),
                created_at=metadata.get('created_at'),
                updated_at=metadata.get('updated_at'),
                additional_stats={
                    'shard_count': metadata.get('shard_count', 0),
                    'ipfs_cid': index_cid if index_name != self._index_name else self._index_cid
                }
            )
        except Exception as e:
            logger.error(f"Failed to get stats for index {index_name}: {e}")
            raise VectorStoreError(f"Failed to get stats for index {index_name}: {e}")
    
    # Document Operations
    
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
        try:
            if not self._is_connected:
                await self.connect()
                
            if index_name is None:
                index_name = self._index_name
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            # If not using current index, load the specified index
            if index_name != self._index_name:
                await self._load_index(index_name)
                
            # Extract vectors and metadata from documents
            vectors = np.array([doc.vector for doc in documents], dtype=np.float32)
            
            # Build metadata for each document
            metadata_list = []
            for doc in documents:
                metadata = doc.metadata or {}
                if doc.text:
                    metadata['text'] = doc.text
                if doc.id:
                    metadata['id'] = doc.id
                if doc.timestamp:
                    metadata['timestamp'] = doc.timestamp
                metadata_list.append(metadata)
                
            # Store in batches based on shard size
            remaining = len(vectors)
            offset = 0
            
            while remaining > 0:
                # Calculate batch size
                batch_size = min(remaining, self.max_shard_size)
                
                # Extract batch
                batch_vectors = vectors[offset:offset+batch_size]
                batch_metadata = metadata_list[offset:offset+batch_size]
                
                # Generate shard ID
                from uuid import uuid4
                shard_id = f"shard_{str(uuid4())[:8]}"
                
                # Store shard
                cid = await self.storage.store_vector_shard(
                    batch_vectors,
                    batch_metadata,
                    shard_id
                )
                
                # Update index metadata
                self._index_metadata['shards'][shard_id] = {
                    'ipfs_hash': cid,
                    'vector_count': len(batch_vectors),
                    'created_at': datetime.now().isoformat()
                }
                self._index_metadata['total_vectors'] += len(batch_vectors)
                self._index_metadata['shard_count'] += 1
                self._index_metadata['updated_at'] = datetime.now().isoformat()
                
                # Update counters
                offset += batch_size
                remaining -= batch_size
                
            # Update index metadata in IPFS
            self._index_cid = await self.storage.store_index_metadata(self._index_metadata)
            
            # Update local manifest
            manifest_path = Path(self.cache_path) / f"{index_name}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump({
                    'index_name': index_name,
                    'cid': self._index_cid,
                    'created_at': self._index_metadata['created_at'],
                    'updated_at': self._index_metadata['updated_at']
                }, f)
                
            return True
        except Exception as e:
            logger.error(f"Failed to add vectors to index {index_name}: {e}")
            raise VectorStoreError(f"Failed to add vectors: {e}")
    
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
        # For IPFS, updating means removing and adding
        try:
            if not self._is_connected:
                await self.connect()
                
            if index_name is None:
                index_name = self._index_name
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            # If not using current index, load the specified index
            if index_name != self._index_name:
                await self._load_index(index_name)
                
            # Get document IDs to remove
            document_ids = [doc.id for doc in documents]
            
            # Remove the documents
            await self.delete_vectors(document_ids, index_name)
            
            # Add the updated documents
            await self.add_vectors(documents, index_name)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update vectors in index {index_name}: {e}")
            raise VectorStoreError(f"Failed to update vectors: {e}")
    
    async def delete_vectors(self, ids: List[str], 
                           index_name: Optional[str] = None) -> bool:
        """
        Delete vectors from the index.
        
        For IPFS, we need to rebuild shards that contain the vectors to delete.
        
        Args:
            ids: List of document IDs to delete
            index_name: Target index name (uses default if None)
            
        Returns:
            True if vectors deleted successfully
        """
        try:
            if not self._is_connected:
                await self.connect()
                
            if index_name is None:
                index_name = self._index_name
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            # If not using current index, load the specified index
            if index_name != self._index_name:
                await self._load_index(index_name)
                
            # Convert IDs to set for faster lookup
            id_set = set(ids)
            
            # We need to scan all shards to find the vectors to delete
            affected_shards = {}
            
            # Load each shard and check for vectors to delete
            for shard_id, shard_info in list(self._index_metadata['shards'].items()):
                cid = shard_info['ipfs_hash']
                shard_data = await self.storage.retrieve_vector_shard(cid)
                
                # Check if any of the IDs are in this shard
                vectors_to_keep = []
                metadata_to_keep = []
                removed_count = 0
                
                metadata_list = shard_data.get('metadata', [])
                
                for i, metadata in enumerate(metadata_list):
                    if metadata.get('id') in id_set:
                        removed_count += 1
                    else:
                        vectors_to_keep.append(shard_data['vectors'][i])
                        metadata_to_keep.append(metadata)
                
                # If we found vectors to delete, we need to rebuild this shard
                if removed_count > 0:
                    if vectors_to_keep:
                        # Rebuild shard with remaining vectors
                        new_vectors = np.array(vectors_to_keep, dtype=shard_data.get('dtype', 'float32'))
                        
                        # Store updated shard
                        new_cid = await self.storage.store_vector_shard(
                            new_vectors,
                            metadata_to_keep,
                            shard_id
                        )
                        
                        # Update metadata
                        self._index_metadata['shards'][shard_id] = {
                            'ipfs_hash': new_cid,
                            'vector_count': len(vectors_to_keep),
                            'updated_at': datetime.now().isoformat()
                        }
                    else:
                        # If shard is empty, remove it
                        del self._index_metadata['shards'][shard_id]
                    
                    # Track vectors removed
                    self._index_metadata['total_vectors'] -= removed_count
                    if not self._index_metadata['shards']:
                        self._index_metadata['shard_count'] = 0
                    else:
                        self._index_metadata['shard_count'] = len(self._index_metadata['shards'])
                    self._index_metadata['updated_at'] = datetime.now().isoformat()
            
            # Update index metadata in IPFS
            self._index_cid = await self.storage.store_index_metadata(self._index_metadata)
            
            # Update local manifest
            manifest_path = Path(self.cache_path) / f"{index_name}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump({
                    'index_name': index_name,
                    'cid': self._index_cid,
                    'created_at': self._index_metadata['created_at'],
                    'updated_at': self._index_metadata['updated_at']
                }, f)
                
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from index {index_name}: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}")
    
    # Vector Search
    
    async def search(self, query: Union[SearchQuery, np.ndarray, List[float]],
                    index_name: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query: Search query object or vector
            index_name: Target index name (uses default if None)
            
        Returns:
            List of search results
        """
        try:
            if not self._is_connected:
                await self.connect()
                
            if index_name is None:
                index_name = self._index_name
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            # If not using current index, load the specified index
            if index_name != self._index_name:
                await self._load_index(index_name)
            
            # Convert query to SearchQuery if it's a vector
            if isinstance(query, (np.ndarray, list)):
                if isinstance(query, np.ndarray) and query.ndim == 2:
                    query_vector = query[0]
                else:
                    query_vector = query
                query = SearchQuery(
                    vector=query_vector,
                    limit=10,
                    include_vectors=False,
                    include_metadata=True
                )
            
            # Convert query vector to numpy array
            query_vector = np.array(query.vector, dtype=np.float32)
            
            # Search in each shard
            all_results = []
            
            # Load each shard and perform search
            shard_tasks = []
            for shard_id, shard_info in self._index_metadata['shards'].items():
                cid = shard_info['ipfs_hash']
                shard_tasks.append(self._search_shard(cid, query_vector, query))
            
            # Wait for all shard searches to complete
            shard_results = await asyncio.gather(*shard_tasks)
            
            # Combine results from all shards
            for results in shard_results:
                all_results.extend(results)
            
            # Sort by score (highest first) and limit
            all_results.sort(key=lambda r: r.score, reverse=True)
            all_results = all_results[:query.limit]
            
            return all_results
        except Exception as e:
            logger.error(f"Failed to search in index {index_name}: {e}")
            raise VectorStoreError(f"Failed to search: {e}")
    
    async def _search_shard(self, cid: str, query_vector: np.ndarray, query: SearchQuery) -> List[SearchResult]:
        """
        Search within a single shard.
        
        Args:
            cid: IPFS CID of the shard
            query_vector: Query vector
            query: Search query parameters
            
        Returns:
            List of search results from this shard
        """
        # Retrieve shard data
        shard_data = await self.storage.retrieve_vector_shard(cid)
        
        # Extract vectors and metadata
        vectors = shard_data.get('vectors')
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        metadata_list = shard_data.get('metadata', [])
        
        # Calculate similarity scores
        results = []
        
        # Use cosine similarity by default
        # Normalize vectors for cosine similarity
        query_vector_normalized = query_vector / np.linalg.norm(query_vector)
        vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        scores = np.dot(vectors_normalized, query_vector_normalized)
        
        # Create search results
        for i in range(len(scores)):
            # Apply filter if provided
            if query.filter_expr and metadata_list and i < len(metadata_list):
                metadata = metadata_list[i]
                if not self._apply_filter(metadata, query.filter_expr):
                    continue
            
            # Apply similarity threshold if provided
            if query.similarity_threshold is not None and scores[i] < query.similarity_threshold:
                continue
                
            # Create search result
            result = SearchResult(
                id=metadata_list[i].get('id', f"id_{i}") if i < len(metadata_list) else f"id_{i}",
                score=float(scores[i]),
                distance=float(1.0 - scores[i]),  # Convert similarity to distance
                text=metadata_list[i].get('text') if i < len(metadata_list) and query.include_metadata else None,
                metadata=metadata_list[i] if i < len(metadata_list) and query.include_metadata else None,
                vector=vectors[i].tolist() if query.include_vectors else None
            )
            results.append(result)
        
        # Sort by score (highest first)
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results
    
    def _apply_filter(self, metadata: Dict[str, Any], filter_expr: Dict[str, Any]) -> bool:
        """
        Apply filter expression to metadata.
        
        Args:
            metadata: Document metadata
            filter_expr: Filter expression
            
        Returns:
            True if metadata matches filter
        """
        # Basic filter implementation
        for key, value in filter_expr.items():
            # Handle special operators
            if key.startswith('$'):
                if key == '$and' and isinstance(value, list):
                    return all(self._apply_filter(metadata, subfilter) for subfilter in value)
                elif key == '$or' and isinstance(value, list):
                    return any(self._apply_filter(metadata, subfilter) for subfilter in value)
                elif key == '$not' and isinstance(value, dict):
                    return not self._apply_filter(metadata, value)
                continue
            
            # Regular key-value matching
            if key not in metadata:
                return False
            
            # Value can be dict for operators
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == '$eq':
                        if metadata[key] != op_value:
                            return False
                    elif op == '$ne':
                        if metadata[key] == op_value:
                            return False
                    elif op == '$gt':
                        if not isinstance(metadata[key], (int, float)) or metadata[key] <= op_value:
                            return False
                    elif op == '$gte':
                        if not isinstance(metadata[key], (int, float)) or metadata[key] < op_value:
                            return False
                    elif op == '$lt':
                        if not isinstance(metadata[key], (int, float)) or metadata[key] >= op_value:
                            return False
                    elif op == '$lte':
                        if not isinstance(metadata[key], (int, float)) or metadata[key] > op_value:
                            return False
                    elif op == '$in' and isinstance(op_value, list):
                        if metadata[key] not in op_value:
                            return False
                    elif op == '$nin' and isinstance(op_value, list):
                        if metadata[key] in op_value:
                            return False
            else:
                # Direct value comparison
                if metadata[key] != value:
                    return False
                    
        return True
    
    # Health and Monitoring
    
    async def get_health(self) -> HealthStatus:
        """
        Get health status of the vector store.
        
        Returns:
            Health status
        """
        try:
            start_time = time.time()
            
            if not self._is_connected:
                return HealthStatus(
                    status=VectorStoreStatus.UNHEALTHY,
                    message="Not connected to IPFS",
                    response_time_ms=0,
                    last_check=datetime.now().isoformat()
                )
                
            # Check IPFS by getting version
            version = self._client.version()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                status=VectorStoreStatus.HEALTHY,
                message=f"Connected to IPFS v{version.get('Version', 'unknown')}",
                response_time_ms=response_time,
                last_check=datetime.now().isoformat(),
                details={
                    'version': version.get('Version', 'unknown'),
                    'current_index': self._index_name,
                    'shards': len(self._index_metadata.get('shards', {})) if self._index_metadata else 0
                }
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                status=VectorStoreStatus.UNHEALTHY,
                message=f"IPFS health check failed: {e}",
                response_time_ms=response_time,
                last_check=datetime.now().isoformat()
            )
    
    # Helper Methods
    
    async def _load_index(self, index_name: str) -> None:
        """
        Load an index by name.
        
        Args:
            index_name: Name of the index to load
            
        Raises:
            VectorStoreError: If index not found or loading fails
        """
        # Try to find the manifest file
        manifest_path = Path(self.cache_path) / f"{index_name}_manifest.json"
        if not manifest_path.exists():
            raise VectorStoreError(f"Index {index_name} not found")
            
        # Load manifest to get CID
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        index_cid = manifest.get('cid')
        if not index_cid:
            raise VectorStoreError(f"Invalid manifest for index {index_name}")
            
        # Load index metadata
        try:
            self._index_metadata = await self.storage.retrieve_index_metadata(index_cid)
            self._index_cid = index_cid
            self._index_name = index_name
            
            # Set dimension
            self.dimension = self._index_metadata.get('dimension', self.dimension)
        except Exception as e:
            raise VectorStoreError(f"Failed to load index {index_name}: {e}")
            
    async def rebuild_shards(self, index_name: Optional[str] = None) -> bool:
        """
        Rebuild index shards to optimize storage.
        
        Args:
            index_name: Name of the index (uses current if None)
            
        Returns:
            True if rebuild successful
        """
        # Implementation would rebalance shards for optimal distribution
        # Not fully implemented in this version
        return True
