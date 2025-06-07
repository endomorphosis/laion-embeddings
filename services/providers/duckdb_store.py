"""
DuckDB/Parquet Vector Store Provider

This module implements the DuckDB/Parquet vector store provider for the unified vector database architecture.
It leverages DuckDB for analytical queries and Parquet for efficient vector storage.
"""

import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import time
from datetime import datetime
import json

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

# Check for DuckDB availability
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Check for PyArrow/Parquet availability
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

logger = logging.getLogger(__name__)


class DuckDBVectorStore(BaseVectorStore):
    """
    DuckDB/Parquet Vector Store implementation.
    
    This implementation uses DuckDB for query processing and Parquet for efficient storage of vectors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DuckDB Vector Store with configuration.
        
        Args:
            config: DuckDB-specific configuration
        """
        super().__init__(config)
        self._client = None
        self._is_connected = False
        
        # Extract DuckDB specific config
        self.database_path = self.config.get('database_path', 'data/vector_store.duckdb')
        self.storage_path = os.path.dirname(self.database_path)
        self.default_table = self.config.get('table_name', 'embeddings')
        self.dimension = self.config.get('dimension', 512)
        self.distance_metric = self.config.get('distance', 'cosine')
        
        # Create storage directory if it doesn't exist
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Track indexes/tables
        self._indexes = {}
        self._current_index = None
        
        # Check dependencies
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required but not installed. Install with: pip install duckdb")
        if not ARROW_AVAILABLE:
            raise ImportError("PyArrow is required but not installed. Install with: pip install pyarrow")
    
    # Connection Management

    async def connect(self) -> None:
        """
        Establish connection to DuckDB.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Connect to DuckDB database
            self._client = duckdb.connect(self.database_path)
            
            # Create vector extension functions
            self._create_vector_functions()
            
            # Load existing indexes
            await self._load_indexes()
            
            self._is_connected = True
            logger.info(f"Connected to DuckDB at {self.database_path}")
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise ConnectionError(f"Failed to connect to DuckDB: {e}")
    
    async def disconnect(self) -> None:
        """Close connection to DuckDB."""
        if self._client:
            self._client.close()
            self._client = None
        self._is_connected = False
    
    async def ping(self) -> bool:
        """
        Test connection to DuckDB.
        
        Returns:
            True if connected
        """
        try:
            if not self._client:
                return False
            
            # Simple query to check connection
            self._client.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False
    
    # Index Management
    
    async def create_index(self, index_name: str, dimension: int, 
                          distance_metric: str = "cosine",
                          index_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new vector index (table in DuckDB).
        
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
            
            # Create the table
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {index_name} (
                id VARCHAR PRIMARY KEY,
                vector FLOAT[{dimension}],
                text VARCHAR,
                metadata JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self._client.execute(create_table_sql)
            
            # Create metadata file to track vector dimension and distance metric
            metadata = {
                'name': index_name,
                'dimension': dimension,
                'distance_metric': distance_metric,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'config': index_config or {},
                'vector_count': 0,
                'parquet_files': []
            }
            
            # Save metadata
            metadata_file = Path(self.storage_path) / f"{index_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Track the new index
            self._indexes[index_name] = metadata
            self._current_index = index_name
            
            logger.info(f"Created index {index_name} with dimension {dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise VectorStoreError(f"Failed to create index {index_name}: {e}")
    
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if index deleted successfully
        """
        try:
            if not self._is_connected:
                await self.connect()
                
            # Drop the table
            self._client.execute(f"DROP TABLE IF EXISTS {index_name}")
            
            # Delete associated Parquet files
            if index_name in self._indexes:
                for parquet_file in self._indexes[index_name].get('parquet_files', []):
                    parquet_path = Path(self.storage_path) / parquet_file
                    if parquet_path.exists():
                        parquet_path.unlink()
                        
            # Delete metadata file
            metadata_file = Path(self.storage_path) / f"{index_name}_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
                
            # Update tracking
            if index_name in self._indexes:
                del self._indexes[index_name]
                
            # If we deleted the current index, reset it
            if self._current_index == index_name:
                self._current_index = next(iter(self._indexes)) if self._indexes else None
                
            logger.info(f"Deleted index {index_name}")
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
                
            # Check if table exists in DuckDB
            result = self._client.execute(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{index_name}'
            """).fetchall()
            
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking if index {index_name} exists: {e}")
            return False
    
    async def list_indexes(self) -> List[str]:
        """
        List all available indexes.
        
        Returns:
            List of index names
        """
        try:
            if not self._is_connected:
                await self.connect()
                
            # Query DuckDB for tables
            result = self._client.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """).fetchall()
            
            # Extract table names
            tables = [row[0] for row in result if not row[0].startswith('sqlite')]
            
            # Update our local tracking
            for table in tables:
                if table not in self._indexes:
                    # Check for metadata file
                    metadata_file = Path(self.storage_path) / f"{table}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            self._indexes[table] = json.load(f)
            
            return tables
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
                index_name = self._current_index
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            # Get row count
            count_result = self._client.execute(f"SELECT COUNT(*) FROM {index_name}").fetchone()
            vector_count = count_result[0] if count_result else 0
            
            # Get metadata
            if index_name in self._indexes:
                metadata = self._indexes[index_name]
            else:
                metadata_file = Path(self.storage_path) / f"{index_name}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Create default metadata
                    metadata = {
                        'name': index_name,
                        'dimension': self.dimension,
                        'distance_metric': self.distance_metric,
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
            
            # Check parquet file sizes
            total_size = 0
            parquet_files = metadata.get('parquet_files', [])
            for parquet_file in parquet_files:
                file_path = Path(self.storage_path) / parquet_file
                if file_path.exists():
                    total_size += file_path.stat().st_size
            
            # Add table size from DuckDB
            # Note: This is an estimate as DuckDB doesn't directly expose table sizes
            db_size = Path(self.database_path).stat().st_size if Path(self.database_path).exists() else 0
            total_size += db_size / max(1, len(self._indexes))  # Rough estimate per table
            
            return IndexStats(
                total_vectors=vector_count,
                index_size_bytes=total_size,
                dimensions=metadata.get('dimension', self.dimension),
                distance_metric=metadata.get('distance_metric', self.distance_metric),
                created_at=metadata.get('created_at'),
                updated_at=metadata.get('updated_at'),
                additional_stats={
                    'parquet_files': len(parquet_files),
                    'table_name': index_name
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
                index_name = self._current_index
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
            
            # Ensure index exists
            if not await self.index_exists(index_name):
                dimension = self.dimension
                # If documents have vectors, use their dimension
                if documents and hasattr(documents[0].vector, '__len__'):
                    dimension = len(documents[0].vector)
                await self.create_index(index_name, dimension, self.distance_metric)
            
            # Prepare data for insert
            rows = []
            for doc in documents:
                # Convert metadata to JSON string
                metadata_json = json.dumps(doc.metadata) if doc.metadata else None
                
                # Add to rows
                rows.append({
                    'id': doc.id,
                    'vector': doc.vector,
                    'text': doc.text,
                    'metadata': metadata_json,
                    'timestamp': doc.timestamp or datetime.now().isoformat()
                })
            
            # Create PyArrow table
            arrays = []
            for field in ['id', 'vector', 'text', 'metadata', 'timestamp']:
                values = [row[field] for row in rows]
                if field == 'vector':
                    # Convert vectors to list of lists for PyArrow
                    vectors_list = [list(v) for v in values]
                    arrays.append(pa.FixedSizeListArray.from_arrays(
                        pa.array(np.array(vectors_list).flatten(), pa.float32()),
                        len(vectors_list[0])
                    ))
                elif field == 'metadata':
                    arrays.append(pa.array(values, pa.string()))
                elif field == 'timestamp':
                    arrays.append(pa.array(values, pa.string()))
                else:
                    arrays.append(pa.array(values))
            
            table = pa.Table.from_arrays(
                arrays,
                names=['id', 'vector', 'text', 'metadata', 'timestamp']
            )
            
            # Generate Parquet filename
            batch_id = int(time.time())
            parquet_filename = f"{index_name}_{batch_id}.parquet"
            parquet_path = Path(self.storage_path) / parquet_filename
            
            # Write to Parquet file
            pq.write_table(table, str(parquet_path))
            
            # Load data into DuckDB
            self._client.execute(f"""
                INSERT INTO {index_name} (id, vector, text, metadata, timestamp)
                SELECT id, vector, text, metadata::JSON, timestamp::TIMESTAMP
                FROM read_parquet('{parquet_path}')
            """)
            
            # Update metadata
            if index_name in self._indexes:
                metadata = self._indexes[index_name]
                metadata['vector_count'] += len(documents)
                metadata['updated_at'] = datetime.now().isoformat()
                if parquet_filename not in metadata['parquet_files']:
                    metadata['parquet_files'].append(parquet_filename)
                
                # Update the metadata file
                metadata_file = Path(self.storage_path) / f"{index_name}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
            else:
                # Load metadata first
                metadata_file = Path(self.storage_path) / f"{index_name}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Create new metadata
                    metadata = {
                        'name': index_name,
                        'dimension': self.dimension if documents else self.dimension,
                        'distance_metric': self.distance_metric,
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat(),
                        'vector_count': len(documents),
                        'parquet_files': [parquet_filename]
                    }
                    
                # Update the metadata file
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
                
                # Track the index
                self._indexes[index_name] = metadata
            
            # Set current index if not set
            if not self._current_index:
                self._current_index = index_name
                
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
        try:
            if not self._is_connected:
                await self.connect()
                
            if index_name is None:
                index_name = self._current_index
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            # Delete existing documents first
            document_ids = [doc.id for doc in documents]
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
                index_name = self._current_index
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            # Convert IDs to comma-separated string for SQL
            id_values = ', '.join([f"'{id}'" for id in ids])
            
            # Delete from table
            self._client.execute(f"DELETE FROM {index_name} WHERE id IN ({id_values})")
            
            # Update metadata
            if index_name in self._indexes:
                deleted_count = len(ids)
                metadata = self._indexes[index_name]
                metadata['vector_count'] = max(0, metadata['vector_count'] - deleted_count)
                metadata['updated_at'] = datetime.now().isoformat()
                
                # Update the metadata file
                metadata_file = Path(self.storage_path) / f"{index_name}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
            
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
                index_name = self._current_index
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
            
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
            
            # Ensure query vector is properly formatted for SQL
            query_vector = np.array(query.vector)
            query_vector_str = str(query_vector.tolist()).replace('[', '{').replace(']', '}')
            
            # Build SQL query based on distance metric
            if self.distance_metric.lower() == 'cosine':
                distance_sql = f"""
                (1 - (vector <*> CAST({query_vector_str} AS FLOAT[])) / 
                (SQRT(vector <*> vector) * SQRT(CAST({query_vector_str} AS FLOAT[]) <*> CAST({query_vector_str} AS FLOAT[]))))
                """
            elif self.distance_metric.lower() == 'euclidean':
                distance_sql = f"""
                SQRT(SUM((vector[i] - CAST({query_vector_str} AS FLOAT[])[i])^2) 
                for i in 0..LENGTH(vector)-1)
                """
            elif self.distance_metric.lower() == 'dot_product':
                # Negative dot product because we want highest dot product first
                distance_sql = f"-(vector <*> CAST({query_vector_str} AS FLOAT[]))"
            else:
                # Default to cosine
                distance_sql = f"""
                (1 - (vector <*> CAST({query_vector_str} AS FLOAT[])) / 
                (SQRT(vector <*> vector) * SQRT(CAST({query_vector_str} AS FLOAT[]) <*> CAST({query_vector_str} AS FLOAT[]))))
                """
            
            # Build filter clause if provided
            filter_clause = ""
            if query.filter_expr:
                filter_parts = []
                for key, value in query.filter_expr.items():
                    if isinstance(value, dict):
                        # Handle operators
                        for op, op_value in value.items():
                            if op == '$eq':
                                filter_parts.append(f"JSON_EXTRACT(metadata, '$.{key}') = '{op_value}'")
                            elif op == '$ne':
                                filter_parts.append(f"JSON_EXTRACT(metadata, '$.{key}') <> '{op_value}'")
                            elif op == '$gt':
                                filter_parts.append(f"CAST(JSON_EXTRACT(metadata, '$.{key}') AS FLOAT) > {op_value}")
                            elif op == '$gte':
                                filter_parts.append(f"CAST(JSON_EXTRACT(metadata, '$.{key}') AS FLOAT) >= {op_value}")
                            elif op == '$lt':
                                filter_parts.append(f"CAST(JSON_EXTRACT(metadata, '$.{key}') AS FLOAT) < {op_value}")
                            elif op == '$lte':
                                filter_parts.append(f"CAST(JSON_EXTRACT(metadata, '$.{key}') AS FLOAT) <= {op_value}")
                            elif op == '$in' and isinstance(op_value, list):
                                in_values = ', '.join([f"'{v}'" for v in op_value])
                                filter_parts.append(f"JSON_EXTRACT(metadata, '$.{key}') IN ({in_values})")
                    else:
                        # Direct value comparison
                        filter_parts.append(f"JSON_EXTRACT(metadata, '$.{key}') = '{value}'")
                
                if filter_parts:
                    filter_clause = "AND " + " AND ".join(filter_parts)
            
            # Apply similarity threshold if provided
            similarity_clause = ""
            if query.similarity_threshold is not None:
                if self.distance_metric.lower() == 'cosine':
                    # Convert similarity threshold to cosine distance threshold
                    distance_threshold = 1.0 - query.similarity_threshold
                    similarity_clause = f"AND {distance_sql} <= {distance_threshold}"
                elif self.distance_metric.lower() == 'dot_product':
                    # For dot product, higher is better (negative distance)
                    similarity_clause = f"AND {distance_sql} <= -{query.similarity_threshold}"
                else:
                    # For Euclidean, this is approximate
                    similarity_clause = f"AND {distance_sql} <= {1.0 - query.similarity_threshold}"
            
            # Build and execute search query
            search_sql = f"""
                SELECT 
                    id, 
                    {distance_sql} AS distance,
                    {"vector," if query.include_vectors else ""}
                    {"text," if query.include_metadata else ""}
                    {"metadata," if query.include_metadata else ""}
                    timestamp
                FROM {index_name}
                WHERE 1=1 {filter_clause} {similarity_clause}
                ORDER BY distance ASC
                LIMIT {query.limit} OFFSET {query.offset}
            """
            
            result = self._client.execute(search_sql).fetchall()
            
            # Build search results
            search_results = []
            for row in result:
                # Extract data from row
                id_val = row[0]
                distance = row[1]
                
                # Calculate index offsets based on include_vectors flag
                vector_idx = 2 if query.include_vectors else None
                text_idx = 3 if query.include_vectors else 2
                metadata_idx = text_idx + 1
                timestamp_idx = metadata_idx + 1
                
                # Map to search result
                vector_val = row[vector_idx] if vector_idx is not None else None
                text_val = row[text_idx] if query.include_metadata else None
                metadata_val = row[metadata_idx] if query.include_metadata else None
                
                # Parse metadata JSON
                if metadata_val:
                    metadata_dict = json.loads(metadata_val) if isinstance(metadata_val, str) else metadata_val
                else:
                    metadata_dict = None
                
                # Convert distance to score (higher is better)
                if self.distance_metric.lower() == 'cosine':
                    score = 1.0 - distance
                elif self.distance_metric.lower() == 'euclidean':
                    # Normalize Euclidean distance to [0,1] score range
                    # This is an approximation
                    max_distance = np.sqrt(self.dimension * 2)  # Max possible Euclidean distance for normalized vectors
                    score = 1.0 - (distance / max_distance)
                elif self.distance_metric.lower() == 'dot_product':
                    # For dot product, we stored negative distance
                    score = -distance
                else:
                    score = 1.0 - distance
                
                search_results.append(SearchResult(
                    id=id_val,
                    score=float(score),
                    distance=float(distance),
                    vector=vector_val.tolist() if vector_val is not None else None,
                    text=text_val,
                    metadata=metadata_dict
                ))
            
            return search_results
        except Exception as e:
            logger.error(f"Failed to search in index {index_name}: {e}")
            raise VectorStoreError(f"Failed to search: {e}")
    
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
                    message="Not connected to DuckDB",
                    response_time_ms=0,
                    last_check=datetime.now().isoformat()
                )
                
            # Check DuckDB with simple query
            self._client.execute("SELECT 1").fetchone()
            
            response_time = (time.time() - start_time) * 1000
            
            # Get DuckDB version
            version = self._client.execute("PRAGMA version").fetchone()[0]
            
            return HealthStatus(
                status=VectorStoreStatus.HEALTHY,
                message=f"Connected to DuckDB v{version}",
                response_time_ms=response_time,
                last_check=datetime.now().isoformat(),
                details={
                    'version': version,
                    'database_path': self.database_path,
                    'current_index': self._current_index,
                    'indexes': list(self._indexes.keys())
                }
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                status=VectorStoreStatus.UNHEALTHY,
                message=f"DuckDB health check failed: {e}",
                response_time_ms=response_time,
                last_check=datetime.now().isoformat()
            )
    
    # Helper Methods
    
    def _create_vector_functions(self) -> None:
        """Set up DuckDB vector functions for similarity calculations."""
        # Create dot product function (<*>) if not already available
        try:
            self._client.execute("""
                -- Create dot product function for vectors
                CREATE OR REPLACE FUNCTION dot_product(a FLOAT[], b FLOAT[]) AS
                    SUM(a[i] * b[i] for i in 0..LENGTH(a)-1);
                
                -- Register dot product as operator
                CREATE OR REPLACE MACRO <*>(a, b) AS dot_product(a, b);
                
                -- Create Euclidean distance function
                CREATE OR REPLACE FUNCTION euclidean_distance(a FLOAT[], b FLOAT[]) AS
                    SQRT(SUM((a[i] - b[i])^2 for i in 0..LENGTH(a)-1));
                
                -- Create cosine distance function
                CREATE OR REPLACE FUNCTION cosine_distance(a FLOAT[], b FLOAT[]) AS
                    1 - (dot_product(a, b) / (SQRT(dot_product(a, a)) * SQRT(dot_product(b, b))));
            """)
        except Exception as e:
            logger.warning(f"Failed to create vector functions: {e}")
    
    async def _load_indexes(self) -> None:
        """Load indexes from storage."""
        # Scan metadata files
        storage_dir = Path(self.storage_path)
        metadata_files = storage_dir.glob("*_metadata.json")
        
        for metadata_file in metadata_files:
            try:
                # Extract index name from filename
                index_name = metadata_file.stem.replace("_metadata", "")
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Track the index
                self._indexes[index_name] = metadata
                
                # Set current index if not set
                if not self._current_index:
                    self._current_index = index_name
            except Exception as e:
                logger.warning(f"Failed to load index metadata from {metadata_file}: {e}")
    
    async def optimize(self, index_name: Optional[str] = None) -> bool:
        """
        Optimize the index by compacting Parquet files.
        
        Args:
            index_name: Name of the index to optimize (uses current if None)
            
        Returns:
            True if optimization was successful
        """
        try:
            if not self._is_connected:
                await self.connect()
                
            if index_name is None:
                index_name = self._current_index
                
            if not index_name:
                raise VectorStoreError("No index specified and no current index")
                
            if index_name not in self._indexes:
                raise VectorStoreError(f"Index {index_name} not found")
                
            metadata = self._indexes[index_name]
            parquet_files = metadata.get('parquet_files', [])
            
            if len(parquet_files) <= 1:
                logger.info(f"No optimization needed for index {index_name}")
                return True
            
            # Create a new consolidated Parquet file
            timestamp = int(time.time())
            new_filename = f"{index_name}_optimized_{timestamp}.parquet"
            new_path = Path(self.storage_path) / new_filename
            
            # Load all data and save to new file
            self._client.execute(f"""
                COPY (SELECT * FROM {index_name}) TO '{new_path}' (FORMAT PARQUET)
            """)
            
            # Register the new file and clear old ones
            metadata['parquet_files'] = [new_filename]
            metadata['updated_at'] = datetime.now().isoformat()
            
            # Save updated metadata
            metadata_file = Path(self.storage_path) / f"{index_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Delete old files
            for old_file in parquet_files:
                old_path = Path(self.storage_path) / old_file
                if old_path.exists():
                    old_path.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Failed to optimize index {index_name}: {e}")
            raise VectorStoreError(f"Failed to optimize index: {e}")
