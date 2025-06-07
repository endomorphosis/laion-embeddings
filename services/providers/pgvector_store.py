"""
pgvector (PostgreSQL) vector store implementation for unified vector database support.

This module provides a PostgreSQL with pgvector extension implementation of the BaseVectorStore interface,
enabling seamless integration with the unified vector service architecture.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json

from ..vector_store_base import (
    BaseVectorStore, 
    SearchResult, 
    VectorDocument,
    IndexStats,
    HealthStatus
)

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 not available. Install with: pip install psycopg2-binary")
    PSYCOPG2_AVAILABLE = False
    # Mock classes for when psycopg2 is not available
    class RealDictCursor:
        pass


class PgVectorStore(BaseVectorStore):
    """PostgreSQL pgvector implementation of BaseVectorStore."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pgvector vector store.
        
        Args:
            config: Configuration dictionary containing:
                - host: PostgreSQL server host (default: localhost)
                - port: PostgreSQL server port (default: 5432)
                - database: Database name
                - username: Username for authentication
                - password: Password for authentication
                - table_name: Name of the vectors table
                - vector_column: Name of the vector column (default: 'embedding')
                - vector_size: Dimensionality of vectors
                - distance_metric: Distance metric (cosine, l2, inner_product)
                - pool_size: Connection pool size
                - timeout: Connection timeout
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")
        
        super().__init__(config)
        
        # PostgreSQL-specific configuration
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 5432)
        self.database = config.get('database', 'embeddings')
        self.username = config.get('username', 'postgres')
        self.password = config.get('password', '')
        self.table_name = config.get('table_name', 'vectors')
        self.vector_column = config.get('vector_column', 'embedding')
        self.vector_size = config.get('vector_size', 768)
        self.distance_metric = config.get('distance_metric', 'cosine')
        self.timeout = config.get('timeout', 30)
        
        # Connection
        self.connection = None
        self._connected = False
        
        # Distance metric operators
        self._distance_ops = {
            'cosine': '<=>',
            'l2': '<->',
            'inner_product': '<#>',
            'euclidean': '<->'
        }
    
    async def connect(self) -> None:
        """Connect to PostgreSQL server."""
        try:
            # Create connection
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                connect_timeout=self.timeout
            )
            
            # Test connection
            if not await self.ping():
                raise ConnectionError("Unable to connect to PostgreSQL")
            
            # Ensure pgvector extension and table exist
            await self._ensure_pgvector_setup()
            
            self._connected = True
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL server."""
        if self.connection:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error during PostgreSQL disconnect: {e}")
            finally:
                self.connection = None
                self._connected = False
    
    async def ping(self) -> bool:
        """Test connection to PostgreSQL server."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"PostgreSQL ping failed: {e}")
            return False
    
    async def _ensure_pgvector_setup(self) -> None:
        """Ensure pgvector extension and table exist."""
        try:
            with self.connection.cursor() as cursor:
                # Create pgvector extension if not exists
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    )
                """, (self.table_name,))
                
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    # Create vectors table
                    create_table_sql = f"""
                        CREATE TABLE {self.table_name} (
                            id TEXT PRIMARY KEY,
                            {self.vector_column} vector({self.vector_size}),
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    cursor.execute(create_table_sql)
                    
                    # Create vector index
                    index_name = f"{self.table_name}_{self.vector_column}_idx"
                    if self.distance_metric == 'cosine':
                        index_sql = f"""
                            CREATE INDEX {index_name} ON {self.table_name} 
                            USING ivfflat ({self.vector_column} vector_cosine_ops)
                            WITH (lists = 100)
                        """
                    elif self.distance_metric in ['l2', 'euclidean']:
                        index_sql = f"""
                            CREATE INDEX {index_name} ON {self.table_name} 
                            USING ivfflat ({self.vector_column} vector_l2_ops)
                            WITH (lists = 100)
                        """
                    else:  # inner_product
                        index_sql = f"""
                            CREATE INDEX {index_name} ON {self.table_name} 
                            USING ivfflat ({self.vector_column} vector_ip_ops)
                            WITH (lists = 100)
                        """
                    
                    cursor.execute(index_sql)
                    
                    logger.info(f"Created pgvector table: {self.table_name}")
                
                self.connection.commit()
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to ensure pgvector setup: {e}")
            raise
    
    async def add_vectors(self, vectors: List[List[float]], 
                         metadata: Optional[List[Dict[str, Any]]] = None,
                         ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the pgvector table.
        
        Args:
            vectors: List of vector embeddings
            metadata: Optional metadata for each vector
            ids: Optional IDs for vectors
            
        Returns:
            List of assigned vector IDs
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to PostgreSQL")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}" 
                   for i in range(len(vectors))]
        
        try:
            with self.connection.cursor() as cursor:
                # Prepare data for insertion
                data = []
                for i, vector in enumerate(vectors):
                    vector_str = '[' + ','.join(map(str, vector)) + ']'
                    metadata_json = json.dumps(metadata[i]) if metadata and i < len(metadata) else '{}'
                    data.append((ids[i], vector_str, metadata_json))
                
                # Bulk insert
                insert_sql = f"""
                    INSERT INTO {self.table_name} (id, {self.vector_column}, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        {self.vector_column} = EXCLUDED.{self.vector_column},
                        metadata = EXCLUDED.metadata
                """
                
                execute_values(cursor, insert_sql, data, template=None)
                self.connection.commit()
                
                logger.info(f"Added {len(vectors)} vectors to pgvector table {self.table_name}")
                return ids
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to add vectors to pgvector: {e}")
            raise
    
    async def search_vectors(self, query_vector: List[float], 
                           limit: int = 10,
                           filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar vectors in pgvector.
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to PostgreSQL")
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build query vector string
                query_vector_str = '[' + ','.join(map(str, query_vector)) + ']'
                
                # Get distance operator
                distance_op = self._distance_ops.get(self.distance_metric, '<=>')
                
                # Build base query
                select_sql = f"""
                    SELECT id, metadata, {self.vector_column} {distance_op} %s as distance
                    FROM {self.table_name}
                """
                
                params = [query_vector_str]
                
                # Add filters if provided
                if filters:
                    where_clauses, filter_params = self._build_pg_filter(filters)
                    if where_clauses:
                        select_sql += " WHERE " + " AND ".join(where_clauses)
                        params.extend(filter_params)
                
                # Add ordering and limit
                select_sql += f" ORDER BY {self.vector_column} {distance_op} %s LIMIT %s"
                params.extend([query_vector_str, limit])
                
                # Execute query
                cursor.execute(select_sql, params)
                results = cursor.fetchall()
                
                # Convert results
                search_results = []
                for row in results:
                    result = SearchResult(
                        id=row['id'],
                        score=float(row['distance']),  # Lower distance = higher similarity
                        metadata=row['metadata'] or {},
                        vector=None  # Not returned for performance
                    )
                    search_results.append(result)
                
                return search_results
                
        except Exception as e:
            logger.error(f"Failed to search vectors in pgvector: {e}")
            raise
    
    def _build_pg_filter(self, filters: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        """Build PostgreSQL filter clauses from generic filter dictionary."""
        where_clauses = []
        params = []
        
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                where_clauses.append("metadata ->> %s = %s")
                params.extend([key, str(value)])
            elif isinstance(value, list):
                placeholders = ','.join(['%s'] * len(value))
                where_clauses.append(f"metadata ->> %s IN ({placeholders})")
                params.append(key)
                params.extend(value)
            elif isinstance(value, dict):
                # Handle range queries
                if 'gte' in value:
                    where_clauses.append("(metadata ->> %s)::numeric >= %s")
                    params.extend([key, value['gte']])
                if 'lte' in value:
                    where_clauses.append("(metadata ->> %s)::numeric <= %s")
                    params.extend([key, value['lte']])
                if 'gt' in value:
                    where_clauses.append("(metadata ->> %s)::numeric > %s")
                    params.extend([key, value['gt']])
                if 'lt' in value:
                    where_clauses.append("(metadata ->> %s)::numeric < %s")
                    params.extend([key, value['lt']])
        
        return where_clauses, params
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors from pgvector.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to PostgreSQL")
        
        try:
            with self.connection.cursor() as cursor:
                # Bulk delete
                placeholders = ','.join(['%s'] * len(ids))
                delete_sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
                cursor.execute(delete_sql, ids)
                
                deleted_count = cursor.rowcount
                self.connection.commit()
                
                logger.info(f"Deleted {deleted_count} vectors from pgvector table {self.table_name}")
                return True
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to delete vectors from pgvector: {e}")
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
            raise RuntimeError("Not connected to PostgreSQL")
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                select_sql = f"""
                    SELECT id, {self.vector_column}, metadata
                    FROM {self.table_name}
                    WHERE id = %s
                """
                cursor.execute(select_sql, (vector_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Convert vector string back to list
                vector_str = row[self.vector_column]
                vector = [float(x) for x in vector_str.strip('[]').split(',')]
                
                return VectorDocument(
                    id=row['id'],
                    vector=vector,
                    metadata=row['metadata'] or {}
                )
                
        except Exception as e:
            logger.error(f"Failed to get vector from pgvector: {e}")
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
            raise RuntimeError("Not connected to PostgreSQL")
        
        try:
            with self.connection.cursor() as cursor:
                # Build update clauses
                update_clauses = []
                params = []
                
                if vector is not None:
                    vector_str = '[' + ','.join(map(str, vector)) + ']'
                    update_clauses.append(f"{self.vector_column} = %s")
                    params.append(vector_str)
                
                if metadata is not None:
                    update_clauses.append("metadata = %s")
                    params.append(json.dumps(metadata))
                
                if not update_clauses:
                    return False
                
                # Execute update
                update_sql = f"""
                    UPDATE {self.table_name}
                    SET {', '.join(update_clauses)}
                    WHERE id = %s
                """
                params.append(vector_id)
                
                cursor.execute(update_sql, params)
                updated_count = cursor.rowcount
                self.connection.commit()
                
                return updated_count > 0
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to update vector in pgvector: {e}")
            return False
    
    async def get_index_stats(self) -> IndexStats:
        """Get statistics about the pgvector table."""
        if not self.is_connected:
            raise RuntimeError("Not connected to PostgreSQL")
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name}")
                count_result = cursor.fetchone()
                total_vectors = count_result['count'] if count_result else 0
                
                # Get table size
                cursor.execute(f"""
                    SELECT pg_total_relation_size('{self.table_name}') as size_bytes
                """)
                size_result = cursor.fetchone()
                size_bytes = size_result['size_bytes'] if size_result else 0
                
                return IndexStats(
                    total_vectors=total_vectors,
                    index_size_bytes=size_bytes,
                    dimensions=self.vector_size,
                    last_updated=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Failed to get pgvector index stats: {e}")
            return IndexStats(
                total_vectors=0,
                index_size_bytes=0,
                dimensions=self.vector_size,
                last_updated=datetime.now()
            )
    
    async def get_health_status(self) -> HealthStatus:
        """Get health status of the pgvector connection."""
        is_healthy = await self.ping()
        
        status = HealthStatus(
            is_healthy=is_healthy,
            database_type="pgvector",
            connection_info={
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "table": self.table_name
            },
            last_check=datetime.now()
        )
        
        if is_healthy:
            try:
                stats = await self.get_index_stats()
                status.additional_info = {
                    "total_vectors": stats.total_vectors,
                    "table_size_bytes": stats.index_size_bytes,
                    "dimensions": stats.dimensions,
                    "distance_metric": self.distance_metric
                }
            except Exception as e:
                status.additional_info = {"stats_error": str(e)}
        
        return status
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to PostgreSQL."""
        return self._connected and self.connection is not None
    
    @property
    def database_type(self) -> str:
        """Get database type identifier."""
        return "pgvector"
