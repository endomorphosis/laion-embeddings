# DuckDB Vector Store Examples

This document provides practical examples for using the DuckDB Vector Store in the unified vector database architecture.

## Basic Usage

### Connecting to DuckDB and Creating an Index

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def basic_duckdb_example():
    # Create a DuckDB vector store
    store = await create_vector_store(VectorDBType.DUCKDB)
    
    # Connect to DuckDB
    await store.connect()
    
    # Check connection
    is_connected = await store.ping()
    print(f"DuckDB connected: {is_connected}")
    
    # Create a vector index with dimension 384
    created = await store.create_index(
        dimension=384,
        overwrite=True  # Replace any existing index
    )
    print(f"Index created: {created}")
    
    # Get index stats
    stats = await store.get_stats()
    print(f"Index stats: {stats}")
    
    # Disconnect when done
    await store.disconnect()

# Run the example
asyncio.run(basic_duckdb_example())
```

### Adding and Searching Vectors

```python
import asyncio
import numpy as np
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def add_search_example():
    # Create and connect to store
    store = await create_vector_store(VectorDBType.DUCKDB)
    await store.connect()
    
    # Dimension for our vectors
    dim = 384
    
    # Create index
    await store.create_index(dimension=dim)
    
    # Create sample documents with random vectors
    num_docs = 100
    docs = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_docs):
        # Create a random vector and normalize it
        vec = np.random.random(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Create document
        doc = VectorDocument(
            id=f"doc-{i}",
            vector=vec.tolist(),
            metadata={
                "title": f"Document {i}",
                "category": f"cat-{i % 5}",
                "value": float(i),
                "date": f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
            }
        )
        docs.append(doc)
    
    # Add documents to DuckDB
    print(f"Adding {len(docs)} documents...")
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Check index stats
    stats = await store.get_stats()
    print(f"Index stats after adding: {stats}")
    
    # Create a query vector (use first document's vector)
    query_vector = docs[0].vector
    
    # Basic search
    query = SearchQuery(
        vector=query_vector,
        top_k=5
    )
    
    print("\nBasic search results:")
    results = await store.search(query)
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
    
    # Search with filter
    filter_query = SearchQuery(
        vector=query_vector,
        top_k=5,
        filter={"category": "cat-2"}  # Only cat-2 documents
    )
    
    print("\nFiltered search results (category=cat-2):")
    filtered_results = await store.search(filter_query)
    for i, match in enumerate(filtered_results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
        print(f"   Category: {match.metadata.get('category')}")
    
    # Clean up
    await store.disconnect()

# Run the example
asyncio.run(add_search_example())
```

## Advanced Usage

### SQL-Based Filtering in DuckDB

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery
import numpy as np

async def sql_filtering_example():
    # Create DuckDB store
    store = await create_vector_store(
        VectorDBType.DUCKDB,
        config_override={
            "dimension": 64,
            "database_path": "data/sql_example.duckdb",
            "table_name": "sql_vectors"
        }
    )
    
    await store.connect()
    await store.create_index(dimension=64)
    
    # Create sample documents with date and numeric fields
    docs = []
    np.random.seed(42)
    
    for i in range(100):
        vec = np.random.random(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Create document with rich metadata for SQL filtering
        doc = VectorDocument(
            id=f"sql-doc-{i}",
            vector=vec.tolist(),
            metadata={
                "title": f"Document {i}",
                "category": f"cat-{i % 5}",
                "score": round(np.random.random() * 100) / 100,  # Random score 0-1
                "price": round(10 + np.random.random() * 90, 2),  # Price $10-$100
                "date": f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",  # Random date
                "in_stock": i % 3 == 0,  # Boolean field
                "tags": [f"tag{j}" for j in range(i % 5 + 1)]  # Array field
            }
        )
        docs.append(doc)
    
    # Add documents
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Query vector
    query_vector = docs[0].vector
    
    # Example 1: Basic filter
    print("\nExample 1: Filter by category")
    results = await store.search(
        SearchQuery(
            vector=query_vector,
            top_k=5,
            filter={"category": "cat-2"}
        )
    )
    print_results(results, include_metadata=True)
    
    # Example 2: SQL filter on numeric value
    print("\nExample 2: SQL filter on price > 50")
    if hasattr(store, 'search_with_sql_filter'):
        results = await store.search_with_sql_filter(
            vector=query_vector,
            top_k=5,
            sql_filter="CAST(json_extract(metadata_json, '$.price') AS DOUBLE) > 50"
        )
        print_results(results, include_metadata=True)
    else:
        # Use regular search with embedded SQL filter
        results = await store.search(
            SearchQuery(
                vector=query_vector,
                top_k=5,
                sql_filter="CAST(json_extract(metadata_json, '$.price') AS DOUBLE) > 50"
            )
        )
        print_results(results, include_metadata=True)
    
    # Example 3: Complex SQL filter with date and numeric conditions
    print("\nExample 3: Date after 2025-06 AND score > 0.5")
    if hasattr(store, 'search_with_sql_filter'):
        results = await store.search_with_sql_filter(
            vector=query_vector,
            top_k=5,
            sql_filter="CAST(json_extract(metadata_json, '$.date') AS VARCHAR) > '2025-06-01' AND "
                      "CAST(json_extract(metadata_json, '$.score') AS DOUBLE) > 0.5"
        )
        print_results(results, include_metadata=True)
    else:
        results = await store.search(
            SearchQuery(
                vector=query_vector,
                top_k=5,
                sql_filter="CAST(json_extract(metadata_json, '$.date') AS VARCHAR) > '2025-06-01' AND "
                          "CAST(json_extract(metadata_json, '$.score') AS DOUBLE) > 0.5"
            )
        )
        print_results(results, include_metadata=True)
    
    # Clean up
    await store.disconnect()

def print_results(results, include_metadata=False):
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
        if include_metadata:
            # Print selected metadata fields
            meta = match.metadata
            price = meta.get('price', 'N/A')
            date = meta.get('date', 'N/A')
            category = meta.get('category', 'N/A')
            score = meta.get('score', 'N/A')
            print(f"   Category: {category}, Price: ${price}, Date: {date}, Score: {score}")

# Run the example
asyncio.run(sql_filtering_example())
```

### Sharding with DuckDB

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery
import numpy as np

async def sharding_example():
    # Create DuckDB store with sharding configuration
    store = await create_vector_store(
        VectorDBType.DUCKDB,
        config_override={
            "dimension": 64,
            "database_path": "data/sharded.duckdb",
            "storage_path": "data/sharded_parquet",
            "sharding_enabled": True,
            "max_shard_size": 25  # Small shard size for demonstration
        }
    )
    
    await store.connect()
    
    # Create index with sharding
    await store.create_index(
        dimension=64,
        sharding=True,
        sharding_params={"max_shard_size": 25}
    )
    
    # Create 100 sample documents (to demonstrate sharding)
    docs = []
    np.random.seed(42)
    
    for i in range(100):
        vec = np.random.random(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        doc = VectorDocument(
            id=f"shard-doc-{i}",
            vector=vec.tolist(),
            metadata={"group": f"group-{i // 10}"}
        )
        docs.append(doc)
    
    # Add documents (will be automatically sharded)
    print("Adding documents with sharding...")
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Check stats to see sharding info
    stats = await store.get_stats()
    print(f"Shard count: {stats.get('shard_count', 0)}")
    print(f"Total vectors: {stats.get('total_vectors', 0)}")
    
    # Search across shards
    query_vector = docs[50].vector  # Use a random document's vector
    
    query = SearchQuery(
        vector=query_vector,
        top_k=5
    )
    
    print("\nCross-shard search results:")
    results = await store.search(query)
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
    
    # Clean up
    await store.disconnect()

# Run the example
asyncio.run(sharding_example())
```

### Batch Operations with DuckDB

```python
import asyncio
import numpy as np
import time
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument

async def batch_operations_example():
    # Create DuckDB store
    store = await create_vector_store(
        VectorDBType.DUCKDB,
        config_override={
            "dimension": 128,
            "database_path": "data/batch_ops.duckdb",
            "storage_path": "data/batch_parquet",
            "performance": {
                "batch_size": 1000  # Process in batches of 1000
            }
        }
    )
    
    await store.connect()
    await store.create_index(dimension=128)
    
    # Generate a large number of vectors
    print("Generating vectors...")
    num_vectors = 10000
    np.random.seed(42)
    
    # Test adding in different batch sizes
    batch_sizes = [100, 500, 1000, 2000, 5000]
    
    for batch_size in batch_sizes:
        # Generate vectors for this test
        docs = []
        for i in range(num_vectors):
            vec = np.random.random(128).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            
            doc = VectorDocument(
                id=f"batch-{batch_size}-doc-{i}",
                vector=vec.tolist(),
                metadata={"batch_size": batch_size}
            )
            docs.append(doc)
        
        # Clear previous index
        await store.create_index(dimension=128, overwrite=True)
        
        # Measure time for adding in batches
        print(f"\nTesting batch size: {batch_size}")
        start_time = time.time()
        
        # If the store supports batch processing directly
        if hasattr(store, 'add_documents_in_batches'):
            count = await store.add_documents_in_batches(docs, batch_size=batch_size)
        else:
            # Manual batch processing
            count = 0
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                batch_count = await store.add_documents(batch)
                count += batch_count
                print(f"  Added batch {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size}: {batch_count} vectors")
        
        elapsed = time.time() - start_time
        print(f"Total: {count} vectors in {elapsed:.2f} seconds")
        print(f"Rate: {count/elapsed:.2f} vectors/second")
    
    # Clean up
    await store.disconnect()

# Run the example
asyncio.run(batch_operations_example())
```

## Working with DuckDB/Parquet Directly

DuckDB's combination with Parquet allows for advanced analytical queries that go beyond simple vector similarity. This example shows how to use DuckDB's SQL capabilities directly with the vector store.

```python
import asyncio
import duckdb
import os
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery
import numpy as np

async def advanced_duckdb_example():
    # Settings for our store
    db_path = "data/advanced_duckdb.db"
    parquet_path = "data/advanced_parquet"
    table_name = "embeddings"
    dimension = 64
    
    # Create directories if they don't exist
    os.makedirs(parquet_path, exist_ok=True)
    
    # Create store and populate with data
    store = await create_vector_store(
        VectorDBType.DUCKDB,
        config_override={
            "dimension": dimension,
            "database_path": db_path,
            "storage_path": parquet_path,
            "table_name": table_name
        }
    )
    
    await store.connect()
    await store.create_index(dimension=dimension)
    
    # Create sample data with categories and values
    docs = []
    np.random.seed(42)
    
    categories = ["technology", "science", "health", "business", "entertainment"]
    
    for i in range(500):
        vec = np.random.random(dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        category = categories[i % len(categories)]
        value = np.random.randint(1, 101)
        
        doc = VectorDocument(
            id=f"adv-doc-{i}",
            vector=vec.tolist(),
            metadata={
                "category": category,
                "value": value,
                "even": (i % 2 == 0)
            }
        )
        docs.append(doc)
    
    # Add documents
    print("Adding documents...")
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Disconnect from the store
    await store.disconnect()
    
    # Now connect directly to DuckDB for advanced analytics
    print("\nPerforming advanced DuckDB analytics...")
    conn = duckdb.connect(db_path)
    
    # List parquet files
    files_df = conn.execute(f"SELECT * FROM glob('{parquet_path}/*.parquet')").fetchdf()
    print(f"Found {len(files_df)} parquet files")
    
    # Query 1: Average value by category
    print("\nAverage value by category:")
    result = conn.execute(f"""
        SELECT 
            json_extract(metadata_json, '$.category') AS category,
            AVG(CAST(json_extract(metadata_json, '$.value') AS INTEGER)) AS avg_value,
            COUNT(*) AS count
        FROM '{parquet_path}/*.parquet'
        GROUP BY category
        ORDER BY avg_value DESC
    """).fetchall()
    
    for row in result:
        print(f"{row[0]}: {row[1]:.2f} (count: {row[2]})")
    
    # Query 2: Advanced vector statistics
    print("\nVector statistics:")
    result = conn.execute(f"""
        SELECT
            COUNT(*) AS vector_count,
            AVG(list_average(vector)) AS avg_vector_mean,
            MIN(list_min(vector)) AS min_value,
            MAX(list_max(vector)) AS max_value
        FROM '{parquet_path}/*.parquet'
    """).fetchall()
    
    for row in result:
        print(f"Count: {row[0]}, Avg mean: {row[1]:.4f}, Min: {row[2]:.4f}, Max: {row[3]:.4f}")
    
    # Query 3: Find vectors where a specific dimension is above average
    dim_index = 5  # Choose a dimension to analyze
    print(f"\nDocuments where dimension {dim_index} is above average:")
    
    result = conn.execute(f"""
        WITH stats AS (
            SELECT AVG(vector[{dim_index+1}]) AS avg_dim
            FROM '{parquet_path}/*.parquet'
        )
        SELECT 
            id,
            json_extract(metadata_json, '$.category') AS category,
            vector[{dim_index+1}] AS dim_value
        FROM '{parquet_path}/*.parquet', stats
        WHERE vector[{dim_index+1}] > stats.avg_dim
        ORDER BY dim_value DESC
        LIMIT 5
    """).fetchall()
    
    for row in result:
        print(f"ID: {row[0]}, Category: {row[1]}, Value at dim {dim_index}: {row[2]:.4f}")
    
    # Clean up
    conn.close()

# Run the example
asyncio.run(advanced_duckdb_example())
```

## Production Usage Patterns

### Resilient Connection Management

```python
import asyncio
import logging
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import ConnectionError, VectorStoreError

async def production_duckdb_example():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("duckdb-example")
    
    # Connection parameters
    db_path = "data/production.duckdb"
    parquet_path = "data/production_parquet"
    
    # Connection management class
    class DuckDBManager:
        def __init__(self, db_path, parquet_path):
            self.db_path = db_path
            self.parquet_path = parquet_path
            self.store = None
            self.connected = False
            
        async def get_store(self):
            """Get connected store or create new connection."""
            if self.store is not None and self.connected:
                return self.store
                
            # Create new connection
            try:
                logger.info("Connecting to DuckDB...")
                self.store = await create_vector_store(
                    VectorDBType.DUCKDB,
                    config_override={
                        "database_path": self.db_path,
                        "storage_path": self.parquet_path,
                        "table_name": "embeddings"
                    }
                )
                await self.store.connect()
                
                # Verify connection
                if await self.store.ping():
                    self.connected = True
                    logger.info("Successfully connected to DuckDB")
                    return self.store
                else:
                    logger.error("Connection failed: ping returned false")
                    self.connected = False
                    return None
                    
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.connected = False
                return None
        
        async def close(self):
            """Safely close the connection."""
            if self.store and self.connected:
                try:
                    await self.store.disconnect()
                    logger.info("Disconnected from DuckDB")
                except Exception as e:
                    logger.error(f"Error during disconnect: {e}")
                finally:
                    self.connected = False
    
    # Create manager
    manager = DuckDBManager(db_path, parquet_path)
    
    # Example usage in a production-like scenario
    max_retries = 3
    
    for attempt in range(max_retries):
        # Get connected store
        store = await manager.get_store()
        
        if not store:
            if attempt < max_retries - 1:
                logger.info(f"Retry {attempt+1}/{max_retries} in 1 second...")
                await asyncio.sleep(1)
                continue
            else:
                logger.error("All connection attempts failed")
                break
        
        try:
            # Perform operations with proper error handling
            
            # 1. Get stats
            try:
                stats = await store.get_stats()
                logger.info(f"Current stats: {stats}")
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
            
            # 2. Create/verify index
            try:
                dimension = 128
                index_created = await store.create_index(
                    dimension=dimension,
                    create_if_not_exists=True
                )
                if index_created:
                    logger.info("Created new index")
                else:
                    logger.info("Index already exists")
            except Exception as e:
                logger.error(f"Index creation failed: {e}")
                break
            
            # 3. Perform some vector operations
            # ... your production operations here ...
            
            # Success, break retry loop
            break
            
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            # Try to reconnect on next attempt
            await manager.close()
            if attempt < max_retries - 1:
                logger.info(f"Will retry operation attempt {attempt+1}/{max_retries}")
            else:
                logger.error("All operation attempts failed")
    
    # Always clean up
    await manager.close()

# Run the example
asyncio.run(production_duckdb_example())
```

## For More Information

- [DuckDB Vector Service Documentation](../duckdb-vector-service.md) - Complete guide to DuckDB integration
- [Vector Stores Overview](../vector-stores.md) - Introduction to the vector store architecture
