# DuckDB Vector Service

The DuckDB Vector Service provides an analytical database approach to vector storage and retrieval using DuckDB and Apache Parquet. This combination offers efficient columnar storage and powerful SQL-based querying capabilities for vector embeddings.

## Overview

The DuckDB Vector Service leverages DuckDB's analytical processing capabilities combined with Parquet's efficient columnar storage format to provide a lightweight yet powerful vector database solution. This approach is particularly well-suited for analytical workloads, batch processing, and environments where traditional vector databases may be too resource-intensive.

## Features

- **Analytical Processing**: Leverage DuckDB's analytical capabilities for vector operations
- **Columnar Storage**: Efficient storage of vectors using Apache Parquet format
- **SQL Querying**: Use SQL expressions for advanced filtering and vector operations
- **Low Resource Usage**: Minimal memory footprint compared to dedicated vector databases
- **Batch Processing**: Efficient processing of large vector batches
- **Portable Storage**: Parquet files can be used by many other systems
- **No Server Required**: Serverless architecture with file-based storage

## Architecture

The DuckDB Vector Service integrates with the unified vector store architecture as follows:

```
┌─────────────────────┐      ┌─────────────────────┐
│  Application Code   │      │  Vector Store API   │
└──────────┬──────────┘      └──────────┬──────────┘
           │                            │
           ▼                            ▼
┌─────────────────────────────────────────────────┐
│            Vector Store Factory                 │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            DuckDBVectorStore                    │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────┐      ┌─────────────────────┐
│      DuckDB         │◄────►│   Parquet Files     │
└─────────────────────┘      └─────────────────────┘
```

## Implementation

The DuckDB Vector Service is implemented in the `DuckDBVectorStore` class, which implements the `BaseVectorStore` interface for the unified architecture.

## Dependencies

The DuckDB Vector Service requires the following libraries:

- **duckdb**: DuckDB database engine
- **pyarrow**: Apache Arrow and Parquet libraries

## Configuration

The DuckDB Vector Store is configured in `config/vector_databases.yaml` under the `duckdb` section:

```yaml
databases:
  duckdb:
    enabled: true
    database_path: "data/vectors.duckdb"
    storage_path: "data/vector_parquet"
    table_name: "embeddings"
    dimension: 768
    index:
      create_index: true
      index_type: "L2"
    search:
      similarity_metric: "l2"
      batch_size: 1000
    performance:
      memory_limit: "4GB"
      threads: 4
```

## Usage Examples

### Basic Usage

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def main():
    # Create DuckDB store
    store = await create_vector_store(VectorDBType.DUCKDB)
    
    # Connect to the store
    await store.connect()
    
    # Create index (optional for DuckDB)
    dimension = 768
    await store.create_index(dimension=dimension)
    
    # Add vectors
    docs = [
        VectorDocument(
            id=f"doc-{i}",
            vector=[0.1 * j for j in range(dimension)],
            metadata={"text": f"Document {i}", "category": f"cat-{i % 5}"}
        )
        for i in range(100)
    ]
    
    await store.add_documents(docs)
    
    # Search
    query = SearchQuery(
        vector=[0.1 * j for j in range(dimension)],
        top_k=5,
        filter={"category": "cat-2"}
    )
    
    results = await store.search(query)
    for match in results.matches:
        print(f"ID: {match.id}, Score: {match.score}")
        print(f"Metadata: {match.metadata}")
    
    # Disconnect
    await store.disconnect()

asyncio.run(main())
```

### Advanced Usage with SQL Filtering

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import SearchQuery

async def advanced_duckdb():
    # Create DuckDB store with custom config
    store = await create_vector_store(
        VectorDBType.DUCKDB,
        config_override={
            "database_path": "data/custom.duckdb",
            "storage_path": "data/custom_parquet",
            "table_name": "custom_embeddings",
            "dimension": 384,
            "search": {
                "similarity_metric": "cosine",
                "sql_filter": "metadata_json::json->>'date' > '2025-01-01' AND metadata_json::json->>'category' = 'technology'"
            }
        }
    )
    
    # Connect
    await store.connect()
    
    # Search with SQL filter in query
    query = SearchQuery(
        vector=[0.1 * j for j in range(384)],
        top_k=10,
        sql_filter="metadata_json::json->>'score' > '0.5'"
    )
    
    results = await store.search(query)
    print(f"Found {len(results.matches)} results with SQL filtering")
    
    # Disconnect
    await store.disconnect()

asyncio.run(advanced_duckdb())
```

## Testing

To test the DuckDB Vector Store:

```bash
# Basic functionality test
./test_vector_stores.py --store duckdb --data

# Advanced features test
./test_vector_advanced.py --store duckdb

# Integration test
./test_vector_integration.py
```

## Implementation Details

### Vector Storage

The DuckDB Vector Service stores vectors in Parquet files:

1. Vectors are stored as arrays in Parquet columnar format
2. Each vector collection is stored in a separate Parquet file
3. DuckDB provides a SQL interface to query and process these files

### Vector Search

Vector search is implemented using SQL and vector distance functions:

1. DuckDB loads vectors from Parquet files
2. Distance calculations are performed using SQL expressions
3. Results are filtered and ordered by distance
4. Top-k results are returned

### Sharding

The DuckDB Vector Service implements sharding via multiple Parquet files:

1. Vectors are divided into shards of configurable size
2. Each shard is stored as a separate Parquet file
3. Searches are performed across all relevant shards using SQL UNION ALL
4. Results are merged and ordered by distance

### Metadata Storage

Vector metadata is stored alongside vectors using JSON:

1. Metadata is serialized to JSON and stored in the Parquet files
2. Rich SQL filtering is supported during search
3. Metadata is automatically parsed and returned with search results

## Performance Considerations

- **Memory Usage**: DuckDB can operate with very low memory usage compared to in-memory vector databases
- **Batch Processing**: DuckDB excels at processing large batches of vectors
- **Query Performance**: While not as fast as specialized vector databases for single-query use cases, performance is excellent for analytical workloads
- **Storage Efficiency**: Parquet's columnar format provides excellent compression and storage efficiency

## For More Information

- [Vector Stores Overview](vector-stores.md) - Introduction to the vector store architecture
- [DuckDB Examples](examples/duckdb-examples.md) - More DuckDB integration examples
