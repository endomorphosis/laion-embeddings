# Unified Vector Store Architecture

This project provides a unified architecture for working with vector embeddings across multiple vector database systems, including traditional vector databases and distributed storage systems.

## Supported Vector Store Providers

The following vector store providers are supported:

1. **Qdrant** - In-memory and persistent vector database
2. **Elasticsearch** - Scalable search engine with vector capabilities
3. **pgvector** - PostgreSQL extension for vector similarity search
4. **FAISS** - Facebook AI Similarity Search library
5. **IPFS/IPLD** - InterPlanetary File System with content-addressable storage
6. **DuckDB/Parquet** - Analytical database with columnar Parquet storage

## Using the Vector Store Providers

### Basic Example

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def main():
    # Create a store instance (will use default from config if None)
    store = await create_vector_store(VectorDBType.IPFS)
    
    # Connect to the store
    await store.connect()
    
    # Add documents
    docs = [
        VectorDocument(
            id="doc1",
            vector=[0.1, 0.2, 0.3, 0.4],
            metadata={"text": "Example document"}
        )
    ]
    
    await store.add_documents(docs)
    
    # Search
    query = SearchQuery(
        vector=[0.1, 0.2, 0.3, 0.4],
        top_k=5
    )
    
    results = await store.search(query)
    print(results)
    
    # Disconnect
    await store.disconnect()

asyncio.run(main())
```

### Using IPFS/IPLD Vector Store

IPFS/IPLD provides a distributed, content-addressable storage system for vector embeddings, enabling decentralized vector databases.

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def use_ipfs():
    # Create IPFS store with custom config
    store = await create_vector_store(
        VectorDBType.IPFS,
        config_override={
            "ipfs_gateway": "localhost:5001",
            "sharding_enabled": True,
            "max_shard_size": 5000,
            "dimension": 768
        }
    )
    
    # Connect and use as with any other store
    await store.connect()
    
    # Store supports distributed vector storage and retrieval
    # with automatic sharding and CID-based addressing
    
    await store.disconnect()

asyncio.run(use_ipfs())
```

### Using DuckDB/Parquet Vector Store

DuckDB combined with Parquet files provides an efficient analytical vector database with columnar storage.

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def use_duckdb():
    # Create DuckDB store with custom config
    store = await create_vector_store(
        VectorDBType.DUCKDB,
        config_override={
            "database_path": "data/vectors.duckdb",
            "storage_path": "data/vector_parquet",
            "table_name": "embeddings"
        }
    )
    
    # Connect and use as with any other store
    await store.connect()
    
    # DuckDB store supports SQL-based vector queries
    # with efficient Parquet storage
    
    await store.disconnect()

asyncio.run(use_duckdb())
```

## Vector Quantization and Sharding

All vector store providers support vector quantization and sharding when the underlying implementation allows.

### Quantization Example

```python
from services.vector_store_base import SearchQuery

# Create index with quantization
await store.create_index(
    dimension=768,
    quantization=True,
    quantization_params={'type': 'scalar', 'bits': 8}
)

# Search using quantized vectors
query = SearchQuery(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    use_quantization=True
)

results = await store.search(query)
```

### Sharding Example

```python
# Create index with sharding
await store.create_index(
    dimension=768,
    sharding=True,
    sharding_params={'max_shard_size': 10000}
)

# Add large number of vectors - will be automatically sharded
await store.add_documents(large_vector_list)

# Search will automatically query across all shards
results = await store.search(query)
```

## Testing

To test individual vector store providers:

```bash
./test_vector_stores.py --store ipfs --data
./test_vector_stores.py --store duckdb --data
```

For advanced testing of vector quantization and sharding:

```bash
./test_vector_advanced.py --store ipfs
./test_vector_advanced.py --store duckdb
```

## Configuration

Vector store providers are configured in `config/vector_databases.yaml`. Each provider has its own section with connection parameters, index parameters, and search parameters.
