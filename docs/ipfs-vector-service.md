# IPFS Vector Service

The IPFS Vector Service provides distributed, content-addressable storage for vector embeddings using the InterPlanetary File System (IPFS). This allows for decentralized vector databases with resilience, redundancy, and content-addressing capabilities.

## Overview

The IPFS Vector Service leverages IPFS/IPLD (InterPlanetary Linked Data) to store vector embeddings in a distributed manner. Each vector collection is automatically sharded and distributed across the IPFS network, with content-addressable identifiers (CIDs) that ensure data integrity.

## Features

- **Distributed Storage**: Store vectors across a network of IPFS nodes
- **Content Addressing**: Vectors are addressed by their content hash (CID)
- **Automatic Sharding**: Large vector collections are automatically divided into manageable shards
- **Resilience**: Data can be retrieved from any node hosting the content
- **Metadata Preservation**: Rich metadata is preserved alongside vector embeddings
- **Fault Tolerance**: Continues functioning despite node failures or network issues

## Architecture

The IPFS Vector Service integrates with the unified vector store architecture as follows:

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
│            IPFSVectorStore                      │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            IPFSVectorStorage                    │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            IPFS Network                         │
└─────────────────────────────────────────────────┘
```

## Implementation

The IPFS Vector Service is implemented in two main components:

1. **IPFSVectorStore**: Implementation of the `BaseVectorStore` interface for the unified architecture
2. **IPFSVectorStorage**: Core functionality for interacting with IPFS (used by `IPFSVectorStore`)

## Dependencies

The IPFS Vector Service requires one of the following libraries:

- **ipfs_kit_py**: Modern IPFS library with enhanced functionality (recommended)
- **ipfshttpclient**: Standard IPFS HTTP client library

Additionally, an IPFS daemon should be running and accessible.

## Configuration

The IPFS Vector Store is configured in `config/vector_databases.yaml` under the `ipfs` section:

```yaml
databases:
  ipfs:
    enabled: true
    ipfs_gateway: "localhost:5001"
    sharding_enabled: true
    max_shard_size: 10000
    dimension: 768
    storage:
      cache_path: "data/ipfs_cache"
    search:
      similarity_metric: "cosine"
      use_quantization: false
    performance:
      concurrent_searches: 4
```

## Usage Examples

### Basic Usage

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def main():
    # Create IPFS store
    store = await create_vector_store(VectorDBType.IPFS)
    
    # Connect to the store
    await store.connect()
    
    # Create index with sharding
    dimension = 768
    await store.create_index(
        dimension=dimension,
        sharding=True,
        sharding_params={'max_shard_size': 5000}
    )
    
    # Add vectors
    docs = [
        VectorDocument(
            id=f"doc-{i}",
            vector=[0.1 * j for j in range(dimension)],
            metadata={"text": f"Document {i}"}
        )
        for i in range(100)
    ]
    
    await store.add_documents(docs)
    
    # Search
    query = SearchQuery(
        vector=[0.1 * j for j in range(dimension)],
        top_k=5,
        filter={"text": {"$contains": "Document"}}
    )
    
    results = await store.search(query)
    for match in results.matches:
        print(f"ID: {match.id}, Score: {match.score}")
        print(f"Metadata: {match.metadata}")
    
    # Disconnect
    await store.disconnect()

asyncio.run(main())
```

### Advanced Usage with Custom Configuration

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def advanced_ipfs():
    # Create IPFS store with custom config
    store = await create_vector_store(
        VectorDBType.IPFS,
        config_override={
            "ipfs_gateway": "localhost:5001",
            "sharding_enabled": True,
            "max_shard_size": 1000,
            "dimension": 384,
            "storage": {
                "cache_path": "data/custom_cache"
            },
            "search": {
                "similarity_metric": "cosine",
                "use_quantization": True,
                "quantization_params": {
                    "type": "scalar",
                    "bits": 8
                }
            }
        }
    )
    
    # Connect and use as with any other store
    await store.connect()
    
    # ... vector operations ...
    
    await store.disconnect()

asyncio.run(advanced_ipfs())
```

## Testing

To test the IPFS Vector Store:

```bash
# Basic functionality test
./test_vector_stores.py --store ipfs --data

# Advanced features test (sharding, quantization)
./test_vector_advanced.py --store ipfs

# Integration test
./test_vector_integration.py
```

## Implementation Details

### Sharding

The IPFS Vector Store automatically shards large vector collections:

1. Vectors are grouped into shards of configurable size (`max_shard_size`)
2. Each shard is stored as a separate IPFS object with its own CID
3. A manifest file tracks all shards and their CIDs
4. Searches are performed across all relevant shards and results are merged

### Vector Quantization

The IPFS Vector Store supports vector quantization to reduce storage size:

1. Vectors are quantized to reduce precision (e.g., from 32-bit float to 8-bit integer)
2. Quantization parameters are stored in the manifest
3. Vectors are automatically dequantized during search operations

### Metadata Storage

Vector metadata is stored alongside the vectors:

1. Metadata is serialized and stored with each vector
2. Rich query filtering is supported during search
3. Metadata is automatically retrieved and returned with search results

## For More Information

- [Vector Stores Overview](vector-stores.md) - Introduction to the vector store architecture
- [IPFS Examples](examples/ipfs-examples.md) - More IPFS integration examples
- [Vector Quantization](advanced/vector-quantization.md) - Details on vector quantization
- [Sharding](advanced/sharding.md) - Details on vector sharding
