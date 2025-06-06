# Sharding

This document explains the sharding capabilities in the LAION Embeddings system and how to effectively use them with different vector store providers, including IPFS and DuckDB.

## Introduction to Sharding

Sharding is a database partitioning technique that divides large datasets into smaller, more manageable pieces called shards. In the context of vector stores, sharding offers several benefits:

- **Scalability**: Handle datasets larger than what fits on a single node
- **Performance**: Parallelize operations across shards for faster processing
- **Availability**: Improve system resilience through redundancy
- **Load Distribution**: Balance computational load across multiple resources

## Sharding Architecture

The LAION Embeddings system implements a unified sharding interface that works across all supported vector store providers:

```
┌─────────────────────────────────────────┐
│           Client Application            │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│          VectorStoreBase (API)          │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│           ShardManager Layer            │
└┬─────────────────┬─────────────────────┬┘
 │                 │                     │
 ▼                 ▼                     ▼
┌─────────┐     ┌─────────┐     ┌────────────┐
│  Shard  │     │  Shard  │     │   Shard    │
│    1    │     │    2    │     │     N      │
└─────────┘     └─────────┘     └────────────┘
```

## Configuration

To use sharding with any vector store provider, configure the sharding settings in your vector store configuration:

```yaml
vector_databases:
  ipfs:
    # Other IPFS configuration...
    sharding:
      enabled: true
      shard_count: 4
      shard_strategy: "hash"  # Options: "hash", "range", "consistent-hash"
      
  duckdb:
    # Other DuckDB configuration...
    sharding:
      enabled: true
      shard_count: 8
      shard_strategy: "range"
```

## Sharding Strategies

The system supports multiple sharding strategies:

1. **Hash-based Sharding**: Distributes documents across shards based on a hash of their ID
2. **Range-based Sharding**: Partitions the vector space into continuous regions
3. **Consistent Hashing**: Minimizes redistribution when adding or removing shards
4. **Directory-based Sharding**: (IPFS-specific) Uses IPFS directory structure for sharding

## IPFS-Specific Sharding

IPFS offers unique distributed sharding capabilities:

### Directory-based Sharding

The IPFS vector store uses IPFS directories to implement sharding:

```
ipfs-root-cid/
  ├── shard-0/
  │   ├── vector-000001.bin
  │   ├── vector-000002.bin
  │   └── metadata.json
  ├── shard-1/
  │   ├── vector-000003.bin
  │   ├── vector-000004.bin
  │   └── metadata.json
  ...
  └── index-metadata.json
```

### Distributed Sharding

IPFS allows for truly distributed sharding across multiple nodes:

```python
# Example: Creating a distributed IPFS vector store with sharding
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType, ShardingConfig

# Configure sharding
sharding = ShardingConfig(
    enabled=True,
    shard_count=4,
    shard_strategy="consistent-hash"
)

# Create IPFS store with sharding
store = await create_vector_store(
    db_type=VectorDBType.IPFS,
    sharding=sharding,
    ipfs_nodes=["http://node1:5001", "http://node2:5001"]
)
```

## DuckDB-Specific Sharding

DuckDB implements sharding using Parquet files:

### Parquet-based Sharding

```
data-directory/
  ├── shard_0.parquet
  ├── shard_1.parquet
  ├── shard_2.parquet
  ...
  └── metadata.json
```

### SQL-based Querying

DuckDB sharding leverages SQL capabilities for efficient querying:

```python
# Example: Creating a DuckDB vector store with sharding
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType, ShardingConfig

# Configure sharding
sharding = ShardingConfig(
    enabled=True,
    shard_count=8,
    shard_strategy="range"
)

# Create DuckDB store with sharding
store = await create_vector_store(
    db_type=VectorDBType.DUCKDB,
    sharding=sharding
)
```

## Cross-shard Operations

### Search

When performing a search, the system executes the query across all relevant shards and merges the results:

```python
from services.vector_store_base import SearchQuery

# Create search query
query = SearchQuery(
    vector=[0.1, 0.2, ...],
    top_k=10
)

# Search across all shards
results = await store.search(query)
# System automatically handles:
# 1. Dispatching the query to all shards
# 2. Collecting results from each shard
# 3. Merging and ranking the combined results
# 4. Returning the top-k matches
```

### Batch Operations

Batch operations are automatically distributed across shards:

```python
# Add documents across shards
await store.add_documents(documents)  # Documents distributed based on sharding strategy

# Delete documents across shards
await store.delete_documents(document_ids)  # Deletions applied to relevant shards
```

## Monitoring and Maintenance

### Shard Balance Monitoring

```python
# Get shard statistics
stats = await store.get_shard_stats()

for shard_id, shard_stats in stats.items():
    print(f"Shard {shard_id}: {shard_stats['count']} vectors, {shard_stats['size_bytes']} bytes")
```

### Rebalancing

```python
# Rebalance shards (redistributes documents if shards are imbalanced)
rebalance_stats = await store.rebalance_shards()
```

## Best Practices

- **Choose the appropriate shard count**: Too many shards increase overhead; too few limit scalability
- **Select the right sharding strategy** for your access patterns:
  - Use **hash-based** for even distribution
  - Use **range-based** when similar vectors are frequently accessed together
  - Use **consistent hashing** when dynamically adding/removing shards
- **Monitor shard balance** regularly and rebalance when necessary
- **Distribute shards** across different physical resources for maximum performance
- **For IPFS**: Consider network topology when designing your sharding strategy

## Conclusion

Sharding is a powerful technique for scaling vector databases. The unified sharding architecture in LAION Embeddings ensures consistent behavior across different vector store providers while leveraging the unique capabilities of each backend.
