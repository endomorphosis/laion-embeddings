# Configuration Guide

This guide explains how to configure the LAION Embeddings system, with particular focus on the vector store providers, including IPFS and DuckDB.

## Configuration Overview

The LAION Embeddings system uses YAML configuration files for most settings. The primary configuration files are:

- `config/vector_databases.yaml`: Configuration for vector stores
- `config/embedding_models.yaml`: Configuration for embedding models
- `config/api.yaml`: Configuration for API endpoints and server settings

## Vector Store Configuration

### Main Vector Store Configuration

The main vector store configuration is in `config/vector_databases.yaml`:

```yaml
vector_store:
  # Set the default vector store provider
  default: "faiss"  # Options: "faiss", "ipfs", "duckdb", "hnsw"
  
  # Global configuration
  global:
    cache_enabled: true
    cache_size_mb: 1024
    metrics_enabled: true
    
  # Provider-specific configuration sections follow
```

### FAISS Configuration

```yaml
vector_databases:
  faiss:
    index_type: "IVF100,Flat"
    nprobe: 10
    quantizer_training_size: 100000
    store_on_disk: true
    storage_path: "./vector_data/faiss"
    use_gpu: false
    gpu_id: 0
    metric_type: "cosine"  # Options: "cosine", "inner_product", "l2"
```

### IPFS Configuration

```yaml
vector_databases:
  ipfs:
    # IPFS Connection
    multiaddr: "/ip4/127.0.0.1/tcp/5001"  # IPFS API address
    gateway: "http://localhost:8080"  # IPFS gateway address
    connection_timeout: 10.0  # seconds
    request_timeout: 30.0  # seconds
    
    # IPFS Storage
    pin: true  # Pin content to the IPFS node
    pin_recursive: true  # Recursively pin directory structures
    unixfs_chunker: "size-262144"  # IPFS chunking strategy
    block_size: 262144  # 256KB default block size
    
    # Vector specific
    metric_type: "cosine"  # Options: "cosine", "inner_product", "l2"
    storage_path: "./vector_data/ipfs"  # Local storage path for metadata
    
    # Sharding
    sharding:
      enabled: true
      shard_count: 4
      shard_strategy: "hash"  # Options: "hash", "range", "consistent-hash", "directory"
      
    # Vector Quantization
    quantization:
      enabled: false
      method: "pq"  # Options: "sq" (scalar), "pq" (product), "opq" (optimized product)
      bits: 8  # For scalar quantization: bits per dimension
      subquantizers: 8  # For product quantization: number of subquantizers
      
    # Performance
    enable_cache: true
    cache_size_mb: 1024
    parallel_operations: true
    max_parallel_requests: 10
    
    # Advanced IPFS 
    ipns_publish: false  # Publish index to IPNS
    ipns_key: "vector-index"
    use_filestore: true  # Use filestore for local block storage
    
    # Network configuration
    bootstrap_peers:
      - "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ"
    swarm_addresses:
      - "/ip4/0.0.0.0/tcp/4001"
      - "/ip6/::/tcp/4001"
```

### DuckDB Configuration

```yaml
vector_databases:
  duckdb:
    # Database Configuration
    database_path: "vectors.duckdb"  # DuckDB database file
    storage_path: "./vector_data/duckdb"  # Directory for Parquet files
    create_tables: true  # Create tables if they don't exist
    
    # Memory and Performance
    memory_limit: "4GB"  # Memory limit for DuckDB
    threads: 4  # Number of threads for parallel processing
    use_prepared_statements: true  # Use prepared statements for better performance
    enable_query_acceleration: true  # Enable DuckDB query acceleration
    read_only: false  # Read-only mode
    
    # Vector Configuration
    metric_type: "cosine"  # Options: "cosine", "inner_product", "l2"
    
    # Parquet Configuration
    parquet_page_size: 1048576  # 1MB
    row_group_size: 10000
    compression: "zstd"  # Options: "zstd", "snappy", "gzip", "none"
    compression_level: 3  # 1-9 (higher = better compression but slower)
    
    # Sharding
    sharding:
      enabled: false
      shard_count: 8
      shard_strategy: "range"  # Options: "hash", "range", "consistent-hash"
      
    # Vector Quantization
    quantization:
      enabled: false
      method: "sq"  # Options: "sq" (scalar)
      bits: 8
      
    # Table Configuration
    table_name: "vectors"
    metadata_format: "json"  # How to store metadata: "json" or "columnar"
    index_metadata_columns: ["category", "type"]  # Create indexes on these metadata columns
    
    # Extension Configuration
    load_extensions:
      - "httpfs"
      - "parquet"
      - "json"
```

### HNSW Configuration

```yaml
vector_databases:
  hnsw:
    # HNSW Parameters
    M: 16  # Max number of connections per layer
    ef_construction: 200  # Size of the dynamic candidate list during construction
    ef_search: 100  # Size of the dynamic candidate list during search
    
    # Storage
    store_on_disk: true
    storage_path: "./vector_data/hnsw"
    
    # Vector Configuration
    metric_type: "cosine"  # Options: "cosine", "inner_product", "l2"
    
    # Sharding
    sharding:
      enabled: false
      shard_count: 4
      shard_strategy: "hash"  # Options: "hash", "range"
```

## Configuration Methods

### Configuration Loading

The system loads configuration files from the following locations (in order of precedence):

1. Path specified in the `CONFIG_PATH` environment variable
2. `./config/` directory in the current working directory
3. `/etc/laion-embeddings/` directory for system-wide configuration

### Environment Variable Override

You can override configuration values using environment variables with the following pattern:

```
VECTOR_STORE_DEFAULT=ipfs
VECTOR_IPFS_MULTIADDR=/ip4/127.0.0.1/tcp/5001
VECTOR_IPFS_ENABLE_CACHE=true
VECTOR_IPFS_CACHE_SIZE_MB=2048
```

### Programmatic Configuration

You can also configure the system programmatically:

```python
from services.vector_config import VectorDatabaseConfig, IPFSConfig, ShardingConfig
from services.vector_store_factory import create_vector_store

# Configure IPFS store
ipfs_config = IPFSConfig(
    multiaddr="/ip4/127.0.0.1/tcp/5001",
    pin=True,
    sharding=ShardingConfig(
        enabled=True,
        shard_count=4
    )
)

# Create store with configuration
store = await create_vector_store(
    db_type=VectorDBType.IPFS,
    config=ipfs_config
)
```

## Advanced Configuration Topics

### Distributed Configuration

For distributed systems, you can configure multiple IPFS nodes:

```yaml
vector_databases:
  ipfs:
    ipfs_nodes:
      - "/ip4/node1.example.com/tcp/5001"
      - "/ip4/node2.example.com/tcp/5001"
      - "/ip4/node3.example.com/tcp/5001"
    replicate_across_nodes: true
    replication_factor: 2
```

### High-Availability Configuration

For high-availability setups:

```yaml
vector_databases:
  ipfs:
    high_availability:
      enabled: true
      min_replicas: 2
      read_quorum: 1
      write_quorum: 2
      health_check_interval: 60
```

### Hybrid Search Configuration

Configure hybrid search (vector + keyword):

```yaml
vector_databases:
  duckdb:
    hybrid_search:
      enabled: true
      keyword_weight: 0.3
      vector_weight: 0.7
      keyword_index: "standard"  # Options: "standard", "bm25", "tfidf"
```

## Monitoring and Metrics

Configure monitoring and metrics:

```yaml
monitoring:
  enabled: true
  log_level: "info"
  metrics:
    enabled: true
    prometheus: true
    prometheus_port: 9100
```

## Security Configuration

### Authentication

```yaml
security:
  authentication:
    enabled: true
    api_key_required: true
    jwt:
      enabled: true
      secret_key: "your-secret-key"
      algorithm: "HS256"
      token_expiry_minutes: 60
```

### API Access Control

```yaml
security:
  access_control:
    allowed_origins:
      - "https://your-app.example.com"
    rate_limiting:
      enabled: true
      requests_per_minute: 60
```

### IPFS Access Control

```yaml
vector_databases:
  ipfs:
    security:
      private_network: true
      swarm_key_path: "/path/to/swarm.key"
      api_auth_required: true
      gateway_auth_required: false
```

## Multi-environment Configuration

For managing multiple environments:

```yaml
environments:
  development:
    vector_store:
      default: "faiss"
    
  staging:
    vector_store:
      default: "ipfs"
    vector_databases:
      ipfs:
        multiaddr: "/ip4/staging-ipfs.example.com/tcp/5001"
    
  production:
    vector_store:
      default: "ipfs"
    vector_databases:
      ipfs:
        multiaddr: "/ip4/prod-ipfs.example.com/tcp/5001"
        sharding:
          enabled: true
          shard_count: 16
```

To select an environment:

```bash
export ENVIRONMENT=production
./run.sh
```

## Configuration Validation

The system validates configuration at startup. You can manually validate configuration:

```bash
python -m services.config_validator
```

## Conclusion

Proper configuration is essential for optimizing the LAION Embeddings system. By leveraging the flexible configuration options for different vector stores, you can tailor the system to your specific use case and performance requirements.
