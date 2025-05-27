# Shard Embeddings Component

The Shard Embeddings component enables distributed processing and storage of large embedding datasets through intelligent sharding strategies. It uses clustering algorithms to partition embeddings into coherent groups, enabling efficient parallel processing and improved search performance.

## Overview

The `shard_embeddings` class provides:
- **Intelligent Sharding**: Use K-means clustering to create semantically coherent shards
- **Distributed Processing**: Split large datasets across multiple nodes
- **Load Balancing**: Distribute computational load efficiently
- **Scalable Storage**: Manage large embedding collections across distributed systems

## Key Features

### K-Means Clustering
- **Semantic Grouping**: Cluster similar embeddings together
- **Configurable Clusters**: Set optimal number of shards for your use case
- **Automatic Optimization**: Find optimal cluster parameters
- **Quality Metrics**: Evaluate clustering quality

### Distributed Architecture
- **Multi-Node Support**: Process across multiple machines
- **Fault Tolerance**: Handle node failures gracefully
- **Dynamic Scaling**: Add/remove nodes based on load
- **Resource Management**: Optimize resource utilization

### Performance Optimization
- **Parallel Processing**: Process multiple shards simultaneously
- **Memory Efficiency**: Process shards independently to reduce memory usage
- **Network Optimization**: Minimize data transfer between nodes
- **Caching**: Cache frequently accessed shards

## Usage

### Basic Sharding

```python
from shard_embeddings import shard_embeddings
import asyncio

# Configuration
metadata = {
    "dataset": "TeraflopAI/Caselaw_Access_Project",
    "column": "text",
    "split": "train",
    "models": [
        "thenlper/gte-small",
        "Alibaba-NLP/gte-large-en-v1.5",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    ],
    "chunk_settings": {
        "chunk_size": 512,
        "n_sentences": 8,
        "step_size": 256,
        "method": "fixed",
        "embed_model": "thenlper/gte-small",
        "tokenizer": None
    },
    "dst_path": "/storage/sharded_embeddings",
    "n_clusters": 100,  # Number of shards
    "cluster_method": "kmeans"
}

resources = {
    "cluster_nodes": [
        "http://node1:8080",
        "http://node2:8080", 
        "http://node3:8080"
    ]
}

# Initialize component
sharder = shard_embeddings(resources, metadata)

# Create shards
result = await sharder(metadata)
```

### Advanced Clustering Configuration

```python
# Custom clustering parameters
advanced_metadata = {
    "dataset": "large-corpus",
    "column": "document_text",
    "split": "train",
    "models": ["sentence-transformers/all-mpnet-base-v2"],
    "clustering": {
        "method": "kmeans",
        "n_clusters": 500,
        "init": "k-means++",
        "max_iter": 300,
        "tol": 1e-4,
        "random_state": 42
    },
    "shard_config": {
        "min_shard_size": 1000,
        "max_shard_size": 50000,
        "overlap_ratio": 0.1
    },
    "dst_path": "/data/advanced_shards"
}

sharder = shard_embeddings(resources, advanced_metadata)
await sharder(advanced_metadata)
```

## Configuration

### Metadata Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `dataset` | str | Dataset name or path | Yes |
| `column` | str | Text column to process | Yes |
| `split` | str | Dataset split | Yes |
| `models` | list | Embedding models to use | Yes |
| `dst_path` | str | Output directory for shards | Yes |
| `n_clusters` | int | Number of shards/clusters | No (default: 50) |
| `cluster_method` | str | Clustering algorithm | No (default: "kmeans") |

### Clustering Configuration

```python
clustering_config = {
    "method": "kmeans",           # Clustering algorithm
    "n_clusters": 100,            # Number of clusters
    "init": "k-means++",          # Initialization method
    "max_iter": 300,              # Maximum iterations
    "tol": 1e-4,                  # Convergence tolerance
    "random_state": 42,           # Reproducibility seed
    "n_init": 10                  # Number of random initializations
}
```

### Shard Configuration

```python
shard_config = {
    "min_shard_size": 100,        # Minimum documents per shard
    "max_shard_size": 10000,      # Maximum documents per shard
    "overlap_ratio": 0.05,        # Overlap between adjacent shards
    "rebalance_threshold": 0.2    # Trigger rebalancing when imbalanced
}
```

## API Reference

### Class: `shard_embeddings`

#### `__init__(resources, metadata)`
Initialize the shard embeddings component.

**Parameters:**
- `resources` (dict): Cluster nodes and resource configuration
- `metadata` (dict): Sharding parameters and dataset information

#### `__call__(metadata=None)`
Execute the sharding process.

**Parameters:**
- `metadata` (dict, optional): Override metadata (uses instance metadata if None)

**Returns:**
- Result from the K-means cluster split operation

#### `test()`
Run tests to validate the sharding functionality.

**Returns:**
- Test results indicating success or failure

## Implementation Examples

### Large-Scale Document Sharding

```python
# Shard a large document collection
large_scale_metadata = {
    "dataset": "wikipedia-en",
    "column": "text", 
    "split": "train",
    "models": ["sentence-transformers/all-mpnet-base-v2"],
    "clustering": {
        "n_clusters": 1000,  # Many shards for large dataset
        "method": "kmeans",
        "batch_size": 5000   # Process in batches
    },
    "dst_path": "/storage/wikipedia_shards"
}

# Distributed resources
resources = {
    "cluster_nodes": [
        f"http://node{i}:8080" for i in range(1, 11)  # 10 nodes
    ],
    "storage_backends": [
        {"type": "local", "path": "/storage/local"},
        {"type": "ipfs", "gateway": "http://ipfs:5001"},
        {"type": "s3", "bucket": "embeddings-shards"}
    ]
}

sharder = shard_embeddings(resources, large_scale_metadata)
result = await sharder()
```

### Multi-Model Sharding

```python
# Create shards for multiple embedding models
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2", 
    "microsoft/DialoGPT-medium"
]

for model in models:
    model_metadata = {
        "dataset": "multi-domain-corpus",
        "column": "text",
        "split": "train", 
        "models": [model],
        "clustering": {
            "n_clusters": 200,
            "method": "kmeans"
        },
        "dst_path": f"/storage/shards_{model.replace('/', '_')}"
    }
    
    sharder = shard_embeddings(resources, model_metadata)
    await sharder(model_metadata)
```

### Hierarchical Sharding

```python
# Create hierarchical shards (coarse -> fine)
async def hierarchical_sharding(dataset, levels=[50, 200, 1000]):
    """Create multiple levels of sharding for different granularities"""
    
    for level, n_clusters in enumerate(levels):
        metadata = {
            "dataset": dataset,
            "column": "text",
            "split": "train",
            "models": ["all-mpnet-base-v2"],
            "clustering": {
                "n_clusters": n_clusters,
                "method": "kmeans"
            },
            "dst_path": f"/storage/hierarchical/level_{level}"
        }
        
        sharder = shard_embeddings(resources, metadata)
        await sharder(metadata)
        print(f"Completed sharding level {level} with {n_clusters} clusters")

# Execute hierarchical sharding
await hierarchical_sharding("scientific-papers")
```

## Clustering Algorithms

### K-Means Clustering
The default and most commonly used clustering method:

```python
kmeans_config = {
    "method": "kmeans",
    "n_clusters": 100,
    "init": "k-means++",      # Smart initialization
    "max_iter": 300,          # Maximum iterations
    "tol": 1e-4,             # Convergence tolerance
    "algorithm": "auto"       # Algorithm selection
}
```

**Advantages:**
- Fast and scalable
- Works well with high-dimensional embeddings
- Produces balanced clusters

**Considerations:**
- Requires specifying number of clusters
- Assumes spherical clusters
- Sensitive to initialization

### Alternative Clustering Methods

```python
# DBSCAN for density-based clustering
dbscan_config = {
    "method": "dbscan",
    "eps": 0.5,              # Neighborhood distance
    "min_samples": 10        # Minimum points per cluster
}

# Hierarchical clustering
hierarchical_config = {
    "method": "hierarchical",
    "n_clusters": 100,
    "linkage": "ward",       # Linkage criterion
    "distance_threshold": None
}
```

## Performance Optimization

### Memory Management

```python
# Optimize for large datasets
memory_optimized_config = {
    "batch_processing": True,
    "batch_size": 10000,
    "memory_limit": "8GB",
    "temp_storage": "/tmp/sharding",
    "cleanup_intermediate": True
}
```

### Parallel Processing

```python
# Configure parallel execution
parallel_config = {
    "n_workers": 8,           # Number of worker processes
    "chunk_size": 1000,       # Documents per chunk
    "prefetch_factor": 2,     # Prefetch chunks
    "max_queue_size": 100     # Maximum queue size
}
```

### Network Optimization

```python
# Optimize for distributed processing
network_config = {
    "compression": "gzip",     # Compress data transfers
    "batch_transfer": True,    # Batch network operations
    "timeout": 300,           # Network timeout
    "retry_attempts": 3       # Retry failed transfers
}
```

## Quality Assessment

### Cluster Quality Metrics

```python
def evaluate_clustering_quality(embeddings, cluster_labels):
    """Evaluate the quality of clustering results"""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    metrics = {
        "silhouette_score": silhouette_score(embeddings, cluster_labels),
        "calinski_harabasz": calinski_harabasz_score(embeddings, cluster_labels),
        "n_clusters": len(set(cluster_labels)),
        "cluster_sizes": np.bincount(cluster_labels)
    }
    
    return metrics
```

### Shard Balance Analysis

```python
def analyze_shard_balance(shard_sizes):
    """Analyze the balance of shard sizes"""
    shard_sizes = np.array(shard_sizes)
    
    balance_metrics = {
        "mean_size": np.mean(shard_sizes),
        "std_size": np.std(shard_sizes),
        "min_size": np.min(shard_sizes),
        "max_size": np.max(shard_sizes),
        "coefficient_of_variation": np.std(shard_sizes) / np.mean(shard_sizes),
        "imbalance_ratio": np.max(shard_sizes) / np.min(shard_sizes)
    }
    
    return balance_metrics
```

## Error Handling

### Common Issues

1. **Memory Overflow**: Large datasets exceed available memory
   - Solution: Use batch processing and streaming
   - Monitor memory usage during clustering

2. **Cluster Imbalance**: Some clusters much larger than others
   - Solution: Adjust clustering parameters
   - Use post-processing to rebalance

3. **Network Failures**: Node communication issues
   - Solution: Implement retry logic and failover
   - Use redundant storage backends

### Debug and Monitoring

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor clustering progress
def monitor_clustering_progress(sharder):
    """Monitor the progress of sharding operation"""
    
    # Test component health
    test_results = sharder.test()
    
    if isinstance(test_results, Exception):
        print(f"Error in sharding: {test_results}")
        return False
    
    print("Sharding component healthy")
    return True

# Usage
if monitor_clustering_progress(sharder):
    result = await sharder()
```

## Integration with Other Components

### Search Integration

```python
# Use sharded embeddings for distributed search
from search_embeddings import search_embeddings

search_config = {
    "shard_strategy": "round_robin",
    "parallel_search": True,
    "merge_strategy": "score_weighted"
}

searcher = search_embeddings(resources, search_config)
results = searcher.search_sharded(
    query="machine learning",
    shard_paths=["/storage/shard_0", "/storage/shard_1"],
    top_k=100
)
```

### IPFS Integration

```python
# Store shards on IPFS for distributed access
ipfs_config = {
    "ipfs_gateway": "http://localhost:5001",
    "pin_shards": True,
    "replicate_factor": 3
}

# Upload shards to IPFS after creation
await sharder.upload_to_ipfs(ipfs_config)
```

## Dependencies

- `scikit-learn`: For clustering algorithms
- `numpy`: For numerical operations
- `ipfs_embeddings_py`: Core embedding infrastructure
- `asyncio`: For asynchronous processing
- `multiprocessing`: For parallel processing

## Related Components

- [Create Embeddings](create-embeddings.md): For generating embeddings to shard
- [Search Embeddings](search-embeddings.md): For distributed search across shards
- [IPFS Integration](../ipfs/README.md): For distributed storage
- [Sparse Embeddings](sparse-embeddings.md): Alternative embedding approach
