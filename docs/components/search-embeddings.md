# Search Embeddings Component

The Search Embeddings component provides semantic search capabilities over embedded datasets. It enables fast, accurate similarity search using various vector database backends.

## Overview

The search component is responsible for:
- Loading and indexing embedding datasets
- Performing vector similarity searches
- Ranking and filtering results
- Supporting multiple search backends

## Core Files

- **Main Implementation**: `search_embeddings/search_embeddings.py`
- **Configuration**: Inherits from core library configuration
- **Dependencies**: FAISS, Qdrant, Elasticsearch support

## Key Features

### Vector Similarity Search
- Cosine similarity
- Euclidean distance
- Dot product similarity
- Custom distance metrics

### Multi-Backend Support
- **FAISS**: High-performance CPU/GPU search
- **Qdrant**: Cloud-native vector database
- **Elasticsearch**: Full-text + vector search

### Query Processing
- Real-time search
- Batch query processing
- Query preprocessing and normalization
- Result post-processing

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Search API    │───▶│  Query Processor │───▶│  Vector Index   │
│   Endpoint      │    │                 │    │   (FAISS/etc)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Result        │    │   Embedding     │    │   Metadata      │
│   Ranking       │    │   Generation    │    │   Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## API Integration

### Search Endpoint

The search component is accessible through the `/search` API endpoint:

```python
@app.post("/search")
async def search_item_task(collection: str, text: str, n: int):
    return await search.search_item(collection, text, n)
```

### Parameters

- `collection`: Name of the loaded embedding collection
- `text`: Query text for semantic search
- `n`: Number of results to return (1-100)

## Implementation Details

### Class Structure

```python
class search_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        
    async def search_item(self, collection, text, n):
        """Main search method"""
        pass
        
    async def load_index(self, dataset, knn_index, splits, columns):
        """Load embedding index"""
        pass
```

### Search Process

1. **Query Preprocessing**
   - Text normalization
   - Tokenization
   - Embedding generation

2. **Vector Search**
   - Index lookup
   - Similarity computation
   - Result ranking

3. **Post-processing**
   - Metadata enrichment
   - Result filtering
   - Score normalization

## Supported Backends

### FAISS Backend

**Advantages:**
- Extremely fast search
- GPU acceleration support
- Memory efficient
- Various index types (IVF, HNSW, etc.)

**Configuration:**
```python
faiss_config = {
    "index_type": "IVFFlat",
    "nlist": 1024,
    "nprobe": 32,
    "gpu_enabled": True
}
```

### Qdrant Backend

**Advantages:**
- Cloud-native architecture
- Built-in filtering
- Distributed search
- Real-time updates

**Configuration:**
```python
qdrant_config = {
    "host": "localhost",
    "port": 6333,
    "collection_name": "embeddings",
    "vector_size": 384
}
```

### Elasticsearch Backend

**Advantages:**
- Full-text + vector search
- Rich filtering capabilities
- Scalable architecture
- Advanced analytics

**Configuration:**
```python
elasticsearch_config = {
    "host": "localhost",
    "port": 9200,
    "index_name": "embeddings",
    "vector_field": "embedding"
}
```

## Search Algorithms

### Approximate Nearest Neighbor (ANN)

For large-scale datasets, the component uses ANN algorithms:

- **IVF (Inverted File)**: Clusters vectors into cells
- **HNSW (Hierarchical NSW)**: Graph-based search
- **PQ (Product Quantization)**: Compressed vectors

### Exact Search

For smaller datasets or high-precision requirements:
- Brute force search
- Exact similarity computation
- No approximation errors

## Performance Optimization

### Indexing Strategies

```python
# Index configuration for different use cases
index_configs = {
    "small_dataset": {
        "type": "Flat",
        "metric": "cosine"
    },
    "medium_dataset": {
        "type": "IVFFlat", 
        "nlist": 1024,
        "metric": "cosine"
    },
    "large_dataset": {
        "type": "IVFPQ",
        "nlist": 4096,
        "m": 64,
        "nbits": 8
    }
}
```

### Query Optimization

1. **Batch Processing**: Process multiple queries together
2. **Caching**: Cache frequent queries and embeddings
3. **Prefiltering**: Filter before vector search when possible
4. **Parallel Search**: Use multiple threads/processes

### Memory Management

- **Memory Mapping**: For large indices
- **Compression**: Reduce memory footprint
- **Lazy Loading**: Load index parts on demand

## Error Handling

### Common Errors

```python
class SearchErrors:
    COLLECTION_NOT_FOUND = "Collection '{}' not found"
    INDEX_NOT_LOADED = "No index loaded for collection '{}'"
    INVALID_QUERY = "Invalid query: {}"
    SEARCH_TIMEOUT = "Search timeout after {} seconds"
    INSUFFICIENT_MEMORY = "Insufficient memory for search operation"
```

### Error Recovery

- Automatic retry with exponential backoff
- Fallback to alternative backends
- Graceful degradation with reduced functionality

## Monitoring and Metrics

### Performance Metrics

```python
search_metrics = {
    "queries_per_second": float,
    "average_latency_ms": float,
    "p95_latency_ms": float,
    "p99_latency_ms": float,
    "cache_hit_rate": float,
    "error_rate": float
}
```

### Health Checks

```python
async def health_check():
    return {
        "status": "healthy",
        "indices_loaded": len(self.loaded_indices),
        "memory_usage_mb": get_memory_usage(),
        "last_search_time": self.last_search_time
    }
```

## Configuration

### Environment Variables

```bash
# Search configuration
SEARCH_DEFAULT_BACKEND=faiss
SEARCH_MAX_RESULTS=1000
SEARCH_TIMEOUT_SECONDS=30
SEARCH_CACHE_ENABLED=true
SEARCH_CACHE_TTL=3600

# Backend-specific configuration
FAISS_GPU_ENABLED=true
FAISS_OMP_THREADS=8
QDRANT_HOST=localhost
QDRANT_PORT=6333
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
```

### YAML Configuration

```yaml
search:
  backends:
    faiss:
      enabled: true
      gpu_enabled: true
      index_type: "IVFFlat"
      
    qdrant:
      enabled: false
      host: "localhost"
      port: 6333
      
    elasticsearch:
      enabled: false
      host: "localhost"
      port: 9200
      
  performance:
    max_results: 1000
    timeout_seconds: 30
    cache_enabled: true
    parallel_queries: 4
    
  quality:
    min_similarity_threshold: 0.1
    result_diversification: false
    reranking_enabled: false
```

## Usage Examples

### Basic Search

```python
# Initialize search component
search = search_embeddings(resources, metadata)

# Load an index
await search.load_index(
    dataset="my_dataset",
    knn_index="my_index", 
    splits=["train"],
    columns=["text"]
)

# Perform search
results = await search.search_item(
    collection="my_collection",
    text="machine learning algorithms",
    n=10
)
```

### Batch Search

```python
# Multiple queries
queries = [
    "artificial intelligence",
    "machine learning",
    "deep learning"
]

batch_results = await search.batch_search(
    collection="my_collection",
    texts=queries,
    n=5
)
```

### Advanced Search with Filters

```python
# Search with metadata filters
results = await search.advanced_search(
    collection="my_collection",
    text="neural networks",
    n=20,
    filters={
        "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
        "category": ["research", "tutorial"],
        "min_score": 0.7
    }
)
```

## Integration with Other Components

### With Create Embeddings
- Use newly created embeddings for search
- Automatic index updates
- Incremental indexing

### With IPFS Storage
- Load indices from IPFS
- Distributed index sharing
- Content-addressable storage

### With Endpoint Management
- Dynamic model selection for query embedding
- Load balancing across endpoints
- Fallback mechanisms

## Best Practices

### Index Design

1. **Choose appropriate index type** based on dataset size
2. **Optimize index parameters** for your use case
3. **Consider memory vs. accuracy tradeoffs**
4. **Plan for index updates and maintenance**

### Query Optimization

1. **Preprocess queries** for better results
2. **Use appropriate similarity metrics**
3. **Implement result caching** for frequent queries
4. **Monitor and tune performance regularly**

### Scalability

1. **Shard large indices** across multiple nodes
2. **Use read replicas** for high query loads
3. **Implement query routing** for distributed setups
4. **Plan for horizontal scaling**

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow search | Large index, inefficient algorithm | Optimize index type, use GPU |
| High memory usage | Large uncompressed index | Use compression, memory mapping |
| Poor recall | Wrong similarity metric | Tune index parameters |
| Search timeouts | Overloaded system | Scale horizontally, optimize queries |

### Debug Mode

Enable debug logging for detailed search information:

```python
import logging
logging.getLogger("search_embeddings").setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

- Real-time index updates
- Advanced filtering capabilities
- Multi-modal search support
- Federated search across multiple indices
- Learning-to-rank integration

### Performance Improvements

- GPU-accelerated preprocessing
- Advanced caching strategies
- Query optimization using machine learning
- Adaptive index selection

## Related Documentation

- [API Reference](../api/README.md) - Search API endpoints
- [Configuration](../configuration.md) - Search configuration
- [FAISS Kit](../ipfs/storage.md) - Vector storage backends
- [Performance Tuning](../troubleshooting/performance.md) - Optimization guides
