# Frequently Asked Questions (FAQ)

## General Questions

### What is LAION Embeddings?

LAION Embeddings is an advanced, production-ready IPFS-based embeddings search engine that provides FastAPI endpoints for creating, searching, and managing embeddings using multiple ML models and storage backends.

### Which machine learning models are supported?

The system supports multiple embedding models, including but not limited to:
- gte-small
- gte-large-en-v1.5
- gte-Qwen2-1.5B-instruct
- And others configurable through the API

### How do I start using LAION Embeddings?

1. Install the system following the [Installation Guide](installation.md)
2. Start the server with `./run.sh`
3. Follow the [Quick Start Guide](quickstart.md) for basic usage

## Vector Store Questions

### What vector stores does the system support?

The system supports multiple vector store backends:
- FAISS (default)
- IPFS (distributed storage)
- DuckDB with Parquet (analytical workloads)
- HNSW (hierarchical navigable small world graphs)

### When should I use the IPFS vector store?

Use the IPFS vector store when you need:
- Distributed storage and retrieval
- Content-addressed storage
- Decentralized architecture
- Data resilience through replication

### When should I use the DuckDB vector store?

Use the DuckDB vector store when you need:
- Analytical capabilities alongside vector search
- SQL-based querying and filtering
- Efficient storage with Parquet
- Integration with data warehousing workflows

### Can I switch between vector stores without changing my application code?

Yes. The unified vector store interface allows you to switch providers while maintaining the same API calls. You only need to change the configuration or provider parameter when creating the store.

### How do I choose the right vector store for my use case?

Consider these factors when choosing a vector store:
- **Scale**: How many vectors will you store?
- **Query rate**: How many searches per second?
- **Distribution**: Do you need a distributed system?
- **Analytical needs**: Do you need SQL-like querying?
- **Integration**: What systems does your app already use?

## IPFS-Specific Questions

### What is the advantage of using IPFS for vector storage?

IPFS provides:
- Content-addressed storage (immutability)
- Distributed architecture
- Built-in data verification
- Deduplication of identical vectors
- Peer-to-peer retrieval capabilities

### How does sharding work with IPFS vector store?

The IPFS vector store implements sharding by:
1. Partitioning vectors across multiple IPFS directories
2. Using a configurable sharding strategy (hash, range, etc.)
3. Maintaining a shard map for routing queries
4. Executing parallel operations across shards

### Does the IPFS vector store require an external IPFS node?

By default, the IPFS vector store connects to local or specified IPFS nodes. You can:
- Use existing IPFS nodes
- Let the system start an embedded IPFS node
- Connect to remote IPFS nodes or gateways

### What IPFS node implementation is compatible?

The system is compatible with:
- go-ipfs
- js-ipfs
- kubo
- And other standard IPFS implementations

## DuckDB-Specific Questions

### How are vectors stored in DuckDB?

Vectors in DuckDB are stored as:
- Parquet files with columnar storage
- Binary vector data in columnar format
- With optional indexes for fast retrieval
- Including metadata in structured columns

### Can I perform SQL queries alongside vector similarity search?

Yes, the DuckDB vector store supports:
- Standard vector similarity search
- SQL filtering conditions
- Combining vector search with SQL predicates
- Exporting data for SQL analysis

### What is the performance difference between DuckDB and other vector stores?

DuckDB vector store typically offers:
- Better analytical query performance
- More efficient filtering operations
- Slightly slower pure KNN search compared to FAISS
- Better compression with Parquet

### Does DuckDB support vector quantization?

Yes, DuckDB supports vector quantization through:
- Scalar quantization (reduced bit precision)
- Compressed vector storage
- Efficient columnar storage

## Performance Questions

### How can I optimize the performance of vector searches?

Key optimization strategies:
- Use appropriate vector quantization
- Configure properly sized shards
- Set appropriate index parameters
- Use filtering to reduce search space
- Consider hardware acceleration

### How many vectors can the system handle?

Capacity depends on:
- Vector store backend (FAISS, IPFS, DuckDB)
- Available memory and storage
- Vector dimensionality
- Use of sharding and quantization
- Hardware specifications

For reference:
- Single-node FAISS: millions to tens of millions
- IPFS distributed: hundreds of millions
- DuckDB with Parquet: tens to hundreds of millions

### How does vector dimensionality affect performance?

Higher dimensionality:
- Increases storage requirements
- Slows down similarity computations
- May require more sophisticated indexing
- Benefits more from vector quantization

## Configuration Questions

### How do I configure different vector stores?

Configure vector stores in `config/vector_databases.yaml`:

```yaml
vector_databases:
  faiss:
    index_type: "IVF100,Flat"
    nprobe: 10
    
  ipfs:
    multiaddr: "/ip4/127.0.0.1/tcp/5001"
    pin: true
    sharding:
      enabled: true
      shard_count: 4
      
  duckdb:
    database_path: "vectors.duckdb"
    storage_path: "./vector_data"
    memory_limit: "4GB"
```

### How do I change the default vector store?

Set the default vector store in your configuration:

```yaml
vector_store:
  default: "ipfs"  # Options: "faiss", "ipfs", "duckdb", "hnsw"
```

Or specify when creating:

```python
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

store = await create_vector_store(db_type=VectorDBType.IPFS)
```

## Implementation Questions

### How do I add multi-model support to my application?

Use different models for embedding generation:

```python
async def create_embeddings_multimodel():
    from services.embedding import get_embedding_model
    
    # Get different models
    model1 = get_embedding_model("thenlper/gte-small")
    model2 = get_embedding_model("Alibaba-NLP/gte-large-en-v1.5")
    
    # Generate embeddings with different models
    embedding1 = await model1.embed("Your text here")
    embedding2 = await model2.embed("Your text here")
    
    return embedding1, embedding2
```

### Can I use the system with my custom embedding models?

Yes, you can integrate custom embedding models by:
1. Creating a wrapper that follows the embedding interface
2. Registering your model in the embedding factory
3. Using your model name when calling the API

### How do I implement hybrid search (vector + keyword)?

Implement hybrid search by combining vector similarity with metadata filtering:

```python
from services.vector_store_base import SearchQuery

# Create a hybrid search query
query = SearchQuery(
    vector=your_query_vector,  # Vector for similarity search
    top_k=100,
    filter={
        "text_field": {"$contains": "keyword"},  # Text filtering
        "category": {"$in": ["category1", "category2"]}  # Metadata filtering
    }
)

# Execute hybrid search
results = await vector_store.search(query)
```

## Troubleshooting Questions

### What do I do if my IPFS connection fails?

If your IPFS connection fails:
1. Verify your IPFS node is running: `ipfs id`
2. Check API access: `curl http://localhost:5001/api/v0/version`
3. Ensure CORS is configured: `ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'`
4. Restart your IPFS node: `ipfs daemon`

### What are common DuckDB performance issues?

Common DuckDB performance issues:
1. **Insufficient memory allocation**: Increase memory_limit parameter
2. **Poor Parquet configuration**: Adjust row_group_size and page_size
3. **Missing indexes**: Add indexes for frequently filtered columns
4. **Large vector dimensions**: Consider using vector quantization

### My vector search returns incorrect results. What should I check?

If vector search results are incorrect:
1. Verify vector dimensions match between query and stored vectors
2. Check normalization of vectors (most similarity metrics expect normalized vectors)
3. Verify index configuration is appropriate for your data
4. Ensure vector quantization settings aren't too aggressive
5. Validate that the metric type matches your expectations (cosine, dot, L2)

### How do I debug the system?

To debug the system:
1. Enable debug logging in configuration
2. Use the diagnostic endpoints for system information
3. Check logs in the `logs` directory
4. Run the test scripts in debug mode
5. Use monitoring tools for performance metrics
