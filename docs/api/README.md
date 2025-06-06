# API Reference

This document provides comprehensive documentation for all FastAPI endpoints and MCP tools in the LAION Embeddings system.

## FastAPI Endpoints

The system provides 17 RESTful API endpoints organized into the following categories:

### 🏥 Health & Status Endpoints

#### `GET /health`
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-06T12:00:00Z"
}
```

#### `GET /health/detailed`
Detailed health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "vector_store": "healthy",
    "ipfs": "healthy"
  },
  "timestamp": "2025-06-06T12:00:00Z"
}
```

#### `GET /`
Root endpoint with system information.

### 🔮 Embedding Generation

#### `POST /create_embeddings`
Generate embeddings from text input.

**Request Body:**
```json
{
  "texts": ["example text", "another example"],
  "model": "thenlper/gte-small",
  "normalize": true
}
```

**Response:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "model": "thenlper/gte-small",
  "dimensions": 384
}
```

### 🔍 Search Endpoints

#### `POST /search`
Perform semantic search using embeddings.

**Request Body:**
```json
{
  "text": "search query",
  "collection": "my_collection",
  "limit": 10,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc_1",
      "score": 0.95,
      "metadata": {"title": "Document 1"},
      "text": "relevant content"
    }
  ],
  "query_time_ms": 25
}
```

### 📂 Index Management

#### `POST /load`
Load a dataset into the vector store.

**Request Body:**
```json
{
  "dataset": "laion/Wikipedia-X-Concat",
  "knn_index": "laion/Wikipedia-M3",
  "dataset_split": "enwiki_concat",
  "knn_index_split": "enwiki_embed",
  "column": "Concat Abstract"
}
```

#### `POST /shard_embeddings`
Shard embeddings across multiple nodes.

**Request Body:**
```json
{
  "collection": "my_collection",
  "num_shards": 4,
  "strategy": "balanced"
}
```

### 🕷️ Sparse Embeddings

#### `POST /index_sparse_embeddings`
Create sparse embeddings using TF-IDF or BM25.

**Request Body:**
```json
{
  "texts": ["document content"],
  "method": "tfidf",
  "max_features": 10000
}
```

### 💾 Storage Operations

#### `POST /index_cluster`
Index data in IPFS cluster.

**Request Body:**
```json
{
  "host": "localhost",
  "collection": "my_collection",
  "content_type": "text",
  "output_path": "/tmp/output",
  "models": ["thenlper/gte-small"]
}
```

#### `GET /storacha_clusters`
List available Storacha clusters.

### 🗄️ Cache Management

#### `GET /cache/stats`
Get cache statistics.

**Response:**
```json
{
  "total_entries": 1000,
  "hit_rate": 0.85,
  "memory_usage_mb": 256
}
```

#### `DELETE /cache/clear`
Clear the cache.

### 🔐 Authentication

#### `POST /auth/login`
User authentication.

**Request Body:**
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### `GET /auth/me`
Get current user information (requires authentication).

### 📊 Monitoring

#### `GET /metrics`
Prometheus-formatted metrics.

#### `GET /metrics/json`
JSON-formatted metrics.

**Response:**
```json
{
  "requests_total": 1000,
  "requests_per_second": 10.5,
  "average_response_time_ms": 150,
  "error_rate": 0.01
}
```

### ⚙️ Administration

#### `POST /add_endpoint`
Add a new endpoint configuration.

**Request Body:**
```json
{
  "name": "my_endpoint",
  "url": "http://localhost:8080",
  "type": "embedding"
}
```

## Model Context Protocol (MCP) Tools

The system provides 40+ MCP tools that expose all FastAPI functionality plus additional capabilities to AI assistants.

### 🔮 Embedding Tools

**EmbeddingGenerationTool**
- Generate embeddings from text
- Supports multiple models
- Batch processing capability

**BatchEmbeddingTool**
- Process large batches of text
- Optimized for throughput
- Progress tracking

**MultimodalEmbeddingTool**
- Handle text, images, and other modalities
- Cross-modal search capabilities

### 🔍 Search Tools

**SemanticSearchTool**
- Vector similarity search
- Metadata filtering
- Relevance scoring

**SimilaritySearchTool**
- Direct vector similarity
- Configurable distance metrics
- Threshold filtering

**FacetedSearchTool**
- Multi-dimensional search
- Category-based filtering
- Aggregated results

### 💾 Storage Tools

**StorageManagementTool**
- Create and manage collections
- Storage optimization
- Backup and restore

**CollectionManagementTool**
- Collection lifecycle management
- Metadata management
- Access control

**RetrievalTool**
- Efficient document retrieval
- Batch operations
- Caching support

### 📊 Analysis Tools

**ClusterAnalysisTool**
- Discover data clusters
- Quality metrics
- Visualization support

**QualityAssessmentTool**
- Embedding quality analysis
- Performance benchmarks
- Anomaly detection

**DimensionalityReductionTool**
- Reduce vector dimensions
- Visualization preparation
- Performance optimization

### 🏪 Vector Store Tools

**VectorIndexTool**
- Index management
- Performance tuning
- Shard coordination

**VectorRetrievalTool**
- Optimized retrieval
- Parallel processing
- Result ranking

**VectorMetadataTool**
- Metadata operations
- Schema management
- Data validation

### 🌐 IPFS Cluster Tools

**IPFSClusterTool**
- Distributed storage
- Node management
- Network coordination

**DistributedVectorTool**
- Cross-node operations
- Load balancing
- Fault tolerance

**IPFSMetadataTool**
- Distributed metadata
- Consistency checks
- Replication management

### 🕷️ Sparse Embedding Tools

**SparseIndexingTool**
- TF-IDF indexing
- BM25 scoring
- Feature selection

**SparseSearchTool**
- Keyword-based search
- Boolean operations
- Term weighting

**SparseCombinationTool**
- Hybrid search (dense + sparse)
- Score combination
- Result fusion

### 🔐 Authentication Tools

**LoginTool**
- User authentication
- Session management
- Token generation

**UserManagementTool**
- User creation and management
- Permission control
- Profile management

**SessionTool**
- Session lifecycle
- Security validation
- Activity tracking

### 🗄️ Cache Tools

**CacheStatsTool**
- Performance monitoring
- Hit rate analysis
- Memory usage tracking

**CacheClearTool**
- Selective clearing
- Policy management
- Optimization triggers

**CacheOptimizationTool**
- Performance tuning
- Memory management
- Eviction strategies

### 📊 Monitoring Tools

**HealthCheckTool**
- System health monitoring
- Component status
- Alerting support

**MetricsTool**
- Performance metrics
- Custom dashboards
- Historical data

**PerformanceTool**
- Benchmarking
- Load testing
- Optimization recommendations

**AlertingTool**
- Notification management
- Threshold monitoring
- Escalation policies

### ⚙️ Admin Tools

**ConfigurationTool**
- System configuration
- Runtime adjustments
- Feature toggles

**EndpointManagementTool**
- Endpoint lifecycle
- Load balancing
- Health monitoring

**SystemMaintenanceTool**
- Maintenance operations
- System optimization
- Cleanup tasks

### 📁 Index Management Tools

**IndexLoadingTool**
- Index initialization
- Data loading
- Validation checks

**IndexOptimizationTool**
- Performance tuning
- Memory optimization
- Query acceleration

### 👤 Session Management Tools

**SessionCreationTool**
- New session initialization
- User context setup
- Resource allocation

**SessionStateTool**
- State management
- Persistence handling
- Context switching

**SessionCleanupTool**
- Resource cleanup
- Memory management
- Session archival

### 🔄 Workflow Tools

**WorkflowExecutionTool**
- Multi-step operations
- State management
- Error recovery

**BatchProcessingTool**
- Large-scale operations
- Progress tracking
- Resource optimization

**DataPipelineTool**
- ETL operations
- Data transformation
- Quality validation

**AutomationTool**
- Scheduled operations
- Event-driven workflows
- Rule-based processing

**IntegrationTool**
- External system integration
- API coordination
- Data synchronization

**ValidationTool**
- Data quality checks
- Schema validation
- Consistency verification

## Error Handling

All endpoints return standardized error responses:

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Invalid input parameters",
    "details": {
      "field": "text",
      "reason": "Field is required"
    }
  },
  "timestamp": "2025-06-06T12:00:00Z"
}
```

## Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **Standard endpoints**: 100 requests/minute
- **Search endpoints**: 50 requests/minute  
- **Heavy operations**: 10 requests/minute

## Authentication

Authentication is required for:
- Admin endpoints
- User-specific operations
- Rate limit overrides

Use the `/auth/login` endpoint to obtain a JWT token, then include it in subsequent requests:

```
Authorization: Bearer <token>
```

## SDKs and Client Libraries

Official client libraries are available for:
- Python: `pip install laion-embeddings-client`
- JavaScript/TypeScript: `npm install laion-embeddings-js`
- Go: `go get github.com/laion/embeddings-go`

## Examples

For complete usage examples and tutorials, see the [Examples Documentation](../examples/README.md).