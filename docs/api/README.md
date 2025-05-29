# API Reference

This document provides comprehensive documentation for all LAION Embeddings API endpoints.

## Recent Updates (May 28, 2025)

- **Enhanced Error Handling**: All endpoints now include robust error handling with validated tokenization workflows
- **Workflow Validation**: New validation endpoints for testing tokenization pipelines
- **CID Validation**: Content identifiers are now validated throughout the processing pipeline
- **Production-Ready Processing**: All text processing includes safe_* function implementations

## Base URL

```
http://localhost:9999
```

## Authentication

Currently, the API does not require authentication. This may change in future versions.

## Content Type

All API requests should include:
```
Content-Type: application/json
```

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/validate_workflow` | POST | Validate tokenization workflow |
| `/load` | POST | Load dataset and KNN index |
| `/search` | POST | Search embeddings |
| `/create_embeddings` | POST | Create embeddings for a dataset |
| `/sparse_embeddings` | POST | Create sparse embeddings |
| `/shard_embeddings` | POST | Shard embeddings across nodes |
| `/ipfs_cluster_index` | POST | Index with IPFS cluster |
| `/storacha_clusters` | POST | Manage Storacha clusters |
| `/add_endpoint` | POST | Add new inference endpoint |

## Detailed API Documentation

### Health Check

**GET** `/health`

Check the health status of the API server.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "endpoints_available": true,
  "models_loaded": ["thenlper/gte-small"],
  "ipfs_connected": true
}
```

### Validate Tokenization Workflow

**POST** `/validate_workflow`

Validate the complete tokenization workflow to ensure all components are working correctly.

**Request Body:**
```json
{
  "text": "Sample text for validation",
  "model": "thenlper/gte-small",
  "chunk_size": 256,
  "validate_full_pipeline": true
}
```

**Parameters:**
- `text` (string, required): Test text for validation
- `model` (string, optional): Model to use for validation (default: "thenlper/gte-small")
- `chunk_size` (integer, optional): Chunk size for testing (default: 256)
- `validate_full_pipeline` (boolean, optional): Whether to validate the complete pipeline (default: true)

**Response:**
```json
{
  "status": "success",
  "validation_results": {
    "tokenization": {
      "encode_success": true,
      "decode_success": true,
      "token_count": 12
    },
    "chunking": {
      "success": true,
      "chunk_count": 1,
      "chunks_validated": true
    },
    "cid_generation": {
      "success": true,
      "cid": "bafkreigh2akiscaildcqabsyg3dfr6chu732xdkw6lg3cw4o6nchcd3k7u",
      "cid_validated": true
    },
    "workflow_sequence": {
      "success": true,
      "processing_time_ms": 45
    }
  },
  "message": "All workflow components validated successfully"
}
```

**Error Response:**
```json
{
  "status": "error",
  "validation_results": {
    "tokenization": {
      "encode_success": false,
      "error": "Tokenization failed: Invalid input"
    }
  },
  "message": "Workflow validation failed"
}
```

### Load Index

**POST** `/load`

Load a dataset and its corresponding KNN index for searching.

**Request Body:**
```json
{
  "dataset": "laion/Wikipedia-X-Concat",
  "knn_index": "laion/Wikipedia-M3", 
  "dataset_split": "enwiki_concat",
  "knn_index_split": "enwiki_embed",
  "columns": ["Concat Abstract"]
}
```

**Parameters:**
- `dataset` (string, required): HuggingFace dataset identifier
- `knn_index` (string, required): KNN index dataset identifier  
- `dataset_split` (string, optional): Dataset split to use
- `knn_index_split` (string, optional): Index split to use
- `columns` (array, required): List of column names to index

**Response:**
```json
{
  "message": "Index loading started",
  "task_id": "load_task_123",
  "estimated_time": "2-4 hours"
}
```

**Notes:**
- This operation runs in the background
- The API will be unavailable during loading for large datasets
- Progress can be monitored through server logs

### Search Embeddings

**POST** `/search`

Perform semantic search on loaded embeddings.

**Request Body:**
```json
{
  "collection": "wikipedia",
  "text": "artificial intelligence machine learning",
  "n": 10
}
```

**Parameters:**
- `collection` (string, required): Name of the loaded collection
- `text` (string, required): Query text for semantic search
- `n` (integer, required): Number of results to return (max 100)

**Response:**
```json
{
  "results": [
    {
      "id": "doc_1234",
      "score": 0.95,
      "text": "Artificial intelligence (AI) is intelligence...",
      "metadata": {
        "title": "Artificial Intelligence",
        "source": "wikipedia"
      }
    }
  ],
  "query_time_ms": 45,
  "total_results": 10,
  "collection": "wikipedia"
}
```

### Create Embeddings

**POST** `/create_embeddings`

Generate embeddings for a dataset using specified models.

**Request Body:**
```json
{
  "dataset": "my_org/my_dataset",
  "split": "train",
  "column": "text",
  "dst_path": "./embeddings_output",
  "models": ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5"]
}
```

**Parameters:**
- `dataset` (string, required): HuggingFace dataset identifier
- `split` (string, required): Dataset split to process
- `column` (string, required): Text column to embed
- `dst_path` (string, required): Output directory path
- `models` (array, required): List of embedding models to use

**Response:**
```json
{
  "message": "Embedding creation started",
  "task_id": "create_task_456",
  "dataset": "my_org/my_dataset",
  "models": ["thenlper/gte-small"],
  "estimated_items": 50000
}
```

### Create Sparse Embeddings

**POST** `/sparse_embeddings`

Create sparse embeddings for large-scale datasets.

**Request Body:**
```json
{
  "dataset": "large_dataset", 
  "split": "train",
  "column": "content",
  "dst_path": "./sparse_output",
  "models": ["thenlper/gte-small"]
}
```

**Parameters:**
- `dataset` (string, required): Dataset identifier
- `split` (string, required): Dataset split
- `column` (string, required): Text column name
- `dst_path` (string, required): Output path
- `models` (array, required): Models to use

**Response:**
```json
{
  "message": "Sparse embedding creation started",
  "task_id": "sparse_task_789",
  "chunking_strategy": "adaptive",
  "estimated_chunks": 10000
}
```

### Shard Embeddings

**POST** `/shard_embeddings`

Distribute embeddings across multiple nodes for scalability.

**Request Body:**
```json
{
  "dataset": "massive_dataset",
  "split": "train", 
  "column": "text",
  "dst_path": "./sharded_output",
  "models": ["thenlper/gte-small"],
  "shard_count": 10
}
```

**Parameters:**
- `dataset` (string, required): Dataset to shard
- `split` (string, required): Dataset split
- `column` (string, required): Text column
- `dst_path` (string, required): Output directory
- `models` (array, required): Embedding models
- `shard_count` (integer, optional): Number of shards (default: auto)

**Response:**
```json
{
  "message": "Sharding started",
  "task_id": "shard_task_101",
  "shard_count": 10,
  "items_per_shard": 5000
}
```

### IPFS Cluster Index

**POST** `/ipfs_cluster_index`

Index data using IPFS cluster for distributed storage.

**Request Body:**
```json
{
  "resources": {
    "cluster_id": "my_cluster",
    "replication_factor": 3
  },
  "metadata": {
    "dataset": "distributed_dataset",
    "description": "Large scale embedding index"
  }
}
```

**Parameters:**
- `resources` (object, required): Cluster configuration
  - `cluster_id` (string): Cluster identifier
  - `replication_factor` (integer): Data replication count
- `metadata` (object, required): Index metadata
  - `dataset` (string): Dataset name
  - `description` (string): Index description

**Response:**
```json
{
  "message": "IPFS cluster indexing started",
  "cluster_id": "my_cluster", 
  "ipfs_hash": "QmX...",
  "replication_status": "pending"
}
```

### Storacha Clusters

**POST** `/storacha_clusters`

Manage Storacha cluster operations for decentralized storage.

**Request Body:**
```json
{
  "resources": {
    "cluster_name": "storacha_cluster_1",
    "storage_quota": "1TB"
  },
  "metadata": {
    "purpose": "embedding_storage",
    "retention_policy": "1_year"
  }
}
```

**Response:**
```json
{
  "message": "Storacha cluster operation initiated",
  "cluster_name": "storacha_cluster_1",
  "status": "provisioning",
  "endpoints": ["https://node1.storacha.io", "https://node2.storacha.io"]
}
```

### Add Endpoint

**POST** `/add_endpoint`

Add a new inference endpoint for load balancing.

**Request Body:**
```json
{
  "model": "thenlper/gte-small",
  "endpoint": "http://localhost:8080/embed",
  "type": "tei",
  "ctx_length": 512
}
```

**Parameters:**
- `model` (string, required): Model identifier
- `endpoint` (string, required): Endpoint URL
- `type` (string, required): Endpoint type (`tei`, `openvino`, `libp2p`, `local`, `cuda`)
- `ctx_length` (integer, required): Maximum context length

**Response:**
```json
{
  "message": "Endpoint added successfully",
  "model": "thenlper/gte-small",
  "endpoint": "http://localhost:8080/embed",
  "type": "tei",
  "status": "active"
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "error": "Invalid request",
  "message": "Missing required parameter: dataset",
  "code": 400
}
```

### 404 Not Found
```json
{
  "error": "Not found", 
  "message": "Collection 'xyz' not found",
  "code": 404
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "Model loading failed",
  "code": 500
}
```

## Rate Limiting

- **Search**: 100 requests per minute per IP
- **Create/Load**: 5 requests per hour per IP
- **Other**: 1000 requests per hour per IP

Exceeded limits return HTTP 429:
```json
{
  "error": "Rate limit exceeded",
  "retry_after_seconds": 60
}
```

## Response Times

Typical response times:

| Operation | Expected Time |
|-----------|---------------|
| Health Check | < 10ms |
| Search (10 results) | 50-200ms |
| Add Endpoint | < 100ms |
| Load Index | 1-4 hours* |
| Create Embeddings | 10min-2hours* |

*Depends on dataset size

## SDK Examples

### Python
```python
import requests

# Search example
response = requests.post(
    "http://localhost:9999/search",
    json={
        "collection": "wikipedia",
        "text": "machine learning",
        "n": 5
    }
)
results = response.json()
```

### JavaScript
```javascript
// Create embeddings example
const response = await fetch('http://localhost:9999/create_embeddings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        dataset: 'my_dataset',
        split: 'train',
        column: 'text',
        dst_path: './output',
        models: ['thenlper/gte-small']
    })
});
const result = await response.json();
```

### cURL
```bash
# Load index example
curl -X POST "http://localhost:9999/load" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "laion/Wikipedia-X-Concat",
    "knn_index": "laion/Wikipedia-M3",
    "dataset_split": "enwiki_concat", 
    "knn_index_split": "enwiki_embed",
    "columns": ["Concat Abstract"]
  }'
```

## Webhooks (Future)

Planned webhook support for long-running operations:

```json
{
  "webhook_url": "https://your-app.com/webhooks/embeddings",
  "events": ["embedding_complete", "index_loaded", "error"]
}
```

## Changelog

### v1.0.0 (Current)
- Initial API release
- Basic CRUD operations
- IPFS integration
- Multi-model support

### Planned Features
- Authentication & authorization
- Batch operations API
- Real-time streaming
- GraphQL interface
- Advanced filtering
