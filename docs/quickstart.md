# Quick Start Guide

Get up and running with LAION Embeddings in just a few minutes!

## Prerequisites

Make sure you have completed the [installation process](installation.md) before proceeding.

## 1. Start the Server

### Using the Run Script
```bash
./run.sh
```

### Manual Start
```bash
python3 -m fastapi run main.py
```

The server will start on `http://localhost:9999` by default.

## 2. Verify Installation

Check if the API is running:

```bash
curl http://localhost:9999/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "endpoints_available": true,
  "models_loaded": ["thenlper/gte-small"]
}
```

## 3. Basic Operations

### Create Your First Embedding

```bash
curl -X POST "http://localhost:9999/create_embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "sample_dataset",
    "split": "train", 
    "column": "text",
    "dst_path": "./embeddings_output",
    "models": ["thenlper/gte-small"]
  }'
```

### Load an Index

Load a pre-existing dataset and KNN index:

```bash
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

⚠️ **Note**: This operation may take several hours for large datasets and will make the FastAPI unavailable during processing.

### Search the Index

Once your index is loaded, perform semantic search:

```bash
curl -X POST "http://localhost:9999/search" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "wikipedia",
    "text": "artificial intelligence machine learning",
    "n": 10
  }'
```

Example response:
```json
{
  "results": [
    {
      "id": "doc_1234",
      "score": 0.95,
      "text": "Artificial intelligence (AI) is intelligence demonstrated by machines...",
      "metadata": {
        "title": "Artificial Intelligence",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
      }
    }
  ],
  "query_time_ms": 45,
  "total_results": 10
}
```

## 4. Working with Models

### List Available Models

The system supports several embedding models:

- `thenlper/gte-small` (384 dimensions, fast)
- `Alibaba-NLP/gte-large-en-v1.5` (1024 dimensions, high quality)
- `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (1536 dimensions, latest)

### Add a New Endpoint

```bash
curl -X POST "http://localhost:9999/add_endpoint" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "thenlper/gte-small",
    "endpoint": "http://localhost:8080/embed",
    "type": "tei",
    "ctx_length": 512
  }'
```

Supported endpoint types:
- `tei` - Text Embeddings Inference
- `openvino` - Intel OpenVINO
- `libp2p` - Peer-to-peer
- `local` - Local inference
- `cuda` - GPU-accelerated

## 5. Example Workflows

### Workflow 1: Index a Custom Dataset

1. **Prepare your data** in HuggingFace datasets format
2. **Create embeddings**:
   ```bash
   curl -X POST "http://localhost:9999/create_embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset": "my_org/my_dataset",
       "split": "train",
       "column": "content", 
       "dst_path": "./my_embeddings",
       "models": ["thenlper/gte-small"]
     }'
   ```

3. **Wait for completion** (check logs for progress)
4. **Search your data** using the search endpoint

### Workflow 2: Distributed Processing

1. **Set up multiple endpoints** for load balancing
2. **Create sparse embeddings** for large datasets:
   ```bash
   curl -X POST "http://localhost:9999/sparse_embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset": "large_dataset",
       "split": "train",
       "column": "text",
       "dst_path": "./sparse_output", 
       "models": ["thenlper/gte-small"]
     }'
   ```

### Workflow 3: IPFS Clustering

1. **Index with IPFS cluster**:
   ```bash
   curl -X POST "http://localhost:9999/ipfs_cluster_index" \
     -H "Content-Type: application/json" \
     -d '{
       "resources": {"cluster_id": "my_cluster"},
       "metadata": {"dataset": "my_dataset"}
     }'
   ```

## 6. Monitoring and Logs

### Check Server Logs
```bash
# If running with docker
docker logs <container_id>

# If running directly
tail -f server.log
```

### Monitor Resource Usage
```bash
# CPU and memory
htop

# GPU usage (if available)
nvidia-smi

# IPFS status
ipfs stats bw
```

## 7. Configuration Tips

### Performance Optimization

1. **Adjust batch sizes** based on your hardware:
   ```python
   # In your configuration
   BATCH_SIZE = 32  # Reduce if you have memory issues
   ```

2. **Use appropriate models** for your use case:
   - Small datasets: `gte-small`
   - High accuracy needs: `gte-large-en-v1.5`
   - Latest features: `gte-Qwen2-1.5B-instruct`

3. **Configure endpoints** for your infrastructure:
   - Local processing: Use `local` endpoints
   - GPU available: Use `cuda` endpoints
   - Distributed setup: Use `libp2p` endpoints

### Memory Management

```bash
# Monitor memory usage
free -h

# Adjust Python memory limits
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
```

## 8. Next Steps

Now that you have the basics working:

1. [Explore the API Reference](api/README.md) for detailed endpoint documentation
2. [Learn about Configuration](configuration.md) for advanced setup
3. [Read about Components](components/README.md) to understand the architecture
4. [Check Examples](examples/basic-usage.md) for more complex scenarios

## Common Commands Reference

```bash
# Start server
./run.sh

# Load example dataset
./load.sh

# Search example
./search.sh

# Create embeddings example  
./create.sh

# Health check
curl http://localhost:9999/health

# Stop server
pkill -f "fastapi run main.py"
```

## Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| Port 9999 in use | `lsof -i :9999` then `kill -9 <PID>` |
| CUDA out of memory | Reduce batch size in configuration |
| IPFS connection failed | Restart IPFS daemon: `ipfs daemon` |
| Model download slow | Use local model cache or mirror |
| API timeout | Increase timeout in client requests |

## Getting Help

- [Troubleshooting Guide](troubleshooting/common-issues.md)
- [API Documentation](api/README.md) 
- [GitHub Issues](https://github.com/laion-ai/embeddings/issues)
- [Community Discussions](https://github.com/laion-ai/embeddings/discussions)
