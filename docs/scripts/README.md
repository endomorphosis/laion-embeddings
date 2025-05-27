# Utility Scripts

The LAION Embeddings project includes several utility scripts that provide convenient shortcuts for common operations. These scripts are located in the project root directory and can be used to quickly perform various tasks without writing complex curl commands.

## Available Scripts

### Server Management

#### `run.sh`
Starts the FastAPI server with default configuration.

```bash
./run.sh
```

**What it does:**
- Starts the FastAPI server on port 9999
- Uses production settings for performance
- Enables automatic reload in development mode

### Data Loading

#### `load.sh`
Loads the Wikipedia-X-Concat dataset with embeddings.

```bash
./load.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/Wikipedia-X-Concat", "knn_index":"laion/Wikipedia-M3", "dataset_split": "enwiki_concat", "knn_index_split": "enwiki_embed", "columns": "Concat Abstract"}' \
    -H 'Content-Type: application/json'
```

**Parameters:**
- Dataset: `laion/Wikipedia-X-Concat`
- KNN Index: `laion/Wikipedia-M3`
- Split: `enwiki_concat` / `enwiki_embed`
- Columns: `Concat Abstract`

#### `load2.sh`
Loads the English-ConcatX-Abstract dataset.

```bash
./load2.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/English-ConcatX-Abstract", "knn_index":"laion/English-ConcatX-M3", "dataset_split": "train", "knn_index_split": "train", "columns": ["Concat Abstract","Title","URL"]}' \
    -H 'Content-Type: application/json'
```

**Parameters:**
- Dataset: `laion/English-ConcatX-Abstract`
- KNN Index: `laion/English-ConcatX-M3`
- Split: `train`
- Columns: `["Concat Abstract","Title","URL"]`

#### `load3.sh`
Loads the German-ConcatX-Abstract dataset.

```bash
./load3.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/German-ConcatX-Abstract", "knn_index":"laion/German-ConcatX-M3", "dataset_split": "train", "knn_index_split": "train", "columns": ["Concat Abstract","Title","URL"] }' \
    -H 'Content-Type: application/json'
```

**Parameters:**
- Dataset: `laion/German-ConcatX-Abstract`
- KNN Index: `laion/German-ConcatX-M3`
- Split: `train`
- Columns: `["Concat Abstract","Title","URL"]`

### Search Operations

#### `search.sh`
Performs a sample search query.

```bash
./search.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"orange juice", "collection": "Wikipedia-X-Concat"}' \
    -H 'Content-Type: application/json'
```

**Parameters:**
- Query text: `"orange juice"`
- Collection: `"Wikipedia-X-Concat"`

#### `search2.sh`
Performs additional search queries (if available).

```bash
./search2.sh
```

### Embedding Creation

#### `create.sh`
Creates dense embeddings from the Caselaw Access Project dataset.

```bash
./create.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:/create \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

**Parameters:**
- Dataset: `TeraflopAI/Caselaw_Access_Project`
- Split: `train`
- Text column: `text`
- Output path: `/storage/teraflopai/tmp`
- Models: `["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]`

**Output:** Dense embedding vectors stored in `./checkpoints/` directory

#### `create_sparse.sh`
Creates sparse embeddings using TF-IDF and BM25.

```bash
./create_sparse.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:/create_sparse \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

**Parameters:**
- Dataset: `TeraflopAI/Caselaw_Access_Project`
- Split: `train`
- Text column: `text`
- Output path: `/storage/teraflopai/tmp`
- Models: `["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]`

**Output:** Sparse embeddings stored in `./sparse_checkpoints/` directory

### Data Sharding

#### `shard_cluster.sh`
Shards large datasets into smaller chunks using K-means clustering.

```bash
./shard_cluster.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:/shard_cluster \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

**Parameters:**
- Dataset: `TeraflopAI/Caselaw_Access_Project`
- Split: `train`
- Text column: `text`
- Output path: `/storage/teraflopai/tmp`
- Models: `["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]`

**Features:**
- Creates shards no larger than 50MB
- Limits shards to maximum 4096 rows
- Uses semantic clustering for coherent shards

### IPFS Operations

#### `index_cluster.sh`
Indexes an IPFS cluster for content discovery.

```bash
./index_cluster.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:/index_cluster \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

**What it does:**
- Queries IPFS node for CID list
- Creates embedding indexes for CIDs
- Outputs results in `./checkpoints/` directory

### Storage Integration

#### `storacha.sh`
Uploads sharded data to the Storacha (Web3.Storage) network.

```bash
./storacha.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:/storacha \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

**Process:**
1. Converts Parquet files to CAR format
2. Uploads to Storacha network
3. Returns CIDs for decentralized access

### FAISS Integration

#### `autofaiss.sh`
Creates optimized FAISS indexes automatically.

```bash
./autofaiss.sh
```

**Equivalent curl command:**
```bash
curl 127.0.0.1:/autofaiss \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

**Features:**
- Automatic index type selection
- Memory-optimized configurations
- Performance-tuned parameters

### Endpoint Management

#### `launch_tei.sh`
Launches Text Embeddings Inference (TEI) Docker containers.

```bash
./launch_tei.sh
```

**Configuration:**
- Requires HuggingFace token in `hf_token` variable
- Uses GPU devices (CUDA_VISIBLE_DEVICES)
- Launches multiple model endpoints
- Configured for production workloads

**Example models launched:**
- `Alibaba-NLP/gte-large-en-v1.5` on port 8080, 8081
- `thenlper/gte-small` on port 8082, 8083
- `Alibaba-NLP/gte-Qwen2-1.5B-instruct` on port 8084, 8085

### Installation and Dependencies

#### `install_depends.sh`
Installs all required Python dependencies.

```bash
./install_depends.sh
```

**Installed packages:**
- Core ML: `torch`, `transformers`, `datasets`
- Vector databases: `faiss-cpu`, `qdrant-client`
- Web framework: `fastapi`, `uvicorn`
- IPFS: `multiformats`
- Search: `elasticsearch`, `rank_bm25`
- Additional utilities: `numpy`, `aiohttp`, `nltk`

## Script Customization

### Modifying Scripts

All scripts can be customized by editing the parameters:

```bash
# Example: Modify create.sh for different dataset
#!/bin/bash
curl 127.0.0.1:/create \
    -X POST \
    -d '["your/dataset", "split", "column", "/your/output/path", ["your-model"]]' \
    -H 'Content-Type: application/json'
```

### Common Parameters

Most scripts use these common parameter patterns:

1. **Dataset identifier** (HuggingFace dataset name)
2. **Split name** (train, test, validation)
3. **Text column** (column containing text data)
4. **Output path** (where to store results)
5. **Model list** (embedding models to use)

### Environment Variables

Scripts respect environment variables:

```bash
# Set custom server address
export SERVER_HOST=192.168.1.100
export SERVER_PORT=8000

# Use in script
curl ${SERVER_HOST}:${SERVER_PORT}/endpoint
```

## Usage Patterns

### Quick Workflow

```bash
# 1. Start server
./run.sh

# 2. Load sample data
./load.sh

# 3. Test search
./search.sh

# 4. Create embeddings for custom data
./create.sh
```

### Production Workflow

```bash
# 1. Launch TEI endpoints
./launch_tei.sh

# 2. Create dense embeddings
./create.sh

# 3. Create sparse embeddings
./create_sparse.sh

# 4. Shard large datasets
./shard_cluster.sh

# 5. Upload to decentralized storage
./storacha.sh
```

### Development Workflow

```bash
# 1. Install dependencies
./install_depends.sh

# 2. Start development server
./run.sh

# 3. Load test datasets
./load.sh
./load2.sh

# 4. Test functionality
./search.sh
```

## Error Handling

### Common Issues

1. **Server not running**: Ensure `./run.sh` is executed first
2. **Port conflicts**: Check if port 9999 is available
3. **Missing dependencies**: Run `./install_depends.sh`
4. **IPFS connection**: Verify IPFS daemon is running
5. **Storage permissions**: Check write access to output directories

### Debugging Scripts

Add debug output to scripts:

```bash
#!/bin/bash
set -x  # Enable debug output
curl -v 127.0.0.1:9999/endpoint \
    -X POST \
    -d '{"data": "value"}' \
    -H 'Content-Type: application/json'
```

### Timeout Configuration

For large datasets, increase timeouts:

```bash
curl --connect-timeout 60 --max-time 3600 \
    127.0.0.1:9999/endpoint \
    -X POST \
    -d '{"data": "value"}' \
    -H 'Content-Type: application/json'
```

## Security Considerations

### Production Usage

1. **Change default ports** in production environments
2. **Use HTTPS** for external access
3. **Implement authentication** for sensitive operations
4. **Restrict network access** to authorized IPs
5. **Monitor resource usage** to prevent abuse

### API Keys

For TEI endpoints requiring authentication:

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Modify launch_tei.sh
--api-key $HF_TOKEN
```

## Performance Optimization

### Batch Processing

For large datasets, use batch-optimized scripts:

```bash
# Create multiple smaller jobs
for i in {1..10}; do
    ./create.sh &
done
wait  # Wait for all jobs to complete
```

### Resource Monitoring

Monitor resource usage during script execution:

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU and memory
htop

# Monitor disk I/O
iotop
```

### Parallel Execution

Run independent scripts in parallel:

```bash
# Run embedding creation and sparse creation simultaneously
./create.sh &
./create_sparse.sh &
wait
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Process Embeddings
on:
  push:
    branches: [main]

jobs:
  embeddings:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: ./install_depends.sh
      - name: Start server
        run: ./run.sh &
      - name: Process data
        run: |
          ./create.sh
          ./create_sparse.sh
```

### Docker Integration

Use scripts in Docker containers:

```dockerfile
FROM python:3.9-slim

COPY . /app
WORKDIR /app

RUN ./install_depends.sh

CMD ["./run.sh"]
```

## Related Documentation

- [API Reference](../api/README.md) - Detailed endpoint documentation
- [Configuration Guide](../configuration.md) - Server and model configuration
- [Quick Start Guide](../quickstart.md) - Basic usage examples
- [Development Guide](../development.md) - Development environment setup
- [Examples](../examples/README.md) - Complete usage examples

## Support

For issues with utility scripts:

1. Check server logs: `tail -f server.log`
2. Verify endpoint availability: `curl http://localhost:9999/health`
3. Review script output for error messages
4. Consult the [troubleshooting guide](../troubleshooting/README.md)
