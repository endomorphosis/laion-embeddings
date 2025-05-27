# Create Embeddings Component

The Create Embeddings component is responsible for generating dense vector embeddings from text datasets using various embedding models. It handles the complete pipeline from dataset loading to embedding creation and storage.

## Overview

The `create_embeddings` class provides a high-level interface for:
- Loading datasets from HuggingFace or local sources
- Processing text data through embedding models
- Managing multiple embedding endpoints
- Storing embeddings to disk or IPFS
- Handling large-scale batch processing

## Key Features

### Multi-Model Support
- **Simultaneous Processing**: Generate embeddings with multiple models in parallel
- **Model Endpoints**: Support for local and remote embedding endpoints
- **Tokenization**: Automatic handling of different tokenizer requirements

### Scalable Processing
- **Batch Processing**: Efficient handling of large datasets
- **Async Operations**: Non-blocking processing for better performance
- **Resource Management**: Automatic load balancing across endpoints

### Flexible Storage
- **Local Storage**: Save embeddings to local filesystem
- **IPFS Integration**: Direct upload to IPFS for distributed storage
- **Metadata Preservation**: Maintain dataset metadata and structure

## Usage

### Basic Usage

```python
from create_embeddings import create_embeddings
import asyncio

# Configuration
metadata = {
    "dataset": "TeraflopAI/Caselaw_Access_Project",
    "split": "train",
    "column": "text",
    "models": [
        "Alibaba-NLP/gte-large-en-v1.5",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    ],
    "dst_path": "/storage/embeddings"
}

resources = {
    "https_endpoints": [
        ["Alibaba-NLP/gte-large-en-v1.5", "http://localhost:8080/embed", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://localhost:8081/embed", 32768]
    ]
}

# Initialize component
embedder = create_embeddings(resources, metadata)

# Create embeddings
await embedder.create_embeddings(
    dataset="your-dataset",
    split="train",
    column="text",
    dst_path="/path/to/output",
    models=["model1", "model2"]
)
```

### Advanced Configuration

```python
# Custom endpoint management
embedder.add_https_endpoint(
    model="custom-model",
    endpoint="http://custom-server:8080/embed",
    ctx_length=4096
)

# Direct call interface
await embedder(
    dataset="dataset-name",
    split="validation",
    column="content",
    dst_path="/custom/path",
    models=["model1"]
)
```

## Configuration

### Metadata Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `dataset` | str | HuggingFace dataset name or local path | Yes |
| `split` | str | Dataset split (train, test, validation) | Yes |
| `column` | str | Text column name to embed | Yes |
| `models` | list | List of embedding model names | Yes |
| `dst_path` | str | Output directory path | Yes |

### Resource Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `https_endpoints` | list | List of [model, endpoint, ctx_length] tuples | No |

### Endpoint Configuration

Each endpoint in `https_endpoints` should specify:
- **Model Name**: Must match the model in metadata
- **Endpoint URL**: HTTP endpoint for embedding generation
- **Context Length**: Maximum tokens the model can process

## API Reference

### Class: `create_embeddings`

#### `__init__(resources, metadata)`
Initialize the create embeddings component.

**Parameters:**
- `resources` (dict): Resource configuration including endpoints
- `metadata` (dict): Processing metadata and parameters

#### `add_https_endpoint(model, endpoint, ctx_length)`
Add a new HTTPS endpoint for embedding generation.

**Parameters:**
- `model` (str): Model name
- `endpoint` (str): HTTP endpoint URL
- `ctx_length` (int): Maximum context length

**Returns:**
- Result from underlying IPFS embeddings module

#### `create_embeddings(dataset, split, column, dst_path, models)`
Generate embeddings for the specified dataset.

**Parameters:**
- `dataset` (str): Dataset identifier
- `split` (str): Dataset split
- `column` (str): Text column name
- `dst_path` (str): Output path
- `models` (list): List of model names

**Returns:**
- None (embeddings saved to specified path)

#### `__call__(dataset, split, column, dst_path, models)`
Alternative interface for creating embeddings (same as `create_embeddings`).

#### `test(dataset, split, column, dst_path, models)`
Test method with predefined endpoints for development/testing.

## Integration Examples

### Custom Dataset Processing

```python
# Process a custom dataset
metadata = {
    "dataset": "my-custom-dataset",
    "split": "train", 
    "column": "document_text",
    "models": ["sentence-transformers/all-MiniLM-L6-v2"],
    "dst_path": "/data/embeddings"
}

resources = {
    "https_endpoints": [
        ["sentence-transformers/all-MiniLM-L6-v2", "http://localhost:8080/embed", 512]
    ]
}

embedder = create_embeddings(resources, metadata)
await embedder.create_embeddings(**metadata)
```

### Multiple Model Comparison

```python
# Generate embeddings with multiple models for comparison
models = [
    "Alibaba-NLP/gte-large-en-v1.5",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "thenlper/gte-small"
]

# Each model needs corresponding endpoint
resources = {
    "https_endpoints": [
        ["Alibaba-NLP/gte-large-en-v1.5", "http://server1:8080/embed", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://server2:8081/embed", 32768],
        ["thenlper/gte-small", "http://server3:8082/embed", 512]
    ]
}

await embedder.create_embeddings(
    dataset="comparison-dataset",
    split="test",
    column="text",
    dst_path="/embeddings/comparison",
    models=models
)
```

## Performance Optimization

### Endpoint Load Balancing
The component automatically distributes requests across multiple endpoints for the same model:

```python
# Multiple endpoints for the same model for load balancing
resources = {
    "https_endpoints": [
        ["gte-large", "http://server1:8080/embed", 8192],
        ["gte-large", "http://server2:8080/embed", 8192],
        ["gte-large", "http://server3:8080/embed", 8192]
    ]
}
```

### Batch Size Optimization
- Adjust batch sizes based on model context length
- Consider GPU memory when setting context lengths
- Monitor endpoint response times for optimal throughput

## Error Handling

The component includes robust error handling for:
- **Network Issues**: Automatic retry for failed endpoint requests
- **Model Errors**: Graceful handling of model-specific failures
- **Storage Errors**: Validation of output paths and permissions
- **Dataset Errors**: Proper handling of malformed or missing data

## Dependencies

- `aiohttp`: Async HTTP client for endpoint communication
- `datasets`: HuggingFace datasets library
- `transformers`: Tokenizer management
- `ipfs_embeddings_py`: Core embedding processing library

## Related Components

- [Search Embeddings](search-embeddings.md): For searching generated embeddings
- [IPFS Integration](../ipfs/README.md): For distributed storage
- [Models](../models/README.md): For supported embedding models

## Troubleshooting

### Common Issues

1. **Endpoint Timeout**
   - Increase timeout values in HTTP client
   - Check endpoint availability and load

2. **Memory Issues**
   - Reduce batch size
   - Use smaller context lengths
   - Process datasets in chunks

3. **Storage Issues**
   - Verify disk space availability
   - Check write permissions
   - Validate output path format

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export EMBEDDING_DEBUG=1
export EMBEDDING_LOG_LEVEL=DEBUG
```
