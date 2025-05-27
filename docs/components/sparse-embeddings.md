# Sparse Embeddings Component

The Sparse Embeddings component provides functionality for creating and managing sparse vector representations of text data. Unlike dense embeddings, sparse embeddings maintain interpretability by using explicit feature dimensions, making them useful for keyword-based search and explainable AI applications.

## Overview

The `sparse_embeddings` class offers:
- **Sparse Vector Generation**: Create sparse representations from text
- **Efficient Storage**: Optimized storage for sparse data structures
- **Hybrid Search**: Combine with dense embeddings for improved search
- **Interpretability**: Maintain human-readable feature mappings

## Key Features

### Sparse Vector Processing
- **TF-IDF Vectors**: Traditional term frequency-inverse document frequency
- **BM25 Scoring**: Best matching scoring for information retrieval
- **Custom Vocabularies**: Support for domain-specific vocabularies
- **Feature Selection**: Automatic selection of important features

### Integration with Dense Embeddings
- **Hybrid Search**: Combine sparse and dense representations
- **Score Fusion**: Multiple fusion strategies for combining scores
- **Model Ensemble**: Use multiple sparse and dense models together

### Scalable Processing
- **Batch Processing**: Handle large datasets efficiently
- **Memory Optimization**: Efficient sparse matrix representations
- **Distributed Computing**: Support for multi-node processing

## Usage

### Basic Sparse Embedding Creation

```python
from sparse_embeddings import sparse_embeddings

# Configuration
metadata = {
    "dataset": "TeraflopAI/Caselaw_Access_Project",
    "column": "text",
    "split": "train",
    "models": ["thenlper/gte-small"],
    "chunk_settings": {
        "chunk_size": 512,
        "n_sentences": 8,
        "step_size": 256,
        "method": "fixed",
        "embed_model": "thenlper/gte-small",
        "tokenizer": None
    },
    "dst_path": "/storage/sparse_embeddings"
}

resources = {
    "https_endpoints": [
        ["thenlper/gte-small", "http://localhost:8080/embed-tiny", 512]
    ]
}

# Initialize component
sparse_emb = sparse_embeddings(resources, metadata)

# Create sparse embeddings
sparse_emb.index_sparse_embeddings(embeddings_data)
```

### Dataset Indexing

```python
# Index a complete dataset
result = sparse_emb.index_dataset("my-dataset")

# The component will automatically:
# 1. Load the dataset
# 2. Process text into sparse vectors
# 3. Store results for fast retrieval
```

### Testing and Validation

```python
# Run component tests
test_results = sparse_emb.test()
print(test_results)

# Results include:
# - IPFS embeddings initialization status
# - Sparse embedding processing status
# - Error details if any issues occur
```

## Configuration

### Metadata Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `dataset` | str | Dataset name or path | Yes |
| `column` | str | Text column to process | Yes |
| `split` | str | Dataset split (train/test/validation) | Yes |
| `models` | list | List of embedding models to use | Yes |
| `chunk_settings` | dict | Text chunking configuration | Yes |
| `dst_path` | str | Output directory for sparse embeddings | Yes |

### Chunk Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `chunk_size` | int | Maximum tokens per chunk | 512 |
| `n_sentences` | int | Maximum sentences per chunk | 8 |
| `step_size` | int | Overlap between chunks | 256 |
| `method` | str | Chunking method (fixed/adaptive) | "fixed" |
| `embed_model` | str | Model for chunk processing | Required |
| `tokenizer` | object | Custom tokenizer (optional) | None |

### Resource Configuration

Resources specify the embedding endpoints:

```python
resources = {
    "https_endpoints": [
        [model_name, endpoint_url, context_length],
        # Multiple endpoints for load balancing
        ["thenlper/gte-small", "http://server1:8080/embed-tiny", 512],
        ["thenlper/gte-small", "http://server2:8080/embed-tiny", 512]
    ]
}
```

## API Reference

### Class: `sparse_embeddings`

#### `__init__(resources, metadata)`
Initialize the sparse embeddings component.

**Parameters:**
- `resources` (dict): Endpoint and resource configuration
- `metadata` (dict): Processing parameters and settings

#### `index_dataset(dataset)`
Process and index a complete dataset for sparse embeddings.

**Parameters:**
- `dataset` (str): Dataset identifier or path

**Returns:**
- Processing result from the underlying IPFS embeddings system

#### `index_sparse_embeddings(embeddings)`
Process pre-computed embeddings into sparse format.

**Parameters:**
- `embeddings` (array/list): Dense embeddings to convert

**Returns:**
- Indexed sparse embedding structure

#### `test()`
Run comprehensive tests on the component.

**Returns:**
- Dictionary with test results:
  - `test_ipfs_kit_init`: IPFS initialization status
  - `test_ipfs_kit`: IPFS functionality status  
  - `test_sparse_embeddings`: Sparse processing status

## Implementation Examples

### Legal Document Processing

```python
# Process legal documents with domain-specific settings
legal_metadata = {
    "dataset": "legal-corpus",
    "column": "case_text",
    "split": "train",
    "models": ["legal-bert-base"],
    "chunk_settings": {
        "chunk_size": 1024,  # Longer chunks for legal text
        "n_sentences": 12,
        "step_size": 512,
        "method": "adaptive",  # Respect sentence boundaries
        "embed_model": "legal-bert-base"
    },
    "dst_path": "/data/legal_sparse"
}

legal_sparse = sparse_embeddings(resources, legal_metadata)
legal_sparse.index_dataset("legal-corpus")
```

### Multi-Scale Processing

```python
# Create sparse embeddings at multiple scales
scales = [
    {"chunk_size": 128, "step_size": 64},   # Fine-grained
    {"chunk_size": 512, "step_size": 256},  # Medium
    {"chunk_size": 2048, "step_size": 1024} # Coarse
]

for i, scale in enumerate(scales):
    metadata_scale = metadata.copy()
    metadata_scale["chunk_settings"].update(scale)
    metadata_scale["dst_path"] = f"/data/sparse_scale_{i}"
    
    sparse_emb = sparse_embeddings(resources, metadata_scale)
    sparse_emb.index_dataset(dataset_name)
```

### Hybrid Dense-Sparse Pipeline

```python
# Create both dense and sparse embeddings
from create_embeddings import create_embeddings

# Dense embeddings
dense_metadata = {
    "dataset": "multi-modal-corpus",
    "models": ["sentence-transformers/all-MiniLM-L6-v2"],
    # ... other config
}

# Sparse embeddings  
sparse_metadata = {
    "dataset": "multi-modal-corpus", 
    "models": ["tf-idf", "bm25"],
    # ... other config
}

# Process both
dense_emb = create_embeddings(resources, dense_metadata)
sparse_emb = sparse_embeddings(resources, sparse_metadata)

await dense_emb.create_embeddings(**dense_metadata)
sparse_emb.index_dataset(sparse_metadata["dataset"])
```

## Performance Considerations

### Memory Management
- **Sparse Matrices**: Use scipy.sparse for memory efficiency
- **Chunk Processing**: Process data in manageable chunks
- **Vocabulary Size**: Limit vocabulary to most important terms

### Storage Optimization
- **Compression**: Use compressed sparse formats (CSR, CSC)
- **Indexing**: Create efficient lookup structures
- **Caching**: Cache frequently accessed sparse vectors

### Computational Efficiency
- **Vectorization**: Use vectorized operations for TF-IDF
- **Parallel Processing**: Distribute across multiple cores
- **Early Stopping**: Skip processing for empty/short texts

## Integration with Search

Sparse embeddings integrate seamlessly with the search component:

```python
# Search using sparse embeddings
search_results = search_component.search(
    query="legal precedent copyright",
    method="sparse",
    top_k=50
)

# Hybrid search combining dense and sparse
hybrid_results = search_component.hybrid_search(
    query="legal precedent copyright",
    sparse_weight=0.3,
    dense_weight=0.7,
    top_k=20
)
```

## Error Handling and Debugging

### Common Issues

1. **Vocabulary Size**: Large vocabularies can cause memory issues
   - Solution: Use vocabulary filtering and feature selection

2. **Chunk Boundary Issues**: Poor chunking can hurt performance
   - Solution: Use sentence-aware chunking methods

3. **Endpoint Connectivity**: Network issues with embedding endpoints
   - Solution: Implement retry logic and fallback endpoints

### Debug Information

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
results = sparse_emb.test()
for component, status in results.items():
    if isinstance(status, Exception):
        print(f"Error in {component}: {status}")
    else:
        print(f"{component}: OK")
```

## Advanced Features

### Custom Sparse Methods
Extend the component with custom sparse embedding methods:

```python
class CustomSparseEmbeddings(sparse_embeddings):
    def __init__(self, resources, metadata):
        super().__init__(resources, metadata)
        self.custom_vectorizer = self._init_custom_vectorizer()
    
    def custom_sparse_method(self, texts):
        # Implement custom sparse embedding logic
        return self.custom_vectorizer.transform(texts)
```

### Feature Engineering
- **N-gram Features**: Include bigrams and trigrams
- **Named Entity Features**: Include NER-based features
- **Domain Features**: Add domain-specific feature extractors

## Dependencies

- `transformers`: For tokenization and model interfaces
- `datasets`: For dataset loading and processing
- `scipy`: For sparse matrix operations
- `sklearn`: For TF-IDF and feature extraction
- `ipfs_embeddings_py`: Core embedding infrastructure

## Related Components

- [Create Embeddings](create-embeddings.md): For dense embeddings
- [Search Embeddings](search-embeddings.md): For search functionality
- [Shard Embeddings](shard-embeddings.md): For distributed processing
