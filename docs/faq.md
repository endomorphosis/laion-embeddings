# Frequently Asked Questions (FAQ)

## General Questions

### What is LAION Embeddings?
LAION Embeddings is a distributed, scalable embeddings search engine built on IPFS (InterPlanetary File System) technology. It provides FastAPI endpoints for creating, searching, and managing embeddings using multiple ML models and storage backends.

### What embedding models are supported?
LAION Embeddings supports several embedding models including:
- gte-small (general text embeddings)
- gte-large-en-v1.5 (large English model)
- gte-Qwen2-1.5B-instruct (instruction-tuned model)

For more details, see the [Models Documentation](models/README.md).

### What endpoint types are available?
The system supports multiple endpoint types:
- TEI (Text Embeddings Inference)
- OpenVINO (Intel's inference engine)
- Local CPU endpoints
- Local CUDA endpoints
- LibP2P (peer-to-peer)
- Intel IPEX endpoints
- Llama.cpp endpoints

For detailed configuration, see the [Endpoint Management Documentation](endpoints/README.md).

## Installation and Setup

### What are the system requirements?
- Python 3.8+
- At least 8GB RAM (16GB+ recommended for large models)
- Optional: CUDA-compatible GPU for acceleration
- Docker (for containerized deployment)

### How do I install LAION Embeddings?
See the [Installation Guide](installation.md) for detailed instructions including:
- Docker installation
- Python virtual environment setup
- Development environment setup

### Can I run this without a GPU?
Yes, the system supports CPU-only inference through various endpoint types. However, GPU acceleration is recommended for better performance with large datasets.

## Configuration

### How do I configure endpoints?
Endpoints can be configured through:
- Environment variables
- YAML configuration files
- Command-line arguments

See the [Configuration Guide](configuration.md) for detailed instructions.

### How do I add custom models?
Custom models can be added by:
1. Creating a model configuration
2. Implementing the model interface
3. Registering the model with the system

See the [Custom Models Guide](models/custom-models.md) for details.

### What storage backends are supported?
- IPFS (InterPlanetary File System)
- Local file system
- Storacha (Web3.Storage)
- FAISS for vector indexing

## Usage

### How do I perform a basic search?
```python
import requests

response = requests.post("http://localhost:8000/search", json={
    "query": "your search text",
    "top_k": 10
})
results = response.json()
```

For more examples, see the [Simple Search Guide](examples/simple-search.md).

### How do I create embeddings for my data?
You can create embeddings through:
- The API endpoints
- Batch processing scripts
- Python client library

See the [Create Embeddings Guide](components/create-embeddings.md) for details.

### What's the difference between dense and sparse embeddings?
- **Dense embeddings**: Fixed-size vectors that capture semantic meaning
- **Sparse embeddings**: High-dimensional vectors with mostly zero values, often used for keyword matching

See the [Sparse Embeddings Guide](components/sparse-embeddings.md) for more information.

## Performance

### How can I optimize performance?
- Use GPU acceleration when available
- Optimize batch sizes for your hardware
- Enable caching
- Use appropriate endpoint types for your use case

See the [Performance Optimization](troubleshooting/README.md#performance-issues) section.

### What batch sizes should I use?
Batch sizes depend on:
- Available memory (GPU/CPU)
- Model size
- Input sequence length

Start with smaller batches (8-32) and increase until you hit memory limits.

### How do I monitor system performance?
The system provides:
- Health check endpoints
- Performance metrics
- Resource utilization monitoring

See the [Troubleshooting Guide](troubleshooting/README.md) for monitoring setup.

## IPFS Integration

### Why use IPFS?
IPFS provides:
- Decentralized storage
- Content addressing (immutable hashes)
- Deduplication
- Peer-to-peer distribution

### How do I work with IPFS data?
The system provides utilities for:
- Converting data formats (Parquet ↔ CAR)
- Content addressing
- Cluster management

See the [IPFS Integration Guide](ipfs/README.md) for details.

### What are CIDs?
Content Identifiers (CIDs) are cryptographic hashes that uniquely identify content in IPFS. They ensure data integrity and enable content addressing.

## Troubleshooting

### The server won't start. What should I check?
1. Verify Python version (3.8+)
2. Check dependencies are installed
3. Ensure ports are available (default: 8000)
4. Check environment variables
5. Review error logs

See the [Troubleshooting Guide](troubleshooting/README.md) for more solutions.

### I'm getting GPU memory errors
- Reduce batch size
- Use CPU endpoints
- Enable gradient checkpointing
- Use model quantization

### Embeddings generation is slow
- Use GPU acceleration
- Optimize batch sizes
- Enable endpoint pooling
- Consider model quantization

### How do I debug API errors?
1. Check the API logs
2. Verify request format
3. Test with simple examples
4. Use the diagnostic scripts

## Development

### How do I contribute to the project?
1. Fork the repository
2. Create a feature branch
3. Follow code standards
4. Add tests
5. Submit a pull request

See the [Development Guide](development.md) for details.

### How do I run tests?
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test/test_embeddings.py

# Run with coverage
python -m pytest --cov=ipfs_embeddings_py
```

### How do I add a new endpoint type?
1. Implement the endpoint interface
2. Add configuration options
3. Update the endpoint manager
4. Add tests and documentation

See the [Endpoint Management Documentation](endpoints/README.md) for implementation details.

## Support

### Where can I get help?
- Check this FAQ
- Review the [Troubleshooting Guide](troubleshooting/README.md)
- Search existing GitHub issues
- Open a new issue with detailed information

### How do I report bugs?
When reporting bugs, include:
- System information (OS, Python version, GPU)
- Complete error messages
- Steps to reproduce
- Sample data (if applicable)

### Where can I find examples?
Complete examples are available in the [Examples Section](examples/README.md), including:
- Simple search operations
- Batch processing workflows
- IPFS integration
- Python client usage

---

*For more detailed information, please refer to the specific documentation sections linked throughout this FAQ.*
