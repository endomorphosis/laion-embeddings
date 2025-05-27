# Examples and Tutorials

This section provides practical examples and step-by-step tutorials for using the LAION Embeddings project effectively.

## Available Examples

### Basic Usage Examples
- [**Simple Search Example**](./simple-search.md) - Basic text similarity search
- [**Batch Processing Example**](./batch-processing.md) - Processing multiple texts efficiently
- [**IPFS Integration Example**](./ipfs-integration.md) - Working with IPFS-stored embeddings

### Advanced Use Cases
- [**Custom Model Integration**](./custom-models.md) - Adding and configuring custom models
- [**Distributed Processing**](./distributed-processing.md) - Setting up multi-node processing
- [**Production Deployment**](./production-deployment.md) - Complete production setup guide

### Integration Examples
- [**Python Client**](./python-client.md) - Building applications with the Python API
- [**Web Application**](./web-application.md) - Creating a web interface
- [**Jupyter Notebooks**](./jupyter-notebooks.md) - Interactive analysis and exploration

### Performance Optimization
- [**Endpoint Selection**](./endpoint-optimization.md) - Choosing the right endpoints for your hardware
- [**Memory Management**](./memory-optimization.md) - Optimizing memory usage for large datasets
- [**Caching Strategies**](./caching-strategies.md) - Implementing effective caching

## Getting Started

If you're new to the LAION Embeddings project, we recommend starting with:

1. [Simple Search Example](./simple-search.md) - Learn basic operations
2. [Python Client](./python-client.md) - Build your first application
3. [Batch Processing Example](./batch-processing.md) - Scale up your processing

## Example Dataset

All examples use a common sample dataset available at:
- **IPFS Hash**: `QmYourSampleDatasetHash`
- **Format**: Parquet files with text embeddings
- **Size**: ~1000 text samples with pre-computed embeddings
- **Models**: gte-small, gte-large-en-v1.5

You can download this dataset using:

```bash
# Using IPFS
ipfs get QmYourSampleDatasetHash

# Using HTTP gateway
curl -O https://ipfs.io/ipfs/QmYourSampleDatasetHash
```

## Code Repository

All example code is available in the project's `examples/` directory and can be run directly after following the [Installation Guide](../installation.md).

## Contributing Examples

We welcome contributions of new examples and tutorials! Please see our [Development Guide](../development.md) for contribution guidelines.

## Support

If you encounter issues with any examples:
1. Check the [Troubleshooting Guide](../troubleshooting/README.md)
2. Review the relevant [Component Documentation](../components/README.md)
3. Open an issue on our GitHub repository
