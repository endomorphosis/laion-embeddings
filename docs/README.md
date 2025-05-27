# LAION Embeddings - IPFS-Based Embeddings Search Engine

Welcome to the comprehensive documentation for LAION Embeddings, an advanced IPFS-based embeddings search engine that provides FastAPI endpoints for creating, searching, and managing embeddings using multiple ML models and storage backends.

## Overview

LAION Embeddings is a distributed, scalable embeddings search engine built on IPFS (InterPlanetary File System) technology. It supports multiple embedding models, various endpoint types, and provides a comprehensive API for embedding generation, storage, and semantic search operations.

## Key Features

- **Multi-Model Support**: Compatible with various embedding models including gte-small, gte-large-en-v1.5, and gte-Qwen2-1.5B-instruct
- **Multiple Endpoint Types**: Supports TEI, OpenVINO, libp2p, local, and CUDA endpoints
- **IPFS Integration**: Distributed storage and retrieval using IPFS
- **Clustering Support**: Advanced clustering capabilities with IPFS clusters and Storacha
- **Sparse Embeddings**: Support for sparse embeddings and sharding
- **FastAPI Interface**: RESTful API for all operations
- **Real-time Search**: High-performance semantic search capabilities

## Documentation Structure

### Getting Started
- [Installation Guide](installation.md) - How to set up and install the system
- [Quick Start](quickstart.md) - Get up and running in minutes
- [Configuration](configuration.md) - Configure endpoints and models

### API Documentation
- [API Reference](api/README.md) - Complete API documentation

### Core Components
- [Components Overview](components/README.md) - Overview of all components
- [Search Embeddings](components/search-embeddings.md) - Semantic search functionality
- [Create Embeddings](components/create-embeddings.md) - Embedding generation
- [Sparse Embeddings](components/sparse-embeddings.md) - Sparse embedding support
- [Shard Embeddings](components/shard-embeddings.md) - Embedding sharding
- [IPFS Cluster Index](components/ipfs-cluster-index.md) - IPFS clustering
- [Storacha Clusters](components/storacha-clusters.md) - Storacha integration

### Embedding Models
- [Supported Models](models/README.md) - Overview of supported models
- [Custom Models](models/custom-models.md) - Adding custom models

### Endpoint Management
- [Endpoint Types](endpoints/README.md) - Overview of endpoint types and configuration

### IPFS Integration
- [IPFS Overview](ipfs/README.md) - IPFS integration overview and workflows

### Utility Scripts
- [Script Reference](scripts/README.md) - Complete utility scripts documentation
- [Server Management](scripts/server.md) - Server control scripts
- [Data Processing](scripts/data.md) - Data processing utilities
- [IPFS Operations](scripts/ipfs.md) - IPFS-related scripts

### Development
- [Developer Guide](development.md) - Development guidelines and setup
- [Testing](development.md#testing) - Testing procedures
- [Deployment](development.md#deployment) - Deployment strategies

### Examples
- [Simple Search](examples/simple-search.md) - Basic search examples
- [Batch Processing](examples/batch-processing.md) - Batch processing examples
- [IPFS Integration](examples/ipfs-integration.md) - IPFS integration examples
- [Python Client](examples/python-client.md) - Python client examples

### Troubleshooting
- [Troubleshooting Guide](troubleshooting/README.md) - Comprehensive troubleshooting guide
- [Performance Issues](troubleshooting/README.md#performance-issues) - Performance optimization tips
- [Common Errors](troubleshooting/README.md#common-issues) - Frequently encountered problems

### Evaluation and Benchmarking
- [Evaluation Framework](evaluation/README.md) - Complete evaluation and benchmarking guide
- [BEIR Benchmarks](evaluation/README.md#beir-evaluation) - BEIR benchmark evaluation
- [Performance Analysis](evaluation/README.md#performance-analysis) - Performance analysis tools

## Architecture

LAION Embeddings follows a modular architecture with the following main components:

1. **FastAPI Server** (`main.py`) - Main API interface
2. **Core Library** (`ipfs_embeddings_py/`) - Core functionality
3. **Embedding Modules** - Specialized modules for different operations
4. **IPFS Integration** - Distributed storage layer
5. **Endpoint Management** - Multi-endpoint support system

## Quick Links

- [Installation Guide](installation.md) - Set up and install the system
- [Quick Start](quickstart.md) - Get running in minutes
- [API Reference](api/README.md) - Complete API documentation
- [Utility Scripts](scripts/README.md) - Convenient scripts for common operations
- [Examples](examples/README.md) - Complete examples and tutorials
- [Troubleshooting](troubleshooting/README.md) - Solutions to common issues
- [Configuration](configuration.md) - Configure endpoints and models

## Support

For support, please:
1. Check the [troubleshooting guide](troubleshooting/README.md)
2. Review the [FAQ](faq.md)
3. Open an issue on GitHub
4. Join our community discussions

---

*Last updated: May 27, 2025*
