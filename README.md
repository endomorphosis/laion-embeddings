# LAION Embeddings - IPFS-Based Embeddings Search Engine

An advanced IPFS-based embeddings search engine that provides FastAPI endpoints for creating, searching, and managing embeddings using multiple ML models and storage backends.

## 📚 Documentation

**[Complete Documentation →](docs/README.md)**

- [Installation Guide](docs/installation.md) - Set up and install the system
- [Quick Start](docs/quickstart.md) - Get running in minutes  
- [API Reference](docs/api/README.md) - Complete API documentation
- [Configuration](docs/configuration.md) - Configure endpoints and models
- [Examples](docs/examples/README.md) - Complete examples and tutorials
- [FAQ](docs/faq.md) - Frequently asked questions

## ⚡ Quick Start

### 1. Start the Server
```bash
./run.sh
```

This runs the FastAPI server:
```bash
python3 -m fastapi run main.py
```

### 2. Load Data (Optional)
```bash
./load.sh
```

This loads embeddings into the system using curl:
```bash
curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/Wikipedia-X-Concat", "knn_index":"laion/Wikipedia-M3", "dataset_split": "enwiki_concat", "knn_index_split": "enwiki_embed", "column": "Concat Abstract"}' \
    -H 'Content-Type: application/json'
```

> **Note**: This will take hours to download/ingest for large datasets. FastAPI is unavailable while this runs.

### 3. Search
```bash
./search.sh
```

Search the index with text:
```bash
curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"orange juice", "collection": "Wikipedia-X-Concat"}' \
    -H 'Content-Type: application/json'
```

### 4. Create Embeddings
```bash
./create.sh
```

Create embeddings from a dataset (outputs stored in "checkpoints" directory):
```bash
curl 127.0.0.1:9999/create \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

## 🚀 Key Features

- **Multi-Model Support**: gte-small, gte-large-en-v1.5, gte-Qwen2-1.5B-instruct
- **Multiple Endpoints**: TEI, OpenVINO, LibP2P, Local, CUDA endpoints  
- **IPFS Integration**: Distributed storage and retrieval
- **Clustering Support**: IPFS clusters and Storacha integration
- **Sparse Embeddings**: TF-IDF and BM25 scoring support
- **FastAPI Interface**: RESTful API for all operations
- **Real-time Search**: High-performance semantic search

## 📁 Project Structure

- `main.py` - FastAPI application with endpoints
- `ipfs_embeddings_py/` - Core functionality library
- `create_embeddings/` - Embedding generation module
- `search_embeddings/` - Search functionality
- `sparse_embeddings/` - Sparse embedding support  
- `shard_embeddings/` - Distributed sharding
- `ipfs_cluster_index/` - IPFS cluster management
- `storacha_clusters/` - Storacha integration
- `docs/` - Comprehensive documentation

## 🛠 Utility Scripts

- `run.sh` - Start the FastAPI server
- `load.sh`, `load2.sh`, `load3.sh` - Load data into the system
- `search.sh`, `search2.sh` - Search operations
- `create.sh` - Create embeddings from datasets
- `create_sparse.sh` - Create sparse embeddings
- `shard_cluster.sh` - Shard embeddings using clustering
- `index_cluster.sh` - IPFS cluster indexing
- `storacha.sh` - Storacha storage operations
- `autofaiss.sh` - FAISS integration
- `launch_tei.sh` - Launch TEI endpoints
- `install_depends.sh` - Install dependencies

## 📖 Documentation Overview

For complete documentation, examples, and guides, visit the **[Documentation Directory](docs/README.md)**.

### Core Documentation
- **[Installation Guide](docs/installation.md)** - Complete setup instructions
- **[Quick Start](docs/quickstart.md)** - Get running in minutes
- **[API Reference](docs/api/README.md)** - Full API documentation
- **[Configuration](docs/configuration.md)** - Endpoint and model configuration
- **[Components Overview](docs/components/README.md)** - System architecture

### Advanced Features  
- **[IPFS Integration](docs/ipfs/README.md)** - Distributed storage workflows
- **[Custom Models](docs/models/custom-models.md)** - Adding and configuring models
- **[Evaluation Framework](docs/evaluation/README.md)** - Benchmarking and testing
- **[Troubleshooting](docs/troubleshooting/README.md)** - Common issues and solutions

## 🔧 Advanced Operations

### Index IPFS Cluster
```bash
./index_cluster.sh
```

Index the local IPFS cluster node and output CID embeddings:
```bash
curl 127.0.0.1:9999/index_cluster \
    -X POST \
    -d '["localhost", "cloudkit_storage", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

### Create Sparse Embeddings
```bash
./create_sparse.sh
```

Generate sparse embeddings (outputs to "sparse_checkpoints" directory):
```bash
curl 127.0.0.1:9999/create_sparse \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

### Shard with K-means Clustering
```bash
./shard_cluster.sh
```

Shard clusters into manageable sizes (≤50MB or 4096 rows):
```bash
curl 127.0.0.1:9999/shard_cluster \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

### Upload to Storacha
```bash
./storacha.sh
```

Convert Parquet files to CAR format and upload to Storacha network:
```bash
curl 127.0.0.1:9999/storacha \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/development.md) for:
- Development environment setup
- Code standards and best practices  
- Testing procedures
- Pull request guidelines

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 🆘 Support

- **[FAQ](docs/faq.md)** - Frequently asked questions
- **[Troubleshooting](docs/troubleshooting/README.md)** - Common issues and solutions
- **GitHub Issues** - Report bugs or request features
- **[Examples](docs/examples/README.md)** - Complete usage examples

---

For detailed documentation, please visit the **[Documentation Directory](docs/README.md)**.
