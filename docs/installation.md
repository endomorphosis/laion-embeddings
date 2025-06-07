# Installation Guide

This guide will help you install and set up the LAION Embeddings search engine with all its dependencies.

## 🎉 Latest Update (v2.2.0)

**Production Ready!** All 22 MCP tools are now fully functional with comprehensive Docker support and CI/CD alignment.

### Key Features ✅
- **Docker-CI/CD Alignment**: Complete configuration consistency across all environments
- **Unified MCP Server**: Single `mcp_server.py` entrypoint for all deployment scenarios
- **Production Validation**: Same validation commands used in CI/CD, Docker, and manual testing

## Installation Options

Choose your preferred installation method:

1. **🐳 Docker (Recommended)** - Fastest setup, production-ready
2. **📦 Native Python** - Full development environment
3. **🚀 Docker Compose** - Full stack with monitoring

## 🐳 Docker Installation (Recommended)

The fastest way to get started with a production-ready setup.

### Prerequisites for Docker

- Docker 20.10+ and Docker Compose 2.0+
- 8GB+ RAM recommended
- 20GB+ disk space

### Quick Docker Start

```bash
# Clone repository
git clone https://github.com/laion-ai/laion-embeddings.git
cd laion-embeddings

# Validate MCP tools (same as CI/CD)
python3 mcp_server.py --validate

# Full deployment (build, test, and run)
./docker-deploy.sh all
```

Your server will be available at `http://localhost:9999`

> **Note**: The Docker deployment uses the same `mcp_server.py` entrypoint as the CI/CD pipeline, ensuring consistent behavior across all environments.

### Docker Commands

```bash
# Build image only
./docker-deploy.sh build

# Test the image
./docker-deploy.sh test

# Run production container
./docker-deploy.sh run

# Check status
./docker-deploy.sh status

# View logs
./docker-deploy.sh logs

# Stop container
./docker-deploy.sh stop

# Full cleanup
./docker-deploy.sh cleanup
```

### Docker Compose (Full Stack)

For a complete setup with monitoring:

```bash
# Start all services (API + IPFS + Monitoring)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f laion-embeddings

# Stop all services
docker-compose down
```

Services will be available at:
- **API Server**: http://localhost:9999
- **Health Check**: http://localhost:9999/health
- **IPFS Gateway**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 📦 Native Python Installation

## Prerequisites

The following prerequisites are required:

- Python 3.9 or higher
- pip (Python package manager)
- IPFS daemon (optional, for IPFS vector store)
- 8GB+ RAM recommended
- 20GB+ disk space for embeddings storage

## Basic Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/laion-ai/laion-embeddings.git
cd laion-embeddings
```

### Step 2: Create and Activate a Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install basic dependencies
./install_depends.sh

# Alternatively, use pip directly
pip install -r requirements.txt
```

## Vector Store Dependencies

Different vector stores require different dependencies:

### FAISS (Built-in)

FAISS is included in the basic installation.

### Qdrant

```bash
pip install qdrant-client>=1.1.6
```

### Elasticsearch

```bash
pip install elasticsearch>=8.0.0
```

### pgvector

```bash
pip install psycopg2-binary>=2.9.5 sqlalchemy>=2.0.0
```

### IPFS/IPLD

```bash
# Option 1: Using ipfs_kit_py (recommended)
pip install ipfs_kit_py>=1.0.0

# Option 2: Using ipfshttpclient
pip install ipfshttpclient>=0.7.0
```

### DuckDB/Parquet

```bash
pip install duckdb>=0.9.0 pyarrow>=14.0.1
```

## IPFS Setup

If you plan to use the IPFS vector store, you'll need a running IPFS daemon:

### Install IPFS

Follow the instructions at [https://docs.ipfs.tech/install/](https://docs.ipfs.tech/install/)

### Start IPFS Daemon

```bash
ipfs daemon
```

## Advanced Installation

### Docker Installation

You can use Docker to run the project:

```bash
# Build the Docker image
docker build -t laion-embeddings .

# Run the container
docker run -p 9999:9999 -v $(pwd)/data:/app/data laion-embeddings
```

### GPU Support

For GPU support, install the appropriate PyTorch version:

```bash
# For CUDA 11.8
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# For CUDA 12.1
pip install torch==2.0.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
```

## Configuration

### Vector Database Configuration

The vector database configuration is stored in `config/vector_databases.yaml`. Edit this file to configure your vector stores:

```yaml
databases:
  qdrant:
    enabled: true
    host: localhost
    port: 6333
    # ... other Qdrant settings ...
    
  elasticsearch:
    enabled: false
    host: localhost
    port: 9200
    # ... other Elasticsearch settings ...
    
  pgvector:
    enabled: false
    connection_string: "postgresql://user:password@localhost:5432/vectors"
    # ... other pgvector settings ...
    
  faiss:
    enabled: true
    storage_path: "data/faiss_indexes"
    # ... other FAISS settings ...
    
  ipfs:
    enabled: false
    ipfs_gateway: "localhost:5001"
    # ... other IPFS settings ...
    
  duckdb:
    enabled: false
    database_path: "data/vectors.duckdb"
    storage_path: "data/vector_parquet"
    # ... other DuckDB settings ...
```

## Verification

To verify your installation:

```bash
# Run a simple check
python -m pytest test/test_imports.py

# Run the comprehensive tests
python run_comprehensive_tests.py
```

## Troubleshooting

### Common Issues

#### IPFS Connection Issues

If you encounter IPFS connection issues:

1. Ensure IPFS daemon is running: `ipfs daemon`
2. Check the IPFS gateway setting in `config/vector_databases.yaml`
3. Try disabling sharding for testing: set `sharding_enabled: false`

#### DuckDB Errors

If you see DuckDB errors:

1. Ensure you have the latest versions: `pip install --upgrade duckdb pyarrow`
2. Check file permissions on the database path
3. Try setting `"memory_limit": "4GB"` in the configuration

#### Out of Memory Errors

If you encounter memory issues:

1. Reduce batch sizes in configuration
2. Use sharding for large vector collections
3. Consider using a disk-based store like DuckDB instead of in-memory stores

## Next Steps

After installation, check out:

- [Quick Start Guide](quickstart.md) to get started quickly
- [Configuration Guide](configuration.md) for detailed configuration options
- [API Reference](api/README.md) for the API documentation
