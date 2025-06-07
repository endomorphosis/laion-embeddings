# Quick Start Guide

This guide will help you get started with the LAION Embeddings search engine quickly.

## 🎉 Latest Update (v2.2.0)

**Great News!** The system is now production-ready with:
- ✅ All 22 MCP tools fully functional (100% success rate)
- ✅ Docker-CI/CD alignment for consistent deployment
- ✅ Unified MCP server entrypoint (`mcp_server.py`)
- ✅ Comprehensive testing and validation
- ✅ Professional project organization
- ✅ Complete documentation

## Before You Begin

Ensure you have completed the [Installation Guide](installation.md) to install all required dependencies.

## Quick Validation

First, verify that all MCP tools are working (same validation as CI/CD and Docker):

```bash
python3 mcp_server.py --validate
```

Expected output: JSON with status "success" and tools count "22"

## Starting the MCP Server (Optional)

For AI assistant integration, start the MCP server:

```bash
python3 mcp_server.py
```

This starts the Model Context Protocol server with all 22 tools available for AI assistants.

> **Note**: The MCP server uses the same entrypoint across all environments (development, CI/CD, Docker) for consistency.

## Starting the Server

Start the FastAPI server:

```bash
./run.sh
```

This runs the FastAPI server on port 9999:

```bash
python3 -m fastapi run main.py
```

## Using the API

### 1. Create Embeddings

Create embeddings from a dataset:

```bash
./create.sh
```

This sends a request to create embeddings:

```bash
curl 127.0.0.1:9999/create \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

### 2. Load Embeddings

Load embeddings into the search engine:

```bash
./load.sh
```

This sends a request to load embeddings:

```bash
curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/Wikipedia-X-Concat", "knn_index":"laion/Wikipedia-M3", "dataset_split": "enwiki_concat", "knn_index_split": "enwiki_embed", "column": "Concat Abstract"}' \
    -H 'Content-Type: application/json'
```

### 3. Search Embeddings

Search for similar embeddings:

```bash
./search.sh
```

This sends a search request:

```bash
curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"orange juice", "collection": "Wikipedia-X-Concat"}' \
    -H 'Content-Type: application/json'
```

## Using the Vector Store API Directly

You can use the vector store API directly in your Python code:

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def main():
    # Create a vector store (FAISS is the default if none specified)
    store = await create_vector_store()
    
    # Connect to the store
    await store.connect()
    
    # Create a new index
    await store.create_index(dimension=384)
    
    # Add documents
    docs = [
        VectorDocument(
            id="doc1",
            vector=[0.1, 0.2, 0.3, ...],  # 384 dimensions
            metadata={"text": "Example document"}
        )
    ]
    
    await store.add_documents(docs)
    
    # Search
    query = SearchQuery(
        vector=[0.1, 0.2, 0.3, ...],  # 384 dimensions
        top_k=5
    )
    
    results = await store.search(query)
    print(results)
    
    # Disconnect
    await store.disconnect()

asyncio.run(main())
```

## Using Different Vector Stores

### Using IPFS Vector Store

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def use_ipfs():
    # Create IPFS store
    store = await create_vector_store(VectorDBType.IPFS)
    
    # Connect and use as with any other store
    await store.connect()
    
    # Your operations here
    
    await store.disconnect()

asyncio.run(use_ipfs())
```

### Using DuckDB Vector Store

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def use_duckdb():
    # Create DuckDB store
    store = await create_vector_store(VectorDBType.DUCKDB)
    
    # Connect and use as with any other store
    await store.connect()
    
    # Your operations here
    
    await store.disconnect()

asyncio.run(use_duckdb())
```

## Testing Your Setup

Run the basic tests:

```bash
python -m pytest test/test_basic.py
```

Run the vector store tests:

```bash
./test_vector_stores.py
```

## Next Steps

- [API Reference](api/README.md) - Complete API documentation
- [Configuration Guide](configuration.md) - Configure endpoints and models
- [IPFS Vector Service](ipfs-vector-service.md) - Learn about IPFS integration
- [DuckDB Vector Service](duckdb-vector-service.md) - Learn about DuckDB integration
- [Examples](examples/README.md) - More examples and tutorials
