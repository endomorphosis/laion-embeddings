# Unified Vector Store Architecture

The LAION Embeddings project uses a unified vector database architecture that provides a consistent interface to multiple vector storage systems. This architecture allows for seamless switching between different vector store implementations while maintaining the same API.

## Architecture Overview

The architecture consists of several key components:

- **BaseVectorStore**: Abstract base class defining the common interface for all vector stores
- **VectorStoreFactory**: Factory for creating and managing vector store instances
- **VectorStoreRegistry**: Registry for vector store implementations
- **VectorDatabaseConfigManager**: Manager for vector database configurations

## Supported Vector Store Providers

The following vector store providers are supported:

1. **Qdrant** - In-memory and persistent vector database
2. **Elasticsearch** - Scalable search engine with vector capabilities
3. **pgvector** - PostgreSQL extension for vector similarity search
4. **FAISS** - Facebook AI Similarity Search library
5. **IPFS/IPLD** - InterPlanetary File System with content-addressable storage
6. **DuckDB/Parquet** - Analytical database with columnar Parquet storage

## Key Features

The unified architecture provides the following key features:

- **Common Interface**: All providers implement the same interface, making it easy to switch between them
- **Configuration-Driven**: Vector databases can be selected and configured through configuration files
- **Graceful Degradation**: Applications work even when some providers are unavailable
- **Extensible**: New providers can be added without changing client code
- **Advanced Features**: Support for vector quantization, sharding, and metadata filtering

## Core Components

### BaseVectorStore

The `BaseVectorStore` abstract base class defines the common interface for all vector store implementations:

```python
class BaseVectorStore(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the vector store."""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the vector store."""
        pass
        
    @abstractmethod
    async def create_index(self, dimension: int, **kwargs) -> bool:
        """Create a new vector index."""
        pass
        
    @abstractmethod
    async def add_documents(self, documents: List[VectorDocument]) -> int:
        """Add vectors to the index."""
        pass
        
    @abstractmethod
    async def search(self, query: SearchQuery) -> SearchResult:
        """Search the vector index."""
        pass
        
    # And other methods...
```

### VectorStoreFactory

The `VectorStoreFactory` provides a factory pattern for creating and managing vector store instances:

```python
class VectorStoreFactory:
    def __init__(self, config_manager: Optional[VectorDatabaseConfigManager] = None):
        self.config_manager = config_manager or get_config_manager()
        self.registry = _registry
        self._auto_register_stores()
        
    async def create_store(self, db_type: Optional[VectorDBType] = None,
                         config_override: Optional[Dict[str, Any]] = None,
                         reuse_instance: bool = True) -> BaseVectorStore:
        # Creates a vector store instance
        # ...
```

## Usage Examples

### Basic Example

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def main():
    # Create a store instance (will use default from config if None)
    store = await create_vector_store(VectorDBType.FAISS)
    
    # Connect to the store
    await store.connect()
    
    # Add documents
    docs = [
        VectorDocument(
            id="doc1",
            vector=[0.1, 0.2, 0.3, 0.4],
            metadata={"text": "Example document"}
        )
    ]
    
    await store.add_documents(docs)
    
    # Search
    query = SearchQuery(
        vector=[0.1, 0.2, 0.3, 0.4],
        top_k=5
    )
    
    results = await store.search(query)
    print(results)
    
    # Disconnect
    await store.disconnect()

asyncio.run(main())
```

## Configuration

Vector stores are configured in `config/vector_databases.yaml`. Each provider has its own section with connection parameters, index parameters, and search parameters.

See the [configuration documentation](configuration.md) for detailed information on configuring vector stores.

## Provider-Specific Documentation

- [IPFS Vector Service](ipfs-vector-service.md) - Distributed vector storage with IPFS
- [DuckDB Vector Service](duckdb-vector-service.md) - Analytical vector database with Parquet

## Testing

The unified architecture includes comprehensive tests for all vector store providers. See the [testing documentation](testing.md) for information on running these tests.
