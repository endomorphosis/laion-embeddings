# Contributing Guide

Thank you for considering contributing to the LAION Embeddings project! This guide will help you understand how to contribute to the project, including adding or enhancing vector store providers.

## Getting Started

### Setting Up Your Development Environment

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/laion-embeddings.git
   cd laion-embeddings
   ```

2. **Install Development Dependencies**:
   ```bash
   ./install_depends.sh
   pip install -r requirements-dev.txt
   ```

3. **Set Up Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Branch Naming Convention

Use the following naming convention for branches:
- `feature/feature-name` for new features
- `fix/bug-name` for bug fixes
- `docs/update-name` for documentation updates
- `refactor/component-name` for code refactoring

### Commit Message Style

Follow the conventional commits style:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Example:
```
feat(ipfs): add distributed search capability

- Implement sharded search across multiple IPFS nodes
- Add node health checking
- Add fault tolerance

Fixes #123
```

### Pull Request Process

1. Create a new branch for your changes
2. Make your changes with appropriate tests
3. Update documentation to reflect your changes
4. Create a pull request against the `main` branch
5. Ensure all CI checks pass

## Code Structure

The project is structured as follows:

```
laion-embeddings-1/
├── config/                 # Configuration files
├── services/               # Core services
│   ├── providers/          # Vector store providers
│   │   ├── __init__.py     # Provider registration
│   │   ├── faiss_store.py  # FAISS provider
│   │   ├── ipfs_store.py   # IPFS provider
│   │   ├── duckdb_store.py # DuckDB provider
│   │   └── hnsw_store.py   # HNSW provider
│   ├── embedding.py        # Embedding model interface
│   ├── vector_config.py    # Configuration classes
│   ├── vector_store_base.py # Vector store base classes
│   └── vector_store_factory.py # Factory for creating providers
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Test configuration
└── docs/                   # Documentation
```

## Adding a New Vector Store Provider

To add a new vector store provider:

1. **Create Provider Implementation**:
   Create a new file in `services/providers/` (e.g., `new_store.py`) that implements the `VectorStoreBase` abstract base class.

2. **Register Provider**:
   Add your provider to `services/providers/__init__.py` to register it with the factory.

3. **Update Configuration**:
   Update `services/vector_config.py` to add configuration options for your provider.

4. **Update Factory**:
   Update `services/vector_store_factory.py` to handle creating your provider.

5. **Add Tests**:
   Add tests for your provider in `tests/unit/test_new_store.py` and integration tests in `tests/integration/test_new_store_integration.py`.

6. **Update Documentation**:
   Add documentation for your provider in `docs/new-store-service.md` and examples in `docs/examples/new-store-examples.md`.

### Example: Provider Implementation

Here's a simplified template for a new vector store provider:

```python
from typing import List, Optional, Dict, Any, Union
import asyncio

from services.vector_store_base import (
    VectorStoreBase,
    VectorDocument,
    SearchQuery,
    SearchResult,
    Match
)
from services.vector_config import NewStoreConfig

class NewVectorStore(VectorStoreBase):
    """
    Implementation of the VectorStoreBase interface for NewStore.
    """
    
    def __init__(self, config: Optional[NewStoreConfig] = None, **kwargs):
        """Initialize the NewStore vector store."""
        super().__init__()
        self.config = config or NewStoreConfig()
        # Initialize any provider-specific attributes
        
    async def connect(self) -> bool:
        """Connect to the vector store."""
        # Implement connection logic
        return True
        
    async def disconnect(self) -> bool:
        """Disconnect from the vector store."""
        # Implement disconnection logic
        return True
        
    async def create_index(self, dimension: int, **kwargs) -> str:
        """Create a new index."""
        # Implement index creation logic
        return "new-index-id"
        
    async def delete_index(self, index_id: Optional[str] = None) -> bool:
        """Delete an index."""
        # Implement index deletion logic
        return True
        
    async def add_documents(self, documents: List[VectorDocument]) -> int:
        """Add documents to the vector store."""
        # Implement document addition logic
        return len(documents)
        
    async def search(self, query: SearchQuery) -> SearchResult:
        """Search for similar vectors."""
        # Implement search logic
        return SearchResult(matches=[])
        
    # Implement other required methods...
```

### Registering Your Provider

Update `services/providers/__init__.py`:

```python
from .faiss_store import FAISSVectorStore
from .ipfs_store import IPFSVectorStore
from .duckdb_store import DuckDBVectorStore
from .new_store import NewVectorStore  # Add your import

# Register providers
PROVIDERS = {
    "faiss": FAISSVectorStore,
    "ipfs": IPFSVectorStore,
    "duckdb": DuckDBVectorStore,
    "new_store": NewVectorStore  # Register your provider
}
```

### Updating the Enum

Update `services/vector_config.py`:

```python
class VectorDBType(str, Enum):
    """Enum representing supported vector database types."""
    FAISS = "faiss"
    IPFS = "ipfs"
    DUCKDB = "duckdb"
    HNSW = "hnsw"
    NEW_STORE = "new_store"  # Add your provider
```

### Updating the Factory

Update `services/vector_store_factory.py`:

```python
async def create_vector_store(db_type: Optional[VectorDBType] = None, **kwargs) -> VectorStoreBase:
    """Create a vector store instance."""
    if db_type is None:
        db_type = _get_default_vector_db_type()
        
    if db_type == VectorDBType.FAISS:
        return await _try_register_faiss(**kwargs)
    elif db_type == VectorDBType.IPFS:
        return await _try_register_ipfs(**kwargs)
    elif db_type == VectorDBType.DUCKDB:
        return await _try_register_duckdb(**kwargs)
    elif db_type == VectorDBType.NEW_STORE:
        return await _try_register_new_store(**kwargs)  # Add your provider
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")

async def _try_register_new_store(**kwargs) -> VectorStoreBase:
    """Try to create and register a NewStore instance."""
    try:
        from services.providers.new_store import NewVectorStore
        config = _get_new_store_config(**kwargs)
        return NewVectorStore(config=config, **kwargs)
    except ImportError as e:
        raise ImportError(f"Could not import NewStore: {str(e)}. "
                          f"Please install required dependencies.") from e
```

## Testing Your Contributions

### Running Tests

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run specific test
python -m pytest tests/unit/test_new_store.py

# Run with coverage
python -m pytest --cov=services
```

### Test Requirements

For new features or providers:
1. **Unit Tests**: Test individual components and functions
2. **Integration Tests**: Test the provider with a real backend
3. **Test Coverage**: Aim for at least 80% test coverage for new code

## Documentation Standards

When adding or updating documentation:

1. **Follow Markdown Best Practices**:
   - Use headers to organize content
   - Include code examples with syntax highlighting
   - Use tables for structured information

2. **Documentation Structure**:
   - Overview/Introduction
   - Installation/Configuration
   - Basic Usage
   - Advanced Features
   - API Reference
   - Examples

3. **Example Structure**:
   - Simple use case
   - Common operations
   - Advanced scenarios

## Code Style

Follow these style guidelines:

1. **Python Style**:
   - Follow PEP 8
   - Use type hints
   - Document classes and functions with docstrings

2. **Imports**:
   - Group imports: standard library, third-party, local
   - Sort alphabetically within groups

3. **Naming**:
   - Use `snake_case` for variables and functions
   - Use `PascalCase` for classes
   - Use `UPPER_CASE` for constants

## Submitting Changes

1. **Create a Pull Request**:
   - Describe what the changes do
   - Reference any related issues
   - Include before/after examples if applicable

2. **Code Review Process**:
   - Address review comments
   - Keep the PR focused on a single issue
   - Be responsive to questions

3. **After Merge**:
   - Update documentation if necessary
   - Help with any follow-up issues

## Getting Help

If you need help, you can:
- Open an issue with the "question" label
- Reach out to the maintainers
- Ask in the community channels

Thank you for contributing to LAION Embeddings!
