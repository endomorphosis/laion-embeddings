"""
Provider implementations for vector stores.
This package contains concrete implementations of the BaseVectorStore interface.
"""

# Import providers that are available
__all__ = []

# Always available
try:
    from .faiss_store import FAISSVectorStore
    __all__.append('FAISSVectorStore')
except ImportError:
    pass

# Optional providers (require specific dependencies)
try:
    from .qdrant_store import QdrantVectorStore
    __all__.append('QdrantVectorStore')
except ImportError:
    pass

try:
    from .elasticsearch_store import ElasticsearchVectorStore
    __all__.append('ElasticsearchVectorStore')
except ImportError:
    pass

try:
    from .pgvector_store import PgVectorStore
    __all__.append('PgVectorStore')
except ImportError:
    pass

# New providers for IPFS and DuckDB
try:
    from .ipfs_store import IPFSVectorStore
    __all__.append('IPFSVectorStore')
except ImportError:
    pass

try:
    from .duckdb_store import DuckDBVectorStore
    __all__.append('DuckDBVectorStore')
except ImportError:
    pass
