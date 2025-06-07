# LAION Embeddings Package

import os
import sys

# Skip problematic imports during testing to avoid dependency conflicts
TESTING = (
    'pytest' in sys.modules or 
    'PYTEST_CURRENT_TEST' in os.environ or
    any('test' in arg for arg in sys.argv)
)

if not TESTING:
    try:
        from ipfs_kit_py.ipfs_kit import ipfs_kit
    except ImportError:
        ipfs_embeddings_py = None
else:
    ipfs_embeddings_py = None

if not TESTING:
    try:
        from create_embeddings import create_embeddings
    except ImportError:
        create_embeddings = None

    try:
        from search_embeddings import search_embeddings
    except ImportError:
        search_embeddings = None

    try:
        from sparse_embeddings import sparse_embeddings
    except ImportError:
        sparse_embeddings = None

    try:
        # DEPRECATED: storacha_clusters is deprecated, use ipfs_kit_py instead
        from storacha_clusters import storacha_clusters
        import warnings
        warnings.warn("storacha_clusters is deprecated. Use ipfs_kit_py.storacha_kit instead.", DeprecationWarning)
    except ImportError:
        storacha_clusters = None

    try:
        from shard_embeddings import shard_embeddings
    except ImportError:
        shard_embeddings = None

    try:
        from ipfs_cluster_index import ipfs_cluster_index
    except ImportError:
        ipfs_cluster_index = None

    try:
        from test import test
    except ImportError:
        test = None
else:
    # Set all to None during testing
    create_embeddings = None
    search_embeddings = None
    sparse_embeddings = None
    storacha_clusters = None
    shard_embeddings = None
    ipfs_cluster_index = None
    test = None

# Only export modules that were successfully imported
__all__ = [
    name for name in [
        'ipfs_embeddings_py', 'create_embeddings', 'search_embeddings',
        'sparse_embeddings', 'storacha_clusters', 'shard_embeddings',
        'ipfs_cluster_index', 'test'
    ] if globals().get(name) is not None
]
