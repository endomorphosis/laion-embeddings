try:
    import ipfs_embeddings
    from ipfs_embeddings import ipfs_embeddings_py
except:
    from .ipfs_embeddings import ipfs_embeddings_py
try:
    import ipfs_multiformats
    from ipfs_multiformats import ipfs_multiformats_py
except:
    from .ipfs_multiformats import ipfs_multiformats_py
try:
    import qdrant_kit
    from qdrant_kit import qdrant_kit_py
except:
    from .qdrant_kit import qdrant_kit_py
