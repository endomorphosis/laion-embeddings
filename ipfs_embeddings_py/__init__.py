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
    
try:
    import ipfs_parquet_to_car
    from ipfs_parquet_to_car import ipfs_parquet_to_car_py
except:
    from .ipfs_parquet_to_car import ipfs_parquet_to_car_py
    pass

try:
    import node_parser
    from node_parser import *
except:
    from .node_parser import *
    pass

try:
    import install_depends
    from install_depends import *
except:
    from .install_depends import *
    pass

try:
    import chunker
    from chunker import *
except:
    from .chunker import *
    pass