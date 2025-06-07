"""
DEPRECATED: storacha_clusters module

This module is deprecated and has been replaced by ipfs_kit_py.

Migration Guide:
---------------

OLD (deprecated):
    from storacha_clusters import storacha_clusters
    storacha = storacha_clusters(resources, metadata)
    
NEW (recommended):
    from ipfs_kit_py.storacha_kit import storacha_kit
    from ipfs_kit_py.ipfs_kit import ipfs_kit
    
    # For Storacha functionality:
    storacha = storacha_kit(api_key="your_api_key")
    
    # For general IPFS functionality:
    ipfs = ipfs_kit(resources, metadata)

Features mapping:
-----------------
- storacha_clusters.test() -> storacha_kit.validate_connection()
- storacha_clusters.kmeans_cluster_split -> ipfs_kit.faiss_kit.kmeans_cluster_split_dataset
- All IPFS operations -> ipfs_kit methods
- All S3 operations -> ipfs_kit.s3_kit methods
- All embedding operations -> ipfs_kit.ipfs_embeddings methods

For more information, see: docs/IPFS_KIT_INTEGRATION_GUIDE.md
"""

import warnings
import logging

logger = logging.getLogger(__name__)

class storacha_clusters:
    """DEPRECATED: Use ipfs_kit_py.storacha_kit instead."""
    
    def __init__(self, resources=None, metadata=None):
        warnings.warn(
            "storacha_clusters is deprecated. Use ipfs_kit_py.storacha_kit instead. "
            "See docs/IPFS_KIT_INTEGRATION_GUIDE.md for migration instructions.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(
            "storacha_clusters is deprecated. Use ipfs_kit_py.storacha_kit instead."
        )
        
        # Initialize with None to prevent usage
        self.resources = resources
        self.metadata = metadata
        self.ipfs_kit_py = None
        self.s3_kit_py = None
        self.ipfs_embeddings_py = None
        self.ipfs_parquet_to_car = None
        self.kmeans_cluster_split = None
        
    def test(self):
        """DEPRECATED: Use storacha_kit.validate_connection() instead."""
        raise DeprecationWarning(
            "storacha_clusters.test() is deprecated. "
            "Use ipfs_kit_py.storacha_kit.validate_connection() instead."
        )
        
    def __getattr__(self, name):
        """Catch all attribute access and show deprecation warning."""
        raise AttributeError(
            f"storacha_clusters.{name} is deprecated. "
            f"Use ipfs_kit_py instead. See docs/IPFS_KIT_INTEGRATION_GUIDE.md"
        )

if __name__ == "__main__":
    print("ERROR: storacha_clusters is deprecated.")
    print("Use ipfs_kit_py instead.")
    print("See docs/IPFS_KIT_INTEGRATION_GUIDE.md for migration instructions.")
    sys.exit(1)
