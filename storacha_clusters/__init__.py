"""DEPRECATED: Storacha Clusters Module

This module is deprecated and has been replaced by ipfs_kit_py.

Use ipfs_kit_py.storacha_kit instead.
See docs/IPFS_KIT_INTEGRATION_GUIDE.md for migration instructions.
"""

import warnings

warnings.warn(
    "storacha_clusters module is deprecated. Use ipfs_kit_py.storacha_kit instead. "
    "See docs/IPFS_KIT_INTEGRATION_GUIDE.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

from .storacha_clusters import storacha_clusters

__all__ = ['storacha_clusters']
