#!/usr/bin/env python3
"""
Basic Integration Test
"""

print("Starting IPFS Kit Integration Test")

try:
    from ipfs_kit_py import storacha_kit
    print("✓ storacha_clusters imported successfully")
except Exception as e:
    print(f"✗ Error importing storacha_clusters: {e}")

try:
    from create_embeddings.create_embeddings import create_embeddings
    print("✓ create_embeddings imported successfully")
except Exception as e:
    print(f"✗ Error importing create_embeddings: {e}")

try:
    from shard_embeddings.shard_embeddings import shard_embeddings
    print("✓ shard_embeddings imported successfully")  
except Exception as e:
    print(f"✗ Error importing shard_embeddings: {e}")

try:
    from ipfs_cluster_index.ipfs_cluster_index import ipfs_cluster_index
    print("✓ ipfs_cluster_index imported successfully")
except Exception as e:
    print(f"✗ Error importing ipfs_cluster_index: {e}")

print("Test completed")
