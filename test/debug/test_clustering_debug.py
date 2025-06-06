#!/usr/bin/env python3
"""
Debug script to test clustering service manually
"""

import os
import sys
import numpy as np

# Set testing mode
os.environ['TESTING'] = 'true'

# Add current directory to Python path
sys.path.insert(0, '.')

try:
    from services.clustering_service import VectorClusterer, ClusterConfig
    print("✅ Successfully imported clustering service")
    
    # Test 1: Basic initialization
    config = ClusterConfig(n_clusters=3)
    clusterer = VectorClusterer(config)
    print(f"✅ Initialization successful")
    print(f"   cluster_centers: {clusterer.cluster_centers}")
    print(f"   cluster_metadata: {clusterer.cluster_metadata}")
    print(f"   is_fitted: {clusterer.is_fitted}")
    
    # Test 2: fit_kmeans with sample vectors
    vectors = np.random.rand(15, 3).astype(np.float32)
    print(f"✅ Created sample vectors: shape {vectors.shape}")
    
    labels = clusterer.fit_kmeans(vectors)
    print(f"✅ fit_kmeans successful")
    print(f"   labels shape: {labels.shape}")
    print(f"   labels: {labels}")
    print(f"   cluster_centers shape: {clusterer.cluster_centers.shape}")
    print(f"   cluster_metadata length: {len(clusterer.cluster_metadata)}")
    print(f"   cluster_metadata keys: {list(clusterer.cluster_metadata.keys())}")
    
    # Test 3: Verify test expectations
    print("\n🔍 Verifying test expectations:")
    print(f"   len(labels) == len(vectors): {len(labels) == len(vectors)}")
    print(f"   cluster_centers.shape == (3, 3): {clusterer.cluster_centers.shape == (3, 3)}")
    print(f"   len(cluster_metadata) == 3: {len(clusterer.cluster_metadata) == 3}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
