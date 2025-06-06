#!/usr/bin/env python3
"""Debug script to investigate sklearn availability issue"""

import sys
import os

# Set testing environment variable 
os.environ['TESTING'] = 'true'

def check_sklearn_import():
    """Check sklearn import in different contexts"""
    print("=== SKLEARN IMPORT DEBUG ===")
    
    print(f"Python path: {sys.path[:3]}...")
    print(f"TESTING env var: {os.environ.get('TESTING', 'not set')}")
    
    # Try direct import
    try:
        import sklearn
        print(f"✓ Direct sklearn import successful, version: {sklearn.__version__}")
        print(f"  sklearn location: {sklearn.__file__}")
    except ImportError as e:
        print(f"✗ Direct sklearn import failed: {e}")
        return False
    
    # Try specific submodules
    try:
        from sklearn.cluster import KMeans
        print("✓ sklearn.cluster.KMeans import successful")
    except ImportError as e:
        print(f"✗ sklearn.cluster.KMeans import failed: {e}")
    
    try:
        from sklearn.cluster import AgglomerativeClustering
        print("✓ sklearn.cluster.AgglomerativeClustering import successful")
    except ImportError as e:
        print(f"✗ sklearn.cluster.AgglomerativeClustering import failed: {e}")
    
    # Check how the services module handles this
    print("\n=== SERVICES MODULE CHECK ===")
    try:
        # Add current directory to path
        if '/home/barberb/laion-embeddings-1' not in sys.path:
            sys.path.insert(0, '/home/barberb/laion-embeddings-1')
        
        from services.clustering_service import SKLEARN_AVAILABLE
        print(f"SKLEARN_AVAILABLE from clustering_service: {SKLEARN_AVAILABLE}")
        
        # Try importing the clusterer directly
        from services.clustering_service import VectorClusterer
        clusterer = VectorClusterer()
        print(f"VectorClusterer created successfully")
        
    except Exception as e:
        print(f"Error importing from services: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def check_transformers_import():
    """Check transformers import issues"""
    print("\n=== TRANSFORMERS IMPORT DEBUG ===")
    
    try:
        import transformers
        print(f"✓ transformers import successful, version: {transformers.__version__}")
        print(f"  transformers location: {transformers.__file__}")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    try:
        from transformers.models.auto import AutoModel
        print("✓ transformers.models.auto.AutoModel import successful")
    except ImportError as e:
        print(f"✗ transformers.models.auto.AutoModel import failed: {e}")
        
    try:
        from transformers import AutoModel
        print("✓ transformers.AutoModel import successful")
    except ImportError as e:
        print(f"✗ transformers.AutoModel import failed: {e}")
    
    return True

if __name__ == "__main__":
    print("Starting sklearn and transformers debug check...")
    check_sklearn_import()
    check_transformers_import()
    print("Debug check complete.")
