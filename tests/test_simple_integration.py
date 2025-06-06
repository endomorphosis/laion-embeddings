#!/usr/bin/env python3
"""
Simple Integration Test without complex fixtures

This test verifies basic integration functionality.
"""

import sys
import os
import warnings
from pathlib import Path

# Add the project paths  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "docs" / "ipfs_kit_py"))

def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")
    
    # Test storacha_clusters deprecation
    print("Testing storacha_clusters deprecation...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            from storacha_clusters import storacha_clusters
            print(f"✓ storacha_clusters imported (with {len(w)} warnings)")
            if w:
                print(f"  Warning: {w[0].message}")
        except Exception as e:
            print(f"✗ Error importing storacha_clusters: {e}")
    
    # Test project modules
    print("Testing project modules...")
    try:
        from create_embeddings.create_embeddings import create_embeddings
        print("✓ create_embeddings imported")
    except Exception as e:
        print(f"✗ Error importing create_embeddings: {e}")
    
    try:
        from shard_embeddings.shard_embeddings import shard_embeddings
        print("✓ shard_embeddings imported")
    except Exception as e:
        print(f"✗ Error importing shard_embeddings: {e}")
    
    try:
        from ipfs_cluster_index.ipfs_cluster_index import ipfs_cluster_index
        print("✓ ipfs_cluster_index imported")
    except Exception as e:
        print(f"✗ Error importing ipfs_cluster_index: {e}")

def test_ipfs_kit_py_fallback():
    """Test that we can handle ipfs_kit_py absence gracefully."""
    print("Testing ipfs_kit_py fallback handling...")
    
    # Try different import paths
    ipfs_kit = None
    storacha_kit = None
    
    # Try direct import
    try:
        import ipfs_kit_py
        from ipfs_kit_py import ipfs_kit
        print("✓ ipfs_kit_py imported directly")
    except Exception as e:
        print(f"✗ Direct import failed: {e}")
        
        # Try manual path
        try:
            sys.path.insert(0, str(project_root / "docs" / "ipfs_kit_py" / "ipfs_kit_py"))
            import ipfs_kit
            print("✓ ipfs_kit imported via manual path")
        except Exception as e:
            print(f"✗ Manual path import failed: {e}")
    
    # Try storacha_kit
    try:
        from ipfs_kit_py import storacha_kit
        print("✓ storacha_kit imported")
    except Exception as e:
        print(f"✗ storacha_kit import failed: {e}")
        
        try:
            import storacha_kit
            print("✓ storacha_kit imported via manual path")
        except Exception as e:
            print(f"✗ Manual storacha_kit import failed: {e}")

def test_instantiation():
    """Test instantiation of available classes."""
    print("Testing instantiation...")
    
    resources = {"test": True}
    metadata = {"test_meta": "value"}
    
    # Test create_embeddings
    try:
        from create_embeddings.create_embeddings import create_embeddings
        instance = create_embeddings(resources, metadata)
        print("✓ create_embeddings instantiation successful")
    except Exception as e:
        print(f"✗ create_embeddings instantiation failed: {e}")
    
    # Test shard_embeddings  
    try:
        from shard_embeddings.shard_embeddings import shard_embeddings
        instance = shard_embeddings(resources, metadata)
        print("✓ shard_embeddings instantiation successful")
    except Exception as e:
        print(f"✗ shard_embeddings instantiation failed: {e}")
    
    # Test ipfs_cluster_index
    try:
        from ipfs_cluster_index.ipfs_cluster_index import ipfs_cluster_index
        instance = ipfs_cluster_index(resources, metadata)
        print("✓ ipfs_cluster_index instantiation successful")
    except Exception as e:
        print(f"✗ ipfs_cluster_index instantiation failed: {e}")

def check_legacy_imports():
    """Check for legacy imports in main files."""
    print("Checking for legacy imports...")
    
    legacy_patterns = [
        "import ipfs_embeddings_py",
        "import ipfs_datasets_py", 
        "import ipfs_accelerate_py",
        "from ipfs_embeddings_py",
        "from ipfs_datasets_py",
        "from ipfs_accelerate_py"
    ]
    
    main_files = [
        "main.py",
        "__init__.py", 
        "create_embeddings/create_embeddings.py",
        "shard_embeddings/shard_embeddings.py",
        "ipfs_cluster_index/ipfs_cluster_index.py"
    ]
    
    issues_found = False
    for file_path in main_files:
        full_path = project_root / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
                for pattern in legacy_patterns:
                    if pattern in content and f"# {pattern}" not in content:
                        print(f"✗ Found legacy import '{pattern}' in {file_path}")
                        issues_found = True
    
    if not issues_found:
        print("✓ No legacy imports found in main files")

def check_ipfs_kit_py_imports():
    """Check that ipfs_kit_py imports are present."""
    print("Checking for ipfs_kit_py imports...")
    
    expected_imports = [
        "from ipfs_kit_py import ipfs_kit",
        "from ipfs_kit_py import storacha_kit",
        "ipfs_kit_py"
    ]
    
    main_files = [
        "create_embeddings/create_embeddings.py",
        "shard_embeddings/shard_embeddings.py",
        "ipfs_cluster_index/ipfs_cluster_index.py"
    ]
    
    for file_path in main_files:
        full_path = project_root / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
                has_import = any(import_stmt in content for import_stmt in expected_imports)
                if has_import:
                    print(f"✓ ipfs_kit_py imports found in {file_path}")
                else:
                    print(f"✗ No ipfs_kit_py imports found in {file_path}")

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("IPFS Kit Integration Test Suite")
    print("=" * 60)
    
    test_basic_imports()
    print()
    test_ipfs_kit_py_fallback()
    print()
    test_instantiation()
    print()
    check_legacy_imports()
    print()
    check_ipfs_kit_py_imports()
    print()
    print("=" * 60)
    print("Test suite completed")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
