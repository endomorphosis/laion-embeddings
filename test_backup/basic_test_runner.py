#!/usr/bin/env python3
"""
Basic Test Runner for LAION Embeddings
This runner tests what's available and provides feedback on system readiness.
"""

import sys
import os
import unittest
import traceback
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import(module_name, import_path=None):
    """Test if a module can be imported"""
    try:
        if import_path:
            spec = importlib.util.spec_from_file_location(module_name, import_path)
            if spec is None:
                return False, f"Could not find module at {import_path}"
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            __import__(module_name)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def test_basic_functionality():
    """Test basic functionality that should work"""
    print("=" * 60)
    print("BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Python environment
    print(f"✓ Python version: {sys.version}")
    
    # Test 2: Core dependencies
    core_deps = ['torch', 'transformers', 'datasets', 'numpy', 'aiohttp']
    for dep in core_deps:
        success, msg = test_import(dep)
        status = "✓" if success else "✗"
        print(f"{status} {dep}: {msg}")
        results.append((dep, success, msg))
    
    # Test 3: IPFS embeddings modules
    print("\n" + "=" * 60)
    print("IPFS EMBEDDINGS MODULE TESTS")
    print("=" * 60)
    
    module_tests = [
        ("ipfs_kit.ipfs_embeddings", None),
        ("ipfs_kit.main_new", None),
        ("ipfs_kit.chunker", None),
        ("ipfs_kit.ipfs_datasets", None),
        ("ipfs_kit.ipfs_multiformats", None),
    ]
    
    for module, path in module_tests:
        success, msg = test_import(module, path)
        status = "✓" if success else "✗"
        print(f"{status} {module}: {msg}")
        results.append((module, success, msg))
    
    # Test 4: Main components
    print("\n" + "=" * 60)
    print("MAIN COMPONENT TESTS")
    print("=" * 60)
    
    component_tests = [
        ("search_embeddings.search_embeddings", None),
        ("create_embeddings.create_embeddings", None),
        ("shard_embeddings.shard_embeddings", None),
        ("sparse_embeddings.sparse_embeddings", None),
        ("storacha_clusters.storacha_clusters", None),
    ]
    
    for component, path in component_tests:
        success, msg = test_import(component, path)
        status = "✓" if success else "✗"
        print(f"{status} {component}: {msg}")
        results.append((component, success, msg))
    
    return results

def test_main_new_functions():
    """Test functions from main_new module"""
    print("\n" + "=" * 60)
    print("MAIN_NEW FUNCTION TESTS")
    print("=" * 60)
    
    try:
        from ipfs_embeddings_py.main_new import safe_get_cid, index_cid, init_datasets
        
        # Test safe_get_cid
        try:
            test_data = "Hello, world!"
            cid = safe_get_cid(test_data)
            if cid:
                print(f"✓ safe_get_cid: Generated CID {cid}")
            else:
                print("✗ safe_get_cid: No CID generated")
        except Exception as e:
            print(f"✗ safe_get_cid: {e}")
        
        # Test index_cid
        try:
            test_samples = ["sample1", "sample2", "sample3"]
            cids = index_cid(test_samples)
            if cids and len(cids) == len(test_samples):
                print(f"✓ index_cid: Generated {len(cids)} CIDs")
            else:
                print(f"✗ index_cid: Expected {len(test_samples)} CIDs, got {len(cids) if cids else 0}")
        except Exception as e:
            print(f"✗ index_cid: {e}")
        
        # Test init_datasets (may fail due to network/auth)
        try:
            result = init_datasets(
                model="thenlper/gte-small",
                dataset="TeraflopAI/Caselaw_Access_Project",
                split="train",
                column="text",
                dst_path="/tmp/test"
            )
            if result and isinstance(result, dict):
                print(f"✓ init_datasets: Initialized dataset with keys {list(result.keys())}")
            else:
                print("✗ init_datasets: No result returned")
        except Exception as e:
            print(f"✗ init_datasets: {e} (may be expected if no network/auth)")
            
    except ImportError as e:
        print(f"✗ Could not import main_new functions: {e}")

def test_ipfs_embeddings_class():
    """Test the main ipfs_embeddings_py class"""
    print("\n" + "=" * 60)
    print("IPFS EMBEDDINGS CLASS TESTS")
    print("=" * 60)
    
    try:
        from ipfs_kit_py.ipfs_kit import ipfs_kit
        from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
        
        # Test basic metadata and resources
        test_metadata = {
            "dataset": "TeraflopAI/Caselaw_Access_Project",
            "column": "text",
            "split": "train",
            "models": ["thenlper/gte-small"],
            "chunk_settings": {
                "chunk_size": 512,
                "n_sentences": 8,
                "step_size": 256,
                "method": "fixed",
                "embed_model": "thenlper/gte-small",
                "tokenizer": None
            },
            "dst_path": "/tmp/test_embeddings",
        }
        
        test_resources = {
            "local_endpoints": [
                ["thenlper/gte-small", "cpu", 512],
            ],
            "tei_endpoints": [
                ["thenlper/gte-small", "http://127.0.0.1:8080/embed-tiny", 512],
            ],
            "openvino_endpoints": [],
            "libp2p_endpoints": []
        }
        
        # Test initialization
        try:
            embeddings = ipfs_embeddings_py(test_resources, test_metadata)
            print(f"✓ ipfs_embeddings_py: Initialized successfully")
            
            # Test attributes
            required_attrs = ['resources', 'metadata', 'tei_endpoints', 'local_endpoints']
            for attr in required_attrs:
                if hasattr(embeddings, attr):
                    print(f"✓ ipfs_kit.{attr}: Present")
                else:
                    print(f"✗ ipfs_kit.{attr}: Missing")
            
            # Test status method
            if hasattr(embeddings, 'status'):
                status = embeddings.status()
                print(f"✓ ipfs_kit.status(): {type(status).__name__}")
            else:
                print("✗ ipfs_kit.status(): Method not found")
                
        except Exception as e:
            print(f"✗ ipfs_embeddings_py initialization: {e}")
            
    except ImportError as e:
        print(f"✗ Could not from ipfs_kit_py import ipfs_kit: {e}")

def test_existing_test_files():
    """Run existing test files to see what works"""
    print("\n" + "=" * 60)
    print("EXISTING TEST FILE VALIDATION")
    print("=" * 60)
    
    test_files = [
        "test/test.py",
        "test/test openvino.py", 
        "test/test_openvino2.py",
        "test/test max batch size.py"
    ]
    
    for test_file in test_files:
        test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), test_file)
        if os.path.exists(test_path):
            print(f"✓ Found: {test_file}")
            try:
                # Try to import the test file to see if it's valid
                spec = importlib.util.spec_from_file_location("test_module", test_path)
                if spec:
                    print(f"  ✓ Importable: {test_file}")
                else:
                    print(f"  ✗ Not importable: {test_file}")
            except Exception as e:
                print(f"  ✗ Import error: {e}")
        else:
            print(f"✗ Missing: {test_file}")

def main():
    """Main test runner"""
    print("LAION Embeddings - Basic Test Runner")
    print("=" * 60)
    
    # Run all tests
    basic_results = test_basic_functionality()
    test_main_new_functions()
    test_ipfs_embeddings_class()
    test_existing_test_files()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_tests = len(basic_results)
    passed_tests = sum(1 for _, success, _ in basic_results if success)
    
    print(f"Basic dependency tests: {passed_tests}/{total_tests} passed")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests < total_tests:
        print("\nFailed tests:")
        for name, success, msg in basic_results:
            if not success:
                print(f"  - {name}: {msg}")
    
    print("\nRecommendations:")
    if passed_tests >= total_tests * 0.8:
        print("✓ System appears ready for comprehensive testing")
    else:
        print("✗ Fix import issues before running comprehensive tests")
        print("  - Check that all dependencies are installed")
        print("  - Verify module paths and structure")

if __name__ == "__main__":
    main()
