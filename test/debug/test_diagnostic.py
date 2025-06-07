#!/usr/bin/env python3
"""
Test diagnostic script - runs individual tests one by one
to identify which ones are failing and why.
"""

import os
import sys
import subprocess
import importlib
import inspect
import pytest

def discover_tests(test_path):
    """Discover test functions in the given path."""
    if not test_path.endswith('.py'):
        print(f"Error: {test_path} is not a Python file")
        return []

    try:
        # Convert file path to module path
        module_path = test_path.replace('/', '.').replace('.py', '')
        
        # Import the module
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            print(f"Error: Cannot find module {module_path}")
            return []
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find test functions and classes
        test_items = []
        
        for name, obj in inspect.getmembers(module):
            # Find test functions
            if name.startswith('test_') and callable(obj):
                test_items.append(f"{module_path}::{name}")
            
            # Find test classes
            elif inspect.isclass(obj) and name.startswith('Test'):
                for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if method_name.startswith('test_'):
                        test_items.append(f"{module_path}::{name}::{method_name}")
        
        return test_items
    except Exception as e:
        print(f"Error discovering tests in {test_path}: {e}")
        return []

def run_test(test_path):
    """Run a single test and return success/failure."""
    os.environ["TESTING"] = "true"
    cmd = ["python", "-m", "pytest", "-v", test_path]
    
    print(f"\n{'=' * 60}")
    print(f"Running: {test_path}")
    print(f"{'=' * 60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ PASSED: {test_path}")
        return True
    else:
        print(f"❌ FAILED: {test_path}")
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return False

def run_fixed_tests():
    """Run the specific tests that were fixed."""
    fixed_tests = [
        "test/ipfs/test_ipfs_fixed.py",
        "test/test_clustering_service.py::TestVectorClusterer::test_get_cluster_stats",
        "test/test_clustering_service.py::TestVectorClusterer::test_sklearn_import_error",
        "test/test_clustering_service.py::TestSmartShardingService::test_create_clustered_shards"
    ]
    
    passed = 0
    failed = 0
    
    for test_path in fixed_tests:
        if run_test(test_path):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"FIXED TESTS SUMMARY: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    
    return failed == 0

def main():
    """Main function to run diagnostic tests."""
    if "--fixed" in sys.argv or "-f" in sys.argv:
        success = run_fixed_tests()
        sys.exit(0 if success else 1)
    elif len(sys.argv) < 2:
        print("Usage: python test_diagnostic.py <test_file_path> or --fixed")
        sys.exit(1)
    
    test_path = sys.argv[1]
    
    # Check if it's a specific test or a file
    if "::" in test_path:
        # It's a specific test
        success = run_test(test_path)
        sys.exit(0 if success else 1)
    else:
        # It's a file, discover tests
        test_items = discover_tests(test_path)
        
        if not test_items:
            print(f"No tests found in {test_path}")
            sys.exit(1)
        
        print(f"Found {len(test_items)} tests in {test_path}")
        
        passed = 0
        failed = 0
        
        for item in test_items:
            if run_test(item):
                passed += 1
            else:
                failed += 1
        
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print(f"{'=' * 60}")
        
        sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
