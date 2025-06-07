#!/usr/bin/env python3
"""
Simple test runner for validating the vector services implementation.

This script runs the essential validation tests to ensure that the vector quantization,
clustering, and sharding implementation is working correctly.
"""

import sys
import subprocess
from pathlib import Path
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def run_validation():
    """Run the validation test suite."""
    print("🚀 Running LAION Embeddings Vector Services Validation Tests")
    print("=" * 70)
    
    try:
        # Run the validation test
        result = subprocess.run([
            sys.executable, 
            "test/validation_test.py"
        ], capture_output=True, text=True, cwd=project_root)
        
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Validation tests completed successfully!")
        else:
            print(f"\n❌ Validation tests failed with exit code {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running validation tests: {e}")
        return False

def run_pytest_tests():
    """Run the comprehensive pytest test suite."""
    print("\n🧪 Running Comprehensive Test Suite with pytest")
    print("=" * 70)
    
    # Test files to run
    test_files = [
        "test/test_vector_service.py",
        "test/test_ipfs_vector_service.py", 
        "test/test_clustering_service.py",
        "test/test_complete_integration.py"
    ]
    
    # Filter existing test files
    existing_files = [f for f in test_files if Path(f).exists()]
    
    if not existing_files:
        print("❌ No test files found")
        return False
    
    try:
        # Run pytest with the test files
        cmd = [
            sys.executable, "-m", "pytest",
            "-v",
            "--tb=short",
            "--asyncio-mode=auto"
        ] + existing_files
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print("\n✅ pytest tests completed successfully!")
        else:
            print(f"\n❌ pytest tests failed with exit code {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running pytest tests: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking Dependencies")
    print("=" * 70)
    
    dependencies = [
        ("numpy", "Vector operations"),
        ("pytest", "Testing framework"),
        ("asyncio", "Async support"),
    ]
    
    optional_dependencies = [
        ("faiss", "FAISS vector search"),
        ("sklearn", "Clustering algorithms"),
        ("ipfshttpclient", "IPFS client")
    ]
    
    all_good = True
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✅ {module}: {description}")
        except ImportError:
            print(f"❌ {module}: {description} - MISSING (required)")
            all_good = False
    
    print("\nOptional dependencies:")
    for module, description in optional_dependencies:
        try:
            __import__(module)
            print(f"✅ {module}: {description}")
        except ImportError:
            print(f"⚠️  {module}: {description} - not available (optional)")
    
    return all_good

def main():
    """Main test runner function."""
    print("LAION Embeddings - Vector Services Test Runner")
    print("=" * 70)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Missing required dependencies. Install with:")
        print("pip install -r requirements.txt")
        return 1
    
    print()
    
    # Run validation tests
    validation_success = run_validation()
    
    # Run pytest tests if available
    pytest_success = True
    try:
        import pytest
        pytest_success = run_pytest_tests()
    except ImportError:
        print("\n⚠️  pytest not available, skipping comprehensive tests")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if validation_success:
        print("✅ Basic validation tests: PASSED")
    else:
        print("❌ Basic validation tests: FAILED")
    
    if pytest_success:
        print("✅ Comprehensive tests: PASSED")
    else:
        print("❌ Comprehensive tests: FAILED")
    
    overall_success = validation_success and pytest_success
    
    if overall_success:
        print("\n🎉 All tests passed! The implementation is working correctly.")
        print("\nNext steps:")
        print("1. Install optional dependencies for full functionality:")
        print("   pip install faiss-cpu scikit-learn ipfshttpclient")
        print("2. Run the services in your application")
        print("3. Monitor performance and optimize as needed")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Check that all dependencies are installed")
        print("2. Review error messages for specific issues") 
        print("3. Run individual test files for more detailed output")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
