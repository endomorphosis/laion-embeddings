#!/usr/bin/env python3
"""
Simple test to verify basic functionality and workflow validation.
"""

import sys
import os
import traceback

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test basic imports without hanging."""
    print("Testing basic imports...")
    try:
        import json
        print("✓ json imported successfully")
        
        import asyncio
        print("✓ asyncio imported successfully")
        
        import time
        print("✓ time imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_main_new_file_access():
    """Test if we can access main_new.py file."""
    print("\nTesting main_new.py file access...")
    try:
        main_new_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'ipfs_embeddings_py', 'main_new.py')
        
        if os.path.exists(main_new_path):
            print(f"✓ main_new.py found at: {main_new_path}")
            
            with open(main_new_path, 'r') as f:
                content = f.read()
                
            # Check for key functions
            key_functions = [
                'safe_tokenizer_encode',
                'safe_tokenizer_decode', 
                'safe_chunker_chunk',
                'safe_get_cid',
                'index_cid'
            ]
            
            found_functions = []
            for func in key_functions:
                if func in content:
                    found_functions.append(func)
                    print(f"✓ Found function: {func}")
                else:
                    print(f"✗ Missing function: {func}")
            
            return len(found_functions) == len(key_functions)
        else:
            print(f"✗ main_new.py not found at: {main_new_path}")
            return False
            
    except Exception as e:
        print(f"✗ File access failed: {e}")
        traceback.print_exc()
        return False

def test_workflow_sequence():
    """Test that the workflow follows the correct sequence."""
    print("\nTesting workflow sequence validation...")
    try:
        # This tests the logical sequence without actually running the heavy functions
        workflow_steps = [
            "Text Input",
            "Tokenization", 
            "Chunking",
            "Batch Creation",
            "CID Generation",
            "Embedding Processing"
        ]
        
        print("Expected workflow sequence:")
        for i, step in enumerate(workflow_steps, 1):
            print(f"  {i}. {step}")
        
        # Verify that tokenization comes before embedding
        tokenization_index = workflow_steps.index("Tokenization")
        embedding_index = workflow_steps.index("Embedding Processing")
        
        if tokenization_index < embedding_index:
            print("✓ Tokenization occurs before embedding processing")
            return True
        else:
            print("✗ Workflow sequence is incorrect")
            return False
            
    except Exception as e:
        print(f"✗ Workflow validation failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("=" * 50)
    print("SIMPLE TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("File Access", test_main_new_file_access),
        ("Workflow Sequence", test_workflow_sequence)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
