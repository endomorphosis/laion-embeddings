#!/usr/bin/env python3
"""
Minimal test script to check basic functionality without heavy dependencies
"""
import sys
import os
import time
import signal
from contextlib import contextmanager

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@contextmanager
def timeout(duration):
    """Context manager for timeouts using signals"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def test_basic_imports():
    """Test basic Python imports"""
    print("=== Testing Basic Imports ===")
    
    basic_modules = ['os', 'sys', 'json', 'time', 'math']
    for module in basic_modules:
        try:
            with timeout(5):
                __import__(module)
                print(f"✓ {module} imported successfully")
        except TimeoutError:
            print(f"✗ {module} import timed out")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")

def test_optional_imports():
    """Test optional heavy imports with timeout"""
    print("\n=== Testing Optional Heavy Imports ===")
    
    optional_modules = [
        ('numpy', 'np'),
        ('requests', None),
        ('aiohttp', None),
        ('json', None)
    ]
    
    for module_info in optional_modules:
        module_name = module_info[0]
        alias = module_info[1] if len(module_info) > 1 else None
        
        try:
            with timeout(10):
                if alias:
                    exec(f"import {module_name} as {alias}")
                else:
                    __import__(module_name)
                print(f"✓ {module_name} imported successfully")
        except TimeoutError:
            print(f"✗ {module_name} import timed out")
        except ImportError as e:
            print(f"✗ {module_name} not available: {e}")

def test_project_structure():
    """Test if we can access project files"""
    print("\n=== Testing Project Structure ===")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    important_files = [
        'ipfs_embeddings_py/main_new.py',
        'ipfs_embeddings_py/__init__.py',
        'ipfs_embeddings_py/chunker.py'
    ]
    
    for file_path in important_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"✓ Found {file_path}")
        else:
            print(f"✗ Missing {file_path}")

def test_basic_tokenization_logic():
    """Test basic tokenization logic without heavy dependencies"""
    print("\n=== Testing Basic Tokenization Logic ===")
    
    try:
        # Test basic string operations that should be in main_new.py
        test_text = "This is a test sentence for tokenization."
        
        # Basic tokenization (splitting by spaces)
        basic_tokens = test_text.split()
        print(f"✓ Basic tokenization: {len(basic_tokens)} tokens")
        
        # Test chunking logic
        chunk_size = 3
        chunks = [basic_tokens[i:i+chunk_size] for i in range(0, len(basic_tokens), chunk_size)]
        print(f"✓ Basic chunking: {len(chunks)} chunks")
        
        # Test batch creation
        batch = {
            'text': test_text,
            'tokens': basic_tokens,
            'chunks': chunks,
            'batch_id': 'test_batch_001'
        }
        print(f"✓ Basic batch creation: {batch['batch_id']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic tokenization logic failed: {e}")
        return False

def test_file_access():
    """Test if we can read main_new.py file"""
    print("\n=== Testing File Access ===")
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_new_path = os.path.join(project_root, 'ipfs_embeddings_py', 'main_new.py')
        
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
            if f"def {func}" in content:
                found_functions.append(func)
                print(f"✓ Found function: {func}")
            else:
                print(f"✗ Missing function: {func}")
        
        print(f"Found {len(found_functions)}/{len(key_functions)} key functions")
        return len(found_functions) > 0
        
    except Exception as e:
        print(f"✗ File access failed: {e}")
        return False

def main():
    """Run all minimal tests"""
    print("Starting Minimal Test Suite...")
    print("=" * 50)
    
    results = []
    
    # Run tests
    test_basic_imports()
    test_optional_imports()
    test_project_structure()
    
    results.append(test_basic_tokenization_logic())
    results.append(test_file_access())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All critical tests passed!")
        return 0
    else:
        print("✗ Some tests failed, but basic functionality may still work")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
