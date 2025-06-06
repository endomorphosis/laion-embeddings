#!/usr/bin/env python3
"""
Hanging Point Diagnostic Script
This script identifies exactly where the system is hanging during imports and execution.
"""

import sys
import os
import time
import traceback
import signal
from contextlib import contextmanager

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout protection"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def test_import_step_by_step():
    """Test imports step by step to identify hanging points"""
    print("Testing imports step by step...")
    
    imports_to_test = [
        ("sys", "import sys"),
        ("os", "import os"),
        ("json", "import json"),
        ("time", "import time"),
        ("traceback", "import traceback"),
        ("unittest", "import unittest"),
        ("asyncio", "import asyncio"),
        ("multiprocessing", "import multiprocessing"),
        ("torch", "import torch"),
        ("transformers", "import transformers"),
        ("datasets", "import datasets"),
        ("aiohttp", "import aiohttp"),
        ("numpy", "import numpy as np"),
        ("requests", "import requests"),
    ]
    
    for name, import_stmt in imports_to_test:
        print(f"Testing import: {name}...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            with timeout_context(10):  # 10 second timeout per import
                exec(import_stmt)
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.2f}s)")
        except TimeoutError:
            print(f"✗ TIMEOUT (>10s)")
            return name
        except ImportError as e:
            print(f"✗ ImportError: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return None

def test_main_new_imports():
    """Test main_new.py imports specifically"""
    print("\nTesting main_new.py specific imports...")
    
    # Test each import from main_new.py separately
    main_new_imports = [
        ("ipfs_accelerate_py", "import ipfs_accelerate_py"),
        ("chunker", "import chunker"),
        ("qdrant_kit", "import qdrant_kit"),
        ("elasticsearch_kit", "import elasticsearch_kit"),
        ("faiss_kit", "import faiss_kit"),
    ]
    
    hanging_import = None
    
    for name, import_stmt in main_new_imports:
        print(f"Testing {name}...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            with timeout_context(15):  # 15 second timeout
                exec(import_stmt)
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.2f}s)")
        except TimeoutError:
            print(f"✗ TIMEOUT (>15s) - HANGING POINT FOUND")
            hanging_import = name
            break
        except ImportError as e:
            print(f"✗ ImportError: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return hanging_import

def test_ipfs_embeddings_imports():
    """Test ipfs_embeddings_py imports specifically"""
    print("\nTesting ipfs_embeddings_py imports...")
    
    ipfs_imports = [
        ("ipfs_kit.ipfs_multiformats", "from ipfs_embeddings_py import ipfs_multiformats"),
        ("ipfs_kit.chunker", "from ipfs_embeddings_py import chunker"),
        ("ipfs_kit.ipfs_datasets", "from ipfs_embeddings_py import ipfs_datasets"),
        ("ipfs_kit.main_new", "from ipfs_embeddings_py import main_new"),
        ("ipfs_kit.ipfs_embeddings", "from ipfs_embeddings_py import ipfs_embeddings"),
    ]
    
    hanging_import = None
    
    for name, import_stmt in ipfs_imports:
        print(f"Testing {name}...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            with timeout_context(20):  # 20 second timeout
                exec(import_stmt)
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.2f}s)")
        except TimeoutError:
            print(f"✗ TIMEOUT (>20s) - HANGING POINT FOUND")
            hanging_import = name
            break
        except ImportError as e:
            print(f"✗ ImportError: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return hanging_import

def test_specific_functions():
    """Test specific functions that might hang"""
    print("\nTesting specific functions...")
    
    try:
        print("Importing main_new functions...", end=" ", flush=True)
        with timeout_context(30):
            from ipfs_kit.main import safe_get_cid, safe_tokenizer_encode, index_cid
        print("✓")
        
        # Test safe_get_cid
        print("Testing safe_get_cid...", end=" ", flush=True)
        with timeout_context(5):
            cid = safe_get_cid("test")
        print(f"✓ (result: {cid})")
        
        # Test safe_tokenizer_encode with None
        print("Testing safe_tokenizer_encode with None...", end=" ", flush=True)
        with timeout_context(5):
            result = safe_tokenizer_encode(None, "test")
        print(f"✓ (result: {result})")
        
        # Test index_cid
        print("Testing index_cid...", end=" ", flush=True)
        with timeout_context(5):
            cids = index_cid(["test1", "test2"])
        print(f"✓ (result: {len(cids)} CIDs)")
        
        return True
        
    except TimeoutError as e:
        print(f"✗ TIMEOUT: {e}")
        return False
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def diagnose_hanging_issues():
    """Main diagnostic function"""
    print("=" * 60)
    print("HANGING POINT DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Step 1: Test basic imports
    hanging_basic = test_import_step_by_step()
    if hanging_basic:
        print(f"\nHANGING POINT IDENTIFIED: Basic import '{hanging_basic}' is hanging")
        print("This suggests a system-level issue with the dependency")
        return hanging_basic
    
    # Step 2: Test main_new specific imports
    hanging_main_new = test_main_new_imports()
    if hanging_main_new:
        print(f"\nHANGING POINT IDENTIFIED: main_new import '{hanging_main_new}' is hanging")
        print("This suggests an issue with the custom module dependencies")
        return hanging_main_new
    
    # Step 3: Test ipfs_embeddings imports
    hanging_ipfs = test_ipfs_embeddings_imports()
    if hanging_ipfs:
        print(f"\nHANGING POINT IDENTIFIED: ipfs_embeddings import '{hanging_ipfs}' is hanging")
        print("This suggests an issue with the main library modules")
        return hanging_ipfs
    
    # Step 4: Test specific functions
    functions_ok = test_specific_functions()
    if not functions_ok:
        print("\nHANGING POINT IDENTIFIED: Function execution is hanging")
        return "function_execution"
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE - NO HANGING POINTS FOUND")
    print("=" * 60)
    print("All imports and basic functions completed successfully.")
    print("The hanging issue may be in more complex operations like:")
    print("- Network requests (aiohttp operations)")
    print("- Dataset loading (HuggingFace datasets)")
    print("- Model loading (transformers AutoTokenizer)")
    print("- Async operations without proper timeout")
    
    return None

def get_hanging_recommendations(hanging_point):
    """Get recommendations based on identified hanging point"""
    recommendations = {
        "torch": [
            "Check CUDA installation and GPU availability",
            "Ensure PyTorch is compatible with your system",
            "Try CPU-only PyTorch if GPU issues persist"
        ],
        "transformers": [
            "Check internet connection for model downloads", 
            "Clear HuggingFace cache: ~/.cache/huggingface",
            "Use offline mode or pre-downloaded models"
        ],
        "datasets": [
            "Check internet connection for dataset downloads",
            "Clear datasets cache: ~/.cache/huggingface/datasets", 
            "Use local datasets instead of remote ones"
        ],
        "aiohttp": [
            "Network connectivity issues",
            "Proxy settings may be interfering",
            "Add timeout parameters to all aiohttp operations"
        ],
        "ipfs_accelerate_py": [
            "Custom module may have circular imports",
            "Check for missing dependencies",
            "Review module initialization code"
        ],
        "chunker": [
            "Custom chunker module issues",
            "May depend on tokenizers that hang",
            "Check chunker initialization code"
        ],
        "function_execution": [
            "Functions are hanging during execution",
            "Add timeout protection to all operations",
            "Use mock objects for testing instead of real operations"
        ]
    }
    
    if hanging_point in recommendations:
        print(f"\nRECOMMENDATIONS FOR '{hanging_point}':")
        for rec in recommendations[hanging_point]:
            print(f"- {rec}")
    
    print("\nGENERAL RECOMMENDATIONS:")
    print("- Run tests with timeout protection")
    print("- Use mock objects for network operations")
    print("- Avoid loading large models in tests")
    print("- Add progress indicators to long-running operations")

if __name__ == "__main__":
    try:
        hanging_point = diagnose_hanging_issues()
        if hanging_point:
            get_hanging_recommendations(hanging_point)
            sys.exit(1)
        else:
            print("\n✓ No hanging points detected in basic operations")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nDiagnostic failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
