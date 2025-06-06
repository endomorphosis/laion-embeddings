#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Python path:")
for p in sys.path[:5]:  # First 5 entries
    print(f"  {p}")

print("\n--- Testing imports ---")

try:
    from src.mcp_server.tools.sparse_embedding_tools import (
        SparseEmbeddingGenerationTool,
        SparseIndexingTool,
        SparseSearchTool
    )
    print("✓ Successfully imported sparse embedding tools")
    
    # Test instantiation
    tool = SparseEmbeddingGenerationTool()
    print(f"✓ Successfully created {tool.name} tool")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Other error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Testing pytest discovery ---")
try:
    import pytest
    print("✓ pytest available")
    
    # Check if test file can be discovered
    test_file = "test/unit/test_mcp_sparse_embedding_tools.py"
    if os.path.exists(test_file):
        print(f"✓ Test file exists: {test_file}")
    else:
        print(f"✗ Test file not found: {test_file}")
        
except ImportError:
    print("✗ pytest not available")
