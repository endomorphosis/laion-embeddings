#!/usr/bin/env python3
"""
Simple test runner to check individual test files
"""
import os
import sys

# Add the project root to Python path
project_root = '/home/barberb/laion-embeddings-1'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_imports():
    """Test basic imports"""
    try:
        import pytest
        print("✓ pytest imported successfully")
    except ImportError as e:
        print(f"✗ pytest import failed: {e}")
        return False
    
    try:
        from src.mcp_server.tools.auth_tools import AuthenticationTool
        print("✓ AuthenticationTool imported successfully")
    except ImportError as e:
        print(f"✗ AuthenticationTool import failed: {e}")
        return False
        
    try:
        from tests.test_mcp_tools.conftest import mock_request
        print("✓ mock_request imported successfully")
    except ImportError as e:
        print(f"✗ mock_request import failed: {e}")
        return False
    
    return True

def run_single_test():
    """Run a single test to check functionality"""
    try:
        # Import necessary modules
        from src.mcp_server.tools.auth_tools import AuthenticationTool
        from tests.test_mcp_tools.conftest import mock_request
        
        print("\n=== Running single test ===")
        
        # Create tool instance
        tool = AuthenticationTool()
        print(f"✓ Created AuthenticationTool: {tool}")
        
        # Test with mock request
        request = mock_request()
        result = tool.authenticate_user("test_user", "test_password", request)
        print(f"✓ Authentication result: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Single test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Basic Functionality ===")
    
    if test_imports():
        print("\n=== All imports successful ===")
        if run_single_test():
            print("\n=== Single test successful ===")
        else:
            print("\n=== Single test failed ===")
    else:
        print("\n=== Import tests failed ===")
