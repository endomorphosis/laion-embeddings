#!/usr/bin/env python3
"""
Simple test runner to identify pytest issues
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def run_simple_test():
    """Run a simple import test."""
    print("Testing basic imports...")
    
    try:
        # Test auth tools import
        from src.mcp_server.tools.auth_tools import AuthenticationTool, TokenValidationTool, UserInfoTool
        print("✓ Auth tools imported successfully")
        
        # Test error handlers import
        from src.mcp_server.error_handlers import MCPError, ValidationError
        print("✓ Error handlers imported successfully")
        
        # Test that we can create a tool instance
        tool = AuthenticationTool()
        print(f"✓ AuthenticationTool created: {tool.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_pytest():
    """Run pytest and capture output."""
    try:
        # Try to run pytest on a specific test file
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_mcp_tools/test_auth_tools.py",
            "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=60)
        
        print("PYTEST STDOUT:")
        print(result.stdout)
        print("\nPYTEST STDERR:")
        print(result.stderr)
        print(f"\nPYTEST EXIT CODE: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Pytest timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Failed to run pytest: {e}")
        return False

def main():
    print("=" * 60)
    print("PYTEST ERROR DIAGNOSIS")
    print("=" * 60)
    
    # Test 1: Basic imports
    print("\n1. Testing basic imports...")
    import_success = run_simple_test()
    
    if not import_success:
        print("Basic imports failed. Fixing imports first...")
        return 1
    
    # Test 2: Run pytest
    print("\n2. Running pytest...")
    pytest_success = run_pytest()
    
    if pytest_success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
