#!/usr/bin/env python3
"""
Test basic import functionality
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test all critical imports."""
    
    errors = []
    
    print("Testing imports...")
    
    # Test 1: Basic MCP imports
    try:
        from src.mcp_server.tool_registry import ClaudeMCPTool
        print("✓ ClaudeMCPTool imported")
    except Exception as e:
        errors.append(f"ClaudeMCPTool import failed: {e}")
        print(f"✗ ClaudeMCPTool import failed: {e}")
    
    # Test 2: Validators
    try:
        from src.mcp_server.validators import validator
        print("✓ validator imported")
    except Exception as e:
        errors.append(f"validator import failed: {e}")
        print(f"✗ validator import failed: {e}")
    
    # Test 3: Error handlers
    try:
        from src.mcp_server.error_handlers import MCPError, ValidationError
        print("✓ Error handlers imported")
    except Exception as e:
        errors.append(f"Error handlers import failed: {e}")
        print(f"✗ Error handlers import failed: {e}")
    
    # Test 4: Auth tools
    try:
        from src.mcp_server.tools.auth_tools import AuthenticationTool, TokenValidationTool, UserInfoTool
        print("✓ Auth tools imported")
    except Exception as e:
        errors.append(f"Auth tools import failed: {e}")
        print(f"✗ Auth tools import failed: {e}")
    
    # Test 5: Test framework
    try:
        import pytest
        print("✓ pytest available")
    except Exception as e:
        errors.append(f"pytest import failed: {e}")
        print(f"✗ pytest import failed: {e}")
    
    return errors

def main():
    print("=" * 50)
    print("IMPORT DIAGNOSIS")
    print("=" * 50)
    
    errors = test_imports()
    
    if not errors:
        print("\n✅ All imports successful!")
        print("Trying to run a simple test...")
        
        # Try to create a tool instance
        try:
            from src.mcp_server.tools.auth_tools import AuthenticationTool
            tool = AuthenticationTool()
            print(f"✓ Tool created successfully: {tool.name}")
            return 0
        except Exception as e:
            print(f"✗ Tool creation failed: {e}")
            return 1
    else:
        print(f"\n❌ {len(errors)} import errors found:")
        for error in errors:
            print(f"  - {error}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)
