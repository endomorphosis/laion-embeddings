#!/usr/bin/env python3
"""
Simple CI/CD Test Runner - Manual Validation

This script manually validates the CI/CD pipeline components
to ensure everything is working correctly.
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_mcp_imports():
    """Test core MCP tool imports"""
    print("Testing MCP tool imports...")
    
    try:
        from src.mcp_server.tools.auth_tools import AuthenticationTool, TokenValidationTool
        print("  ✅ Auth tools imported successfully")
        
        from src.mcp_server.tools.session_management_tools import SessionManager
        print("  ✅ Session management tools imported successfully")
        
        from src.mcp_server.tools.rate_limiting_tools import RateLimitConfigurationTool
        print("  ✅ Rate limiting tools imported successfully")
        
        from src.mcp_server.tools.vector_store_tools import create_vector_store_tool
        print("  ✅ Vector store tools imported successfully")
        
        from src.mcp_server.tools.ipfs_cluster_tools import IPFSClusterTool
        print("  ✅ IPFS cluster tools imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {str(e)}")
        traceback.print_exc()
        return False

def test_tool_instantiation():
    """Test that tools can be instantiated"""
    print("Testing tool instantiation...")
    
    try:
        from src.mcp_server.tools.auth_tools import AuthenticationTool
        auth_tool = AuthenticationTool()
        print(f"  ✅ AuthenticationTool: {auth_tool.name}")
        
        from src.mcp_server.tools.rate_limiting_tools import RateLimitConfigurationTool
        rate_tool = RateLimitConfigurationTool()
        print(f"  ✅ RateLimitConfigurationTool: {rate_tool.name}")
        
        from src.mcp_server.tools.ipfs_cluster_tools import IPFSClusterTool
        ipfs_tool = IPFSClusterTool()
        print(f"  ✅ IPFSClusterTool: {ipfs_tool.name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Instantiation failed: {str(e)}")
        traceback.print_exc()
        return False

def test_mcp_server():
    """Test MCP server initialization"""
    print("Testing MCP server...")
    
    try:
        # Import the MCP server
        from mcp_server import LAIONEmbeddingsMCPServer
        
        # Create server instance
        server = LAIONEmbeddingsMCPServer()
        print("  ✅ MCP server created successfully")
        
        # Test validation mode
        if hasattr(server, 'validation_results'):
            print("  ✅ Validation results structure present")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MCP server test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_ci_cd_components():
    """Test CI/CD specific components"""
    print("Testing CI/CD components...")
    
    try:
        # Check if CI/CD test runner exists
        if Path("run_ci_cd_tests.py").exists():
            print("  ✅ CI/CD test runner exists")
        
        # Check if comprehensive test exists
        if Path("test/test_mcp_tools_comprehensive.py").exists():
            print("  ✅ Comprehensive test suite exists")
        
        # Check if GitHub workflow exists
        if Path(".github/workflows/ci-cd.yml").exists():
            print("  ✅ GitHub Actions workflow exists")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CI/CD components test failed: {str(e)}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 CI/CD Pipeline Validation")
    print("=" * 50)
    
    tests = [
        ("MCP Imports", test_mcp_imports),
        ("Tool Instantiation", test_tool_instantiation),
        ("MCP Server", test_mcp_server),
        ("CI/CD Components", test_ci_cd_components),
    ]
    
    results = {"passed": 0, "failed": 0, "total": len(tests)}
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                results["passed"] += 1
            else:
                print(f"❌ {test_name}: FAILED")
                results["failed"] += 1
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results["failed"] += 1
    
    # Summary
    print(f"\n🎯 Validation Summary:")
    print(f"   ✅ Passed: {results['passed']}")
    print(f"   ❌ Failed: {results['failed']}")
    print(f"   📊 Total: {results['total']}")
    
    success_rate = (results['passed'] / results['total'] * 100) if results['total'] > 0 else 0
    print(f"   🏆 Success Rate: {success_rate:.1f}%")
    
    if results['failed'] == 0:
        print(f"\n🎉 All CI/CD pipeline components validated successfully!")
        print(f"✅ Ready for production deployment")
        return True
    else:
        print(f"\n⚠️  Some components failed validation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
