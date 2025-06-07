#!/usr/bin/env python3
"""
Quick MCP Tools Validation Script

This script provides a fast validation of core MCP tools functionality
for use in CI/CD pipelines. It focuses on core imports and basic instantiation
without complex dependencies.

Usage:
    python tools/validation/mcp_tools_quick_validation.py
    
Exit codes:
    0: All validations passed
    1: Some validations failed
"""

import sys
import json
from pathlib import Path

def main():
    """Run quick validation of MCP tools"""
    print("🔧 MCP Tools Quick Validation")
    print("=" * 50)
    
    # Add project paths
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    # Track validation results
    results = {
        "imports": False,
        "instantiation": 0,
        "total_tools": 4,
        "critical_files": False
    }
    
    # Test 1: Core imports
    print("\n1. Testing core imports...")
    try:
        from src.mcp_server.tools.auth_tools import AuthenticationTool, TokenValidationTool
        from src.mcp_server.tools.session_management_tools import SessionManager
        from src.mcp_server.tools.rate_limiting_tools import RateLimitConfigurationTool
        from src.mcp_server.tools.vector_store_tools import create_vector_store_tool
        from src.mcp_server.tools.ipfs_cluster_tools import IPFSClusterTool
        print("   ✅ All core imports successful")
        results["imports"] = True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return 1
    
    # Test 2: Basic instantiation
    print("\n2. Testing basic tool instantiation...")
    tools_tested = []
    
    try:
        auth_tool = AuthenticationTool()
        tools_tested.append(f"AuthenticationTool ({auth_tool.name})")
        results["instantiation"] += 1
    except Exception as e:
        print(f"   ❌ AuthenticationTool failed: {e}")
    
    try:
        token_tool = TokenValidationTool()
        tools_tested.append(f"TokenValidationTool ({token_tool.name})")
        results["instantiation"] += 1
    except Exception as e:
        print(f"   ❌ TokenValidationTool failed: {e}")
    
    try:
        session_mgr = SessionManager()
        tools_tested.append("SessionManager")
        results["instantiation"] += 1
    except Exception as e:
        print(f"   ❌ SessionManager failed: {e}")
    
    try:
        rate_tool = RateLimitConfigurationTool()
        tools_tested.append(f"RateLimitConfigurationTool ({rate_tool.name})")
        results["instantiation"] += 1
    except Exception as e:
        print(f"   ❌ RateLimitConfigurationTool failed: {e}")
    
    for tool in tools_tested:
        print(f"   ✅ {tool}")
    
    # Test 3: Critical files
    print("\n3. Checking critical files...")
    critical_files = [
        "src/mcp_server/tools/auth_tools.py",
        "src/mcp_server/tools/session_management_tools.py",
        "src/mcp_server/tools/rate_limiting_tools.py", 
        "src/mcp_server/tools/vector_store_tools.py",
        "src/mcp_server/tools/ipfs_cluster_tools.py",
        "mcp_server_enhanced.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file_path in critical_files:
        full_path = project_root / file_path
        if not full_path.exists() or not full_path.is_file():
            missing_files.append(file_path)
    
    if not missing_files:
        print("   ✅ All critical files exist")
        results["critical_files"] = True
    else:
        print(f"   ❌ Missing files: {missing_files}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Results Summary:")
    print(f"   • Core Imports: {'✅ PASS' if results['imports'] else '❌ FAIL'}")
    print(f"   • Tools Instantiated: {results['instantiation']}/{results['total_tools']}")
    print(f"   • Critical Files: {'✅ PASS' if results['critical_files'] else '❌ FAIL'}")
    
    # Overall status
    success_rate = results['instantiation'] / results['total_tools']
    overall_pass = (results['imports'] and 
                   results['critical_files'] and 
                   success_rate >= 0.75)  # 75% success rate minimum
    
    if overall_pass:
        print(f"   • Overall Status: ✅ PASS ({success_rate:.0%} success rate)")
        print("\n🎉 MCP Tools validation successful!")
        print("✅ Ready for CI/CD pipeline execution")
        
        # Output JSON for CI/CD consumption
        output = {
            "status": "success",
            "validation": results,
            "success_rate": success_rate,
            "tools_validated": tools_tested
        }
        print(f"\n📄 JSON Output: {json.dumps(output, indent=2)}")
        return 0
    else:
        print(f"   • Overall Status: ❌ FAIL ({success_rate:.0%} success rate)")
        print("\n⚠️ MCP Tools validation failed!")
        print("🔧 Please check the issues above before CI/CD")
        
        # Output JSON for CI/CD consumption
        output = {
            "status": "failed",
            "validation": results,
            "success_rate": success_rate,
            "missing_files": missing_files if missing_files else None
        }
        print(f"\n📄 JSON Output: {json.dumps(output, indent=2)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
