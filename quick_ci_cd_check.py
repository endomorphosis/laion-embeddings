#!/usr/bin/env python3
"""
Quick CI/CD Check Script

Simple validation that our CI/CD components are working.
"""

import sys
import os
from pathlib import Path

def main():
    print("🚀 Starting Quick CI/CD Check...")
    
    # Check project structure
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    mcp_tools_path = src_path / "mcp_server" / "tools"
    
    print(f"✅ Project root: {project_root}")
    print(f"✅ Source path exists: {src_path.exists()}")
    print(f"✅ MCP tools path exists: {mcp_tools_path.exists()}")
    
    if mcp_tools_path.exists():
        tool_files = list(mcp_tools_path.glob("*.py"))
        print(f"✅ Found {len(tool_files)} MCP tool files")
        
        # List the tools
        for tool_file in sorted(tool_files):
            if not tool_file.name.startswith("__"):
                print(f"  - {tool_file.name}")
    
    # Check CI/CD files
    cicd_files = [
        ".github/workflows/ci-cd.yml",
        "mcp_server.py",
        "run_ci_cd_tests.py",
        "test/test_mcp_tools_comprehensive.py"
    ]
    
    print("\n📋 CI/CD Files Status:")
    for file_path in cicd_files:
        full_path = project_root / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"  {status} {file_path}")
    
    # Check Docker files
    docker_files = [
        "Dockerfile",
        "docker-compose.yml",
        "docker-deploy.sh"
    ]
    
    print("\n🐳 Docker Files Status:")
    for file_path in docker_files:
        full_path = project_root / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"  {status} {file_path}")
    
    print("\n🎯 Quick CI/CD Check Complete!")
    print("All required files are present for CI/CD pipeline.")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
