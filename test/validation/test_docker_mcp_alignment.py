#!/usr/bin/env python3
"""
Test script to verify Docker configurations align with CI/CD MCP server approach.
This ensures the same mcp_server.py entrypoint and arguments are used consistently.
"""

import subprocess
import sys
import json
import time
from pathlib import Path

def run_command(cmd, timeout=30):
    """Run a command with timeout"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_mcp_server_validation():
    """Test MCP server validation (same as CI/CD)"""
    print("🔍 Testing MCP server validation (CI/CD approach)...")
    
    success, stdout, stderr = run_command("python3 mcp_server.py --validate", timeout=15)
    
    if success:
        print("✅ MCP server validation passed")
        try:
            # Try to parse JSON output
            if stdout.strip():
                validation_data = json.loads(stdout.strip())
                print(f"   📊 Tools count: {validation_data.get('tools_count', 'unknown')}")
                print(f"   📈 Status: {validation_data.get('status', 'unknown')}")
            return True
        except json.JSONDecodeError:
            print("⚠️  Validation passed but output not JSON formatted")
            return True
    else:
        print(f"❌ MCP server validation failed: {stderr}")
        return False

def test_docker_file_consistency():
    """Test Dockerfile configuration consistency"""
    print("\n🔍 Testing Dockerfile configuration...")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return False
    
    content = dockerfile_path.read_text()
    
    # Check for correct CMD
    if 'CMD ["python3", "mcp_server.py"]' in content:
        print("✅ Dockerfile CMD uses correct mcp_server.py entrypoint")
    else:
        print("❌ Dockerfile CMD does not match CI/CD approach")
        return False
    
    # Check for validation in healthcheck
    if "mcp_server.py --validate" in content:
        print("✅ Dockerfile HEALTHCHECK uses --validate (same as CI/CD)")
    else:
        print("❌ Dockerfile HEALTHCHECK does not use validation")
        return False
    
    # Check for virtual environment setup
    if "/opt/venv/bin" in content:
        print("✅ Dockerfile sets up virtual environment correctly")
    else:
        print("⚠️  Virtual environment setup not found in Dockerfile")
    
    return True

def test_docker_compose_consistency():
    """Test docker-compose.yml configuration consistency"""
    print("\n🔍 Testing docker-compose.yml configuration...")
    
    compose_path = Path("docker-compose.yml")
    if not compose_path.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    content = compose_path.read_text()
    
    # Check healthcheck
    if 'python3", "mcp_server.py", "--validate"' in content:
        print("✅ docker-compose healthcheck uses --validate (same as CI/CD)")
    else:
        print("❌ docker-compose healthcheck does not match CI/CD")
        return False
    
    # Check container name
    if "laion-embeddings-mcp-server" in content:
        print("✅ Container name properly set")
    else:
        print("⚠️  Container name not found")
    
    return True

def test_deployment_script_consistency():
    """Test docker-deploy.sh script consistency"""
    print("\n🔍 Testing docker-deploy.sh script...")
    
    script_path = Path("docker-deploy.sh")
    if not script_path.exists():
        print("❌ docker-deploy.sh not found")
        return False
    
    content = script_path.read_text()
    
    # Check validation approach
    if "mcp_server.py --validate" in content:
        print("✅ docker-deploy.sh uses same validation as CI/CD")
    else:
        print("❌ docker-deploy.sh does not use CI/CD validation approach")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 Testing Docker-CI/CD MCP Server Alignment")
    print("=" * 50)
    
    tests = [
        ("MCP Server Validation", test_mcp_server_validation),
        ("Dockerfile Consistency", test_docker_file_consistency),
        ("Docker Compose Consistency", test_docker_compose_consistency),
        ("Deployment Script Consistency", test_deployment_script_consistency),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🏆 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Docker configurations align with CI/CD approach.")
        return True
    else:
        print("⚠️  Some tests failed. Docker configurations need adjustment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
