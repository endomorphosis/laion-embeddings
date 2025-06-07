#!/usr/bin/env python3
"""
Comprehensive test runner to identify and report all pytest errors.
"""
import subprocess
import sys
import os
import json
from pathlib import Path

def run_pytest_with_collection():
    """Run pytest with collection to identify all test discovery issues."""
    
    # Change to project directory
    os.chdir('/home/barberb/laion-embeddings-1')
    
    results = {
        "collection_errors": [],
        "import_errors": [],
        "test_results": [],
        "summary": {}
    }
    
    try:
        # First, try to collect tests
        print("=== Running pytest --collect-only ===")
        collect_result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/', '--collect-only', '-q'
        ], capture_output=True, text=True, timeout=120)
        
        results["collection_stdout"] = collect_result.stdout
        results["collection_stderr"] = collect_result.stderr
        results["collection_returncode"] = collect_result.returncode
        
        if collect_result.returncode != 0:
            print(f"Collection failed with return code: {collect_result.returncode}")
            print("STDERR:", collect_result.stderr)
            results["collection_errors"].append({
                "type": "collection_failure",
                "returncode": collect_result.returncode,
                "stderr": collect_result.stderr
            })
        
        # Try to run a specific test file
        print("\n=== Running single test file ===")
        single_test_result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_mcp_tools/test_auth_tools.py', '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=120)
        
        results["single_test_stdout"] = single_test_result.stdout
        results["single_test_stderr"] = single_test_result.stderr
        results["single_test_returncode"] = single_test_result.returncode
        
        # Try importing individual modules
        print("\n=== Testing imports ===")
        import_tests = [
            "src.mcp_server.tools.auth_tools",
            "src.mcp_server.tools.analysis_tools",
            "src.mcp_server.error_handlers",
            "src.mcp_server.validators",
            "tests.test_mcp_tools.conftest"
        ]
        
        for module_name in import_tests:
            try:
                __import__(module_name)
                results["import_errors"].append({
                    "module": module_name,
                    "status": "success"
                })
                print(f"✓ {module_name}")
            except Exception as e:
                results["import_errors"].append({
                    "module": module_name,
                    "status": "error",
                    "error": str(e)
                })
                print(f"✗ {module_name}: {e}")
        
    except subprocess.TimeoutExpired:
        results["timeout"] = True
        print("Test execution timed out")
    except Exception as e:
        results["execution_error"] = str(e)
        print(f"Execution error: {e}")
    
    # Save results to file
    with open('comprehensive_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results saved to comprehensive_test_results.json ===")
    
    return results

if __name__ == "__main__":
    run_pytest_with_collection()
